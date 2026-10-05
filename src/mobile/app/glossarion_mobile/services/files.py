"""FileBridge: bring files into app storage and hand outputs back to the user (plan §4, design §5.2).

Import
  ``FilePicker.pick_files`` returns a cache copy on Android/iOS (the app has no
  storage permissions; the picker goes through SAF / UIDocumentPicker). Each
  file is copied (in a worker thread) into ``<data>/Inbox`` or, for "Add to
  Library", into ``Library/Raw`` and registered with the shared
  ``library_core.record_library_raw_input``. Names are kept, so the translation
  output folder keeps the book's name; a different file with the same name gets
  a `` (2)`` suffix (the Library's own ``_unique_dest`` scheme), while the same
  content imported again reuses the existing copy, so re-importing a book
  resumes its translation. Cache copies under the app's cache/temp folders are
  removed after the copy; a path outside them (desktop dev) is never touched.

Folders
  ``FilePicker.get_directory_path`` + a recursive copy into ``Inbox/<folder>``.
  Android usually cannot read a picked tree path (SAF); then
  ``FolderPickUnavailable`` tells the UI to offer "Select files" or "Pick a
  .zip" (ZIP/CBZ inputs are supported by the backend).

Export
  ``Share.share_files`` · ``FilePicker.save_file(src_bytes=...)`` (confirm
  above 200 MB; the bytes are read in a worker thread) · Android "Save to
  Downloads/Glossarion" (``GlossarionNative.save_to_downloads``, MediaStore)
  · iOS "Show in Files" (``shareddocuments://``, Documents is Files-visible).

Blocking work never runs on the UI loop: the sync methods are called through
``run_io`` (``UiDispatcher.run_in_thread`` in the app). Flet objects are
created lazily by factories, so this module imports without Flet.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import mimetypes
import os
import re
import shutil
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Iterable, Optional, Sequence

__all__ = [
    "BOOK_EXTENSIONS",
    "ExportOption",
    "FileBridge",
    "FolderPickUnavailable",
    "IMPORT_EXTENSIONS",
    "ImportedFile",
    "ImportedFolder",
    "LIBRARY_EXTENSIONS",
    "SAVE_CONFIRM_BYTES",
    "SaveResult",
    "mime_type_for",
    "safe_name",
    "same_content",
    "unique_destination",
]

log = logging.getLogger("glossarion.files")

#: Desktop input filter (translator_gui Browse + the Direct Text attachment extras).
IMPORT_EXTENSIONS = (
    "epub", "zip", "cbz", "pdf", "txt", "json", "csv", "md", "sdlxliff", "xliff", "srt", "ass", "lrc", "vtt",
    "html", "htm", "xhtml", "png", "jpg", "jpeg", "gif", "bmp", "webp", "mp4",
)
#: What the Library accepts as raw books (epub_library ``_import_paths`` raw target).
LIBRARY_EXTENSIONS = (".epub", ".txt", ".pdf", ".html", ".htm")
#: Inputs the chat opens as a book (Plan card).
BOOK_EXTENSIONS = (".epub", ".pdf", ".txt", ".md", ".html", ".htm", ".xhtml", ".zip", ".cbz")
SAVE_CONFIRM_BYTES = 200 * 1024 * 1024
_COPY_CHUNK = 1024 * 1024
_MAX_NAME = 180
_RESERVED = re.compile(r'[\x00-\x1f<>:"/\\|?*]')


class FolderPickUnavailable(Exception):
    """The platform picker cannot hand over a readable folder; offer files or a ZIP instead."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason
        self.fallbacks = ("files", "zip")


@dataclass(frozen=True)
class ImportedFile:
    path: str  # the copy in app storage
    name: str
    size: int
    source: str  # the picker / share path it came from
    target: str  # "inbox" | "library"
    reused: bool = False  # identical content was already there

    @property
    def extension(self) -> str:
        return os.path.splitext(self.name)[1].lower()


@dataclass(frozen=True)
class ImportedFolder:
    path: str
    name: str
    files: int
    size: int


@dataclass(frozen=True)
class ExportOption:
    id: str  # "share" | "save" | "downloads" | "files"
    label: str
    icon: str
    disabled_reason: Optional[str] = None


@dataclass(frozen=True)
class SaveResult:
    ok: bool
    needs_confirm: bool = False
    location: Optional[str] = None
    error: Optional[str] = None
    size: int = 0


def safe_name(name: Any, fallback: str = "file") -> str:
    """A file name safe in app storage (no folders, control or reserved characters)."""
    text = os.path.basename(str(name or "").replace("\\", "/"))
    text = _RESERVED.sub("_", text).strip().strip(".")
    if not text:
        text = fallback
    stem, ext = os.path.splitext(text)
    if len(text) > _MAX_NAME:
        text = stem[: _MAX_NAME - len(ext)] + ext
    return text


def unique_destination(directory: str, name: str) -> str:
    """``directory/name``, or ``name (2)``, ``name (3)``... when taken (files and folders)."""
    candidate = os.path.join(directory, name)
    if not os.path.exists(candidate):
        return candidate
    stem, ext = os.path.splitext(name)
    if os.path.isdir(candidate):
        stem, ext = name, ""
    counter = 2
    while True:
        candidate = os.path.join(directory, f"{stem} ({counter}){ext}")
        if not os.path.exists(candidate):
            return candidate
        counter += 1


def _sha1(path: str) -> str:
    digest = hashlib.sha1()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(_COPY_CHUNK)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def same_content(a: str, b: str) -> bool:
    try:
        if os.path.getsize(a) != os.path.getsize(b):
            return False
        return _sha1(a) == _sha1(b)
    except OSError:
        return False


def mime_type_for(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    known = {
        ".epub": "application/epub+zip",
        ".pdf": "application/pdf",
        ".txt": "text/plain",
        ".md": "text/markdown",
        ".csv": "text/csv",
        ".json": "application/json",
        ".html": "text/html",
        ".htm": "text/html",
        ".xhtml": "application/xhtml+xml",
        ".zip": "application/zip",
        ".cbz": "application/vnd.comicbook+zip",
        ".srt": "application/x-subrip",
        ".sdlxliff": "application/xliff+xml",
    }
    return known.get(ext) or mimetypes.guess_type(path)[0] or "application/octet-stream"


def _under(path: str, roots: Iterable[str]) -> bool:
    try:
        real = os.path.realpath(path)
    except OSError:
        return False
    for root in roots:
        if not root:
            continue
        try:
            base = os.path.realpath(root)
        except OSError:
            continue
        try:
            if os.path.commonpath([real, base]) == base:
                return True
        except ValueError:  # different drives
            continue
    return False


RunIo = Callable[..., Awaitable[Any]]


async def _default_run_io(fn: Callable[..., Any], *args: Any) -> Any:
    return await asyncio.to_thread(fn, *args)


class FileBridge:
    def __init__(
        self,
        *,
        inbox_dir: str,
        library_raw_dir: Any = None,
        record_library_input: Optional[Callable[[str], Any]] = None,
        cache_dirs: Sequence[str] = (),
        platform: str = "desktop",
        picker_factory: Optional[Callable[[], Any]] = None,
        share_factory: Optional[Callable[[], Any]] = None,
        url_launcher: Any = None,
        native: Any = None,
        run_io: Optional[RunIo] = None,
        files_visible_root: Optional[str] = None,
    ) -> None:
        self.inbox_dir = os.fspath(inbox_dir)
        self._library_raw_dir = library_raw_dir
        self._record_library_input = record_library_input
        self.cache_dirs = [os.fspath(d) for d in cache_dirs if d]
        self.platform = platform
        self._picker_factory = picker_factory
        self._share_factory = share_factory
        self._picker: Any = None
        self._share: Any = None
        self.url_launcher = url_launcher
        self.native = native
        self.run_io: RunIo = run_io or _default_run_io
        self.files_visible_root = files_visible_root

    # ---- locations ------------------------------------------------------------------------

    def library_raw_dir(self) -> str:
        target = self._library_raw_dir
        if callable(target):
            target = target()
        if target is None:
            import library_core  # shared (Library registry)

            target = library_core.get_library_raw_dir()
        os.makedirs(target, exist_ok=True)
        return os.fspath(target)

    def _record(self, path: str) -> None:
        record = self._record_library_input
        if record is None:
            try:
                import library_core

                record = library_core.record_library_raw_input
            except ImportError:
                log.warning("library_core unavailable; %s is not registered in the Library", path)
                return
        try:
            record(path)
        except Exception:
            log.exception("registering %s in the Library failed", path)

    def is_cache_copy(self, path: str) -> bool:
        """Only picker/share copies inside the app's cache or temp folders may be removed."""
        return self.platform in ("android", "ios") and _under(path, self.cache_dirs)

    # ---- import (blocking; call through run_io) ------------------------------------------------

    def _copy_into(self, source: str, directory: str, name: Optional[str] = None) -> tuple[str, bool]:
        os.makedirs(directory, exist_ok=True)
        name = safe_name(name or os.path.basename(source))
        existing = os.path.join(directory, name)
        if os.path.isfile(existing):
            if os.path.realpath(existing) == os.path.realpath(source) or same_content(existing, source):
                return existing, True
        dest = unique_destination(directory, name)
        tmp = dest + ".part"
        try:
            with open(source, "rb") as src, open(tmp, "wb") as dst:
                shutil.copyfileobj(src, dst, _COPY_CHUNK)
            os.replace(tmp, dest)
        finally:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass
        try:
            shutil.copystat(source, dest)
        except OSError:
            pass
        return dest, False

    def import_paths(self, paths: Iterable[Any], *, target: str = "inbox",
                     names: Optional[Sequence[Optional[str]]] = None) -> list[ImportedFile]:
        """Copy files into the Inbox (or Library/Raw) and return what landed where."""
        names = list(names or [])
        out: list[ImportedFile] = []
        directory = self.library_raw_dir() if target == "library" else self.inbox_dir
        for index, raw in enumerate(paths):
            if not raw:
                continue
            source = os.fspath(raw)
            if not os.path.isfile(source):
                log.warning("import skipped, not a file: %s", source)
                continue
            display = names[index] if index < len(names) and names[index] else os.path.basename(source)
            dest, reused = self._copy_into(source, directory, display)
            if target == "library":
                self._record(dest)
            if self.is_cache_copy(source) and os.path.realpath(source) != os.path.realpath(dest):
                try:
                    os.remove(source)
                except OSError:
                    pass
            out.append(ImportedFile(path=dest, name=os.path.basename(dest), size=os.path.getsize(dest),
                                    source=source, target=target, reused=reused))
        return out

    def add_to_library(self, path: str) -> ImportedFile:
        """Copy an imported file into Library/Raw and register it (Add to Library)."""
        if os.path.splitext(path)[1].lower() not in LIBRARY_EXTENSIONS:
            raise ValueError("Only EPUB, TXT, PDF and HTML files go to the Library")
        imported = self.import_paths([path], target="library")
        if not imported:
            raise FileNotFoundError(path)
        return imported[0]

    def import_folder(self, folder: str, *, target_dir: Optional[str] = None) -> ImportedFolder:
        """Recursive copy of a picked folder into ``Inbox/<name>`` (suffixed when taken)."""
        if not folder or not os.path.isdir(folder):
            raise FolderPickUnavailable("The folder could not be read")
        try:
            os.listdir(folder)
        except OSError as exc:
            raise FolderPickUnavailable(f"The folder could not be read ({exc.__class__.__name__})") from exc
        base = target_dir or self.inbox_dir
        os.makedirs(base, exist_ok=True)
        dest = unique_destination(base, safe_name(os.path.basename(os.path.normpath(folder)), "Folder"))
        files = size = 0
        for root, dirs, names in os.walk(folder):
            dirs[:] = [d for d in dirs if not d.startswith(".")]
            rel = os.path.relpath(root, folder)
            out_dir = dest if rel == "." else os.path.join(dest, rel)
            os.makedirs(out_dir, exist_ok=True)
            for name in names:
                src = os.path.join(root, name)
                if not os.path.isfile(src):
                    continue
                shutil.copy2(src, os.path.join(out_dir, safe_name(name)))
                files += 1
                size += os.path.getsize(src)
        return ImportedFolder(path=dest, name=os.path.basename(dest), files=files, size=size)

    # ---- pickers (UI loop) ----------------------------------------------------------------------

    def _get_picker(self) -> Any:
        if self._picker is None:
            if self._picker_factory is not None:
                self._picker = self._picker_factory()
            else:
                import flet as ft

                self._picker = ft.FilePicker()  # page service: keep the reference
        return self._picker

    def _get_share(self) -> Any:
        if self._share is None:
            if self._share_factory is not None:
                self._share = self._share_factory()
            else:
                import flet as ft

                self._share = ft.Share()
        return self._share

    async def pick_files(self, *, target: str = "inbox", allowed_extensions: Optional[Sequence[str]] = None,
                         allow_multiple: bool = True, dialog_title: Optional[str] = None) -> list[ImportedFile]:
        """FilePicker -> copies in the Inbox / Library (empty list when cancelled)."""
        picker = self._get_picker()
        kwargs: dict = {"allow_multiple": allow_multiple, "dialog_title": dialog_title}
        # The desktop filter list on desktop; the mobile pickers map extensions to MIME types / UTIs
        # and reject unknown ones (sdlxliff, lrc, ass...), so they show every file unless a caller
        # asks for specific extensions.
        extensions = list(allowed_extensions) if allowed_extensions is not None else (
            None if self.platform in ("android", "ios") else list(IMPORT_EXTENSIONS))
        if extensions:
            try:
                import flet as ft

                kwargs["file_type"] = ft.FilePickerFileType.CUSTOM
            except ImportError:
                pass
            kwargs["allowed_extensions"] = extensions
        picked = await picker.pick_files(**kwargs) or []
        paths = [getattr(f, "path", None) for f in picked]
        names = [getattr(f, "name", None) for f in picked]
        if any(p is None for p in paths):
            log.warning("%d picked file(s) came without a path", sum(1 for p in paths if p is None))
        pairs = [(p, n) for p, n in zip(paths, names) if p]
        if not pairs:
            return []
        return await self.run_io(lambda: self.import_paths([p for p, _ in pairs], target=target,
                                                           names=[n for _, n in pairs]))

    async def pick_folder(self, *, dialog_title: Optional[str] = None) -> ImportedFolder:
        """Folder picker -> ``Inbox/<folder>``; ``FolderPickUnavailable`` on Android SAF trees."""
        picker = self._get_picker()
        try:
            folder = await picker.get_directory_path(dialog_title=dialog_title)
        except Exception as exc:
            raise FolderPickUnavailable(f"Folder picking is not available here ({exc.__class__.__name__})") from exc
        if not folder:
            raise FolderPickUnavailable("No folder was chosen")
        if folder.startswith("content:") or not os.path.isdir(folder):
            raise FolderPickUnavailable("Android does not let apps read a picked folder directly")
        return await self.run_io(self.import_folder, folder)

    # ---- export ---------------------------------------------------------------------------------------

    def export_options(self, path: str) -> list[ExportOption]:
        options = [ExportOption("share", "Share…", "IOS_SHARE"), ExportOption("save", "Save to…", "SAVE_ALT")]
        if self.platform == "android":
            options.append(ExportOption("downloads", "Save to Downloads", "DOWNLOAD"))
        else:
            options.append(ExportOption("downloads", "Save to Downloads", "DOWNLOAD",
                                        disabled_reason="Android only"))
        if self.platform == "ios":
            visible = self.files_visible_root and _under(path, [self.files_visible_root])
            options.append(ExportOption("files", "Show in Files", "FOLDER_OPEN",
                                        None if visible else "Only for files in Glossarion's Documents"))
        return options

    async def share(self, paths: Sequence[str], *, text: Optional[str] = None, title: Optional[str] = None) -> bool:
        files = [p for p in paths if p and os.path.isfile(p)]
        if not files:
            return False
        share = self._get_share()
        try:
            import flet as ft

            items = [ft.ShareFile.from_path(p, name=os.path.basename(p)) for p in files]
        except ImportError:
            items = list(files)
        try:
            await share.share_files(items, text=text, title=title)
        except Exception as exc:
            log.warning("share failed: %s", exc)
            return False
        return True

    async def save_as(self, path: str, *, confirmed: bool = False) -> SaveResult:
        """FilePicker.save_file with the file's bytes (mobile needs ``src_bytes``)."""
        try:
            size = os.path.getsize(path)
        except OSError as exc:
            return SaveResult(False, error=f"File not found ({exc.__class__.__name__})")
        if size > SAVE_CONFIRM_BYTES and not confirmed:
            return SaveResult(False, needs_confirm=True, size=size)

        def read() -> bytes:
            with open(path, "rb") as handle:
                return handle.read()

        data = await self.run_io(read)
        picker = self._get_picker()
        try:
            location = await picker.save_file(file_name=os.path.basename(path), src_bytes=data)
        except Exception as exc:
            return SaveResult(False, error=str(exc), size=size)
        if not location and self.platform in ("android", "ios"):
            # The mobile pickers write src_bytes themselves; None means cancelled.
            return SaveResult(False, size=size)
        if location and self.platform not in ("android", "ios") and not os.path.exists(location):
            def write() -> None:
                with open(location, "wb") as handle:
                    handle.write(data)

            await self.run_io(write)
        return SaveResult(bool(location), location=location, size=size)

    async def save_to_downloads(self, path: str) -> Optional[str]:
        if self.platform != "android" or self.native is None:
            return None
        return await self.native.call("save_to_downloads", path, os.path.basename(path), mime_type_for(path),
                                      "Glossarion", default=None)

    async def show_in_files(self, path: str) -> bool:
        if self.platform != "ios" or self.url_launcher is None:
            return False
        folder = path if os.path.isdir(path) else os.path.dirname(path)
        try:
            await self.url_launcher.launch_url("shareddocuments://" + folder)
        except Exception as exc:
            log.info("show in files failed: %s", exc)
            return False
        return True

    async def export(self, option_id: str, path: str, *, confirmed: bool = False) -> Any:
        if option_id == "share":
            return await self.share([path])
        if option_id == "save":
            return await self.save_as(path, confirmed=confirmed)
        if option_id == "downloads":
            return await self.save_to_downloads(path)
        if option_id == "files":
            return await self.show_in_files(path)
        raise ValueError(option_id)

