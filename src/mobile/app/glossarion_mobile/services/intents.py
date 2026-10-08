"""IntentRouter: Open-with / Share / iOS "Open in" -> import -> action sheet (plan §4, UI_SPEC §7.6).

Sources: ``GlossarionNative.get_initial_shared()`` at launch and ``on_share``
events afterwards. Shared files arrive as copies in the app's cache
(``cacheDir/shared`` / ``tmp/shared``); each one is imported into the Inbox
through ``FileBridge`` (the cache copy is removed) and then offered:

* **Translate in new chat** (the chat's handler opens a chat with the file attached);
* **Add to Library** (copy into ``Library/Raw``, then the shared
  ``library_core.import_paths`` registers it and scaffolds its workspace, through
  ``FileBridge.library_import``; EPUB / TXT / PDF / HTML only; the Library feature
  adds a handler that also offers "Open" on the new book);
* **Open in Reader** (EPUB / TXT; the Library feature's handler routes to
  ``/reader/<bid>`` with the file's opaque id; disabled while no handler is
  registered);
* **Manga translator** (images, CBZ and ZIP only: the manga feature's handler adds
  the file to Tools › Manga › Files; disabled while no handler is registered).

Shared text (and http(s) links) prefill the composer. Nothing here is ever
routed: ``glossarion://`` launch links are left to the app's router
(``router.launch_links``), and ``content:`` / ``file:`` URIs are dropped, never
turned into routes. Items are de-duplicated by their native id (the initial
items and a later ``share`` event can repeat the same item).

UI-free: the app passes ``present(imports)`` to show the sheet and handlers
per action.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional, Sequence

from glossarion_mobile.services.files import LIBRARY_EXTENSIONS, FileBridge, ImportedFile

__all__ = [
    "ACTION_ADD_TO_LIBRARY",
    "ACTION_COMPOSE",
    "ACTION_MANGA",
    "ACTION_OPEN_IN_READER",
    "ACTION_TRANSLATE_NEW_CHAT",
    "IntentAction",
    "IntentImport",
    "IntentRouter",
    "MANGA_EXTENSIONS",
    "READER_EXTENSIONS",
]

log = logging.getLogger("glossarion.intents")

ACTION_TRANSLATE_NEW_CHAT = "translate_new_chat"
ACTION_ADD_TO_LIBRARY = "add_to_library"
ACTION_OPEN_IN_READER = "open_in_reader"
ACTION_COMPOSE = "compose"
ACTION_MANGA = "manga"

_BLOCKED_SCHEMES = ("content:", "file:", "intent:", "data:", "javascript:")
READER_REASON = "The Reader is not available in this session"
READER_TYPES_REASON = "The Reader opens EPUB and TXT files"
#: Shared files the Reader opens straight from the Inbox.
READER_EXTENSIONS = (".epub", ".txt")
LIBRARY_REASON = "Only EPUB, TXT, PDF and HTML files go to the Library"
#: Shared files the manga translator takes (its Files tab: images, CBZ, ZIP).
MANGA_EXTENSIONS = (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".cbz", ".zip")
MANGA_REASON = "The manga translator is not available in this session"


@dataclass(frozen=True)
class IntentAction:
    id: str
    label: str
    icon: str
    disabled_reason: Optional[str] = None


@dataclass
class IntentImport:
    item: Mapping[str, Any]
    imported: Optional[ImportedFile] = None
    text: Optional[str] = None
    error: Optional[str] = None
    actions: list = field(default_factory=list)

    @property
    def is_file(self) -> bool:
        return self.imported is not None

    @property
    def label(self) -> str:
        if self.imported is not None:
            return self.imported.name
        if self.text:
            return self.text[:60]
        return str(self.item.get("name") or "Shared item")


Handler = Callable[[IntentImport], Any]


class IntentRouter:
    def __init__(
        self,
        *,
        files: FileBridge,
        native: Any = None,
        present: Optional[Callable[[list], Any]] = None,
        handlers: Optional[Mapping[str, Handler]] = None,
        notify: Optional[Callable[[str], Any]] = None,
    ) -> None:
        self.files = files
        self.native = native
        self.present = present
        self.handlers: dict = dict(handlers or {})
        self.notify = notify
        self._seen: set[str] = set()
        self.history: list[IntentImport] = []

    # ---- classification ---------------------------------------------------------------------

    @staticmethod
    def _kind(item: Mapping[str, Any]) -> str:
        return str(item.get("kind") or ("file" if item.get("path") else "text"))

    def accepts(self, item: Any) -> bool:
        """Whether ``item`` is ours (files, text, web links); launch links belong to the router."""
        if not isinstance(item, Mapping):
            return False
        kind = self._kind(item)
        if kind == "url":
            text = str(item.get("text") or "").strip().lower()
            if item.get("source") == "launch" or text.startswith("glossarion:"):
                return False
            return text.startswith(("http://", "https://"))
        if kind == "text":
            text = str(item.get("text") or "").strip()
            return bool(text) and not text.lower().startswith(_BLOCKED_SCHEMES)
        return kind == "file" and bool(item.get("path"))

    def actions_for(self, imp: IntentImport) -> list[IntentAction]:
        if imp.imported is None:
            return [IntentAction(ACTION_COMPOSE, "Paste into the composer", "EDIT_NOTE")]
        actions = [IntentAction(ACTION_TRANSLATE_NEW_CHAT, "Translate in new chat", "ADD_COMMENT",
                                None if ACTION_TRANSLATE_NEW_CHAT in self.handlers else "Chat import is not ready")]
        library_ok = imp.imported.extension in LIBRARY_EXTENSIONS
        actions.append(IntentAction(ACTION_ADD_TO_LIBRARY, "Add to Library", "LOCAL_LIBRARY",
                                    None if library_ok else LIBRARY_REASON))
        if ACTION_OPEN_IN_READER not in self.handlers:
            reader_reason: Optional[str] = READER_REASON
        elif imp.imported.extension not in READER_EXTENSIONS:
            reader_reason = READER_TYPES_REASON
        else:
            reader_reason = None
        actions.append(IntentAction(ACTION_OPEN_IN_READER, "Open in Reader", "AUTO_STORIES", reader_reason))
        if imp.imported.extension in MANGA_EXTENSIONS:
            actions.append(IntentAction(ACTION_MANGA, "Manga translator", "AUTO_STORIES",
                                        None if ACTION_MANGA in self.handlers else MANGA_REASON))
        return actions

    # ---- handling -------------------------------------------------------------------------------

    def _item_id(self, item: Mapping[str, Any]) -> str:
        item_id = item.get("id")
        if item_id:
            return str(item_id)
        return "|".join(str(item.get(k) or "") for k in ("kind", "path", "text", "name"))

    async def handle(self, items: Sequence[Any], *, source: str = "share") -> list[IntentImport]:
        """Import every new item, then present the action sheet; returns the imports."""
        fresh: list[Mapping[str, Any]] = []
        for item in items or ():
            if not self.accepts(item):
                if isinstance(item, Mapping) and str(item.get("uri") or "").lower().startswith(_BLOCKED_SCHEMES):
                    log.info("ignoring a shared URI (never routed)")
                continue
            key = self._item_id(item)
            if key in self._seen:
                continue
            self._seen.add(key)
            fresh.append(item)
        if not fresh:
            return []
        imports: list[IntentImport] = []
        file_items = [i for i in fresh if self._kind(i) == "file"]
        for item in fresh:
            if self._kind(item) != "file":
                imports.append(IntentImport(item=item, text=str(item.get("text") or "")))
        if file_items:
            try:
                imported = await self.files.run_io(
                    lambda: self.files.import_paths([i.get("path") for i in file_items],
                                                    names=[i.get("name") for i in file_items])
                )
            except Exception as exc:
                log.exception("importing shared files failed")
                imported = []
                for item in file_items:
                    imports.append(IntentImport(item=item, error=f"Import failed: {exc}"))
            by_source = {os.path.normcase(os.path.abspath(f.source)): f for f in imported}
            for item in file_items:
                found = by_source.get(os.path.normcase(os.path.abspath(str(item.get("path")))))
                if found is not None:
                    imports.append(IntentImport(item=item, imported=found))
                elif not any(i.item is item for i in imports):
                    imports.append(IntentImport(item=item, error=item.get("error") or "The file could not be read"))
        for imp in imports:
            imp.actions = self.actions_for(imp) if imp.error is None else []
        self.history.extend(imports)
        del self.history[:-50]
        if self.native is not None and file_items:
            try:
                await self.native.clear_shared(False)
            except Exception:
                pass
        log.info("shared %d item(s) via %s", len(imports), source)
        if self.present is not None:
            result = self.present(imports)
            if hasattr(result, "__await__"):
                await result
        return imports

    async def perform(self, action_id: str, imp: IntentImport) -> Any:
        if action_id == ACTION_ADD_TO_LIBRARY and ACTION_ADD_TO_LIBRARY not in self.handlers:
            if imp.imported is None:
                return None
            added = await self.files.run_io(self.files.add_to_library, imp.imported.path)
            if self.notify is not None:
                self.notify(f"Added to Library: {added.name}")
            return added
        handler = self.handlers.get(action_id)
        if handler is None:
            if self.notify is not None:
                self.notify("Not available yet")
            return None
        result = handler(imp)
        if hasattr(result, "__await__"):
            result = await result
        return result
