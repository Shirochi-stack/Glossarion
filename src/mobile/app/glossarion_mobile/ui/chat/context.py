"""ChatEnv: what the chat home needs from the app, in one object (U3).

It extends the settings ``SettingsContext`` (same threading helpers: ``on_ui``,
``run_io``, ``spawn``, ``say``, ``go``) with the chat services. Built by
``integration.ChatFeature``; tests build it with fakes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from glossarion_mobile.ui.settings.context import SettingsContext

__all__ = ["ChatEnv"]


@dataclass
class ChatEnv(SettingsContext):
    chats: Any = None  # state.chat_store_adapter.ChatStoreAdapter
    runs: Any = None  # ui.chat.run_controller.ChatRuns
    jobs: Any = None  # ui.chat.job_binding.JobsAdapter
    oauth: Any = None  # services.oauth.OAuthBridge
    copy_text: Optional[Callable[[str], Any]] = None  # async or sync
    read_clipboard: Optional[Callable[[], Any]] = None  # async -> str | None
    share_files: Optional[Callable[[list], Any]] = None  # async (paths)
    export_file: Optional[Callable[[str], Any]] = None  # ExportSheet for one file (Share / Save to / Downloads)
    open_output: Optional[Callable[[str], Any]] = None  # output folder -> file browser
    open_reader: Optional[Callable[..., Any]] = None  # async (workspace folder, attachment path) -> Reader (U5)
    pick_files: Optional[Callable[..., Any]] = None  # async (extensions, multiple) -> [paths]
    # U9: async () -> the picked folder's app-owned copy (FileBridge.pick_folder); raises
    # FolderPickUnavailable where the platform cannot hand over a folder (Android SAF trees)
    pick_folder: Optional[Callable[[], Any]] = None
    # U9: (route name, source path) -> a tool screen with this file as its source (Plan "Run as async batch")
    open_tool_with_source: Optional[Callable[[str, str], Any]] = None
    # U9: () -> the Tools ToolsContext (＋ › From Library: the SourcePicker of Library books)
    tools_context: Optional[Callable[[], Any]] = None
    import_file: Optional[Callable[[str], Any]] = None  # picked path -> app-owned copy (FileBridge); blocking
    push_overlay: Optional[Callable[[Any], Any]] = None  # full-screen ft.View
    pop_overlay: Optional[Callable[[Any], Any]] = None
    # U7: a workspace in the Progress manager (Chapters: Retranslate / Resolve QA), a file handed to
    # the system (Open externally), FileBridge "Save to…", and the Library hand-off after Migrate.
    open_progress: Optional[Callable[..., Any]] = None  # (workspace folder, attachment path) -> route id
    open_external: Optional[Callable[[str], Any]] = None
    save_file: Optional[Callable[[str], Any]] = None  # async: FileBridge save_as
    after_migrate: Optional[Callable[..., Any]] = None  # (target folder, attachment path)
    profiles: Callable[[], list] = field(default=lambda: [])
    languages: tuple = ()
    mono: str = "monospace"
    is_android: bool = False
    is_ios: bool = False
    temp_dir: Optional[str] = None

    def config_get(self, key: str, default: Any = None) -> Any:
        store = self.store
        if store is None:
            return default
        try:
            return store.get(key, default)
        except Exception:
            return default

    def config_set_many(self, values: dict) -> None:
        if self.store is not None and values:
            self.store.set_many(values)
