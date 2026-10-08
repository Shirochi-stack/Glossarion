"""MangaFeature: wires Tools › Manga (UI_SPEC §4.6) into GlossarionApp.

``await MangaFeature.install(app)`` — call it after the Tools feature (it reuses its
``ToolsContext``) and after the jobs layer (FileBridge, IntentRouter, JobService):

1. exposes the feature as ``app.manga`` and wraps the shell's screen factory for
   ``tools.manga`` (``/tools/manga?tab=files|settings|editor``); every other route falls through;
2. registers the IntentRouter "Manga translator" action (shared images / CBZ / ZIP open in
   the Files tab);
3. wraps the chat's tool handler (like the glossary feature): the ＋ sheet's Manga tool and the
   "Translate as manga" quick chip pass the composer's image / CBZ attachment to the Files tab.

The session state (``MangaSession``: the Files list, the model manager, the editor's state
store, the jobs the screen follows) outlives the screen, so leaving and reopening the tool keeps
the selection, a running batch and the editor page.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from typing import Any, Callable, Iterable, Optional

from glossarion_mobile.ui.router import RouteMatch

__all__ = ["IMPLEMENTED_ROUTES", "MangaFeature", "MangaSession", "SCREEN_ROUTES", "accepts_path"]

log = logging.getLogger("glossarion.tools.manga")

SCREEN_ROUTES = ("tools.manga",)
#: Routes this feature ships (Integrate merges them into the hub / drawer "implemented" sets).
IMPLEMENTED_ROUTES = frozenset(SCREEN_ROUTES)
INTENT_EXTENSIONS = (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".cbz", ".zip")


def accepts_path(path: Optional[str]) -> bool:
    """Images, CBZ and ZIP files open in the manga translator."""
    return bool(path) and os.path.splitext(str(path))[1].lower() in INTENT_EXTENSIONS


class MangaSession:
    """Per-app manga state shared by the three tabs (kept across screen visits): the Files list
    (``MangaFileList`` over the moved Files methods), the model manager and the editor session
    (``manga_editor_core.MangaEditorSession``, created on first use; jobs reach it by token)."""

    def __init__(self, *, data_dir: str, config_source: Optional[Callable[[], dict]] = None,
                 save: Optional[Callable[[dict], Any]] = None, config: Optional[dict] = None,
                 prefs: Any = None) -> None:
        from glossarion_mobile.services import manga as svc

        self.data_dir = data_dir or os.getcwd()
        self.root = os.path.join(self.data_dir, "manga")
        self.cbz_root = os.path.join(self.root, "cbz")
        self.folders_root = os.path.join(self.root, "folders")
        self.view_cache = os.path.join(self.root, "view")
        self.exports_dir = os.path.join(self.root, "exports")
        self._save = save
        self._config_source = config_source or (lambda: dict(config or {}))
        # prefs (mobile_state.json): the CBZ archives of the selection survive an app restart
        self.files = svc.MangaFileList(self._config_source(), save=self.save, temp_root=self.cbz_root,
                                       config_source=self._config_source, folders_root=self.folders_root,
                                       prefs=prefs)
        self.models = svc.ModelManager()
        self.editor: Any = None  # MangaEditorSession
        self.editor_token: Optional[str] = None
        self.editor_error: Optional[str] = None
        self._editor_lock = threading.Lock()
        self.editor_listeners: list = []
        self.loaded = False
        self.page_index = 0
        self.batch_job_id: Optional[str] = None
        self.batch_end_applied: Optional[str] = None  # the batch whose end a Files tab has shown
        self.step_job_id: Optional[str] = None
        self.last_outputs: list = []
        self.last_cbz: list = []  # CBZ archives the last run / Create CBZ wrote (Share / Save)
        self.last_result: dict = {}
        # the last imported OCR JSON ({"path", "matched", "files"}): the next Start reuses it (desktop)
        self.imported_ocr: Optional[dict] = None
        self.pending: list = []  # paths handed over before the selection was loaded

    def config(self) -> dict:
        return dict(self._config_source() or {})

    def save(self, updates: dict) -> None:
        if self._save is not None:
            self._save(dict(updates))

    def load(self, config: dict) -> int:
        """Restore the persisted selection (blocking: checks every saved path)."""
        count = self.files.load(config)
        self.loaded = True
        return count

    def ensure_editor(self) -> Any:
        """Blocking (io pool): the editor session over the Files pages (None + ``editor_error``
        when this build has no editor core)."""
        from glossarion_mobile.services import manga as svc

        with self._editor_lock:
            if self.editor is None:
                try:
                    self.editor = svc.new_editor_session(image_paths=self.files.files,
                                                         event_callback=self._on_editor_event)
                    self.editor_token = svc.register_editor_session(self.editor)
                    self.editor_error = None
                except Exception as exc:
                    self.editor_error = str(exc)
                    log.info("manga editor session unavailable: %s", exc)
                    return None
            else:
                self.editor.set_pages(self.files.files)
            return self.editor

    def _on_editor_event(self, kind: str, data: Any) -> None:
        for listener in list(self.editor_listeners):
            try:
                listener(kind, data)
            except Exception:
                log.debug("editor event listener failed", exc_info=True)


class MangaFeature:
    def __init__(self, app: Any) -> None:
        self.app = app
        self.page = getattr(app, "page", None)
        self.dispatcher = getattr(app, "dispatcher", None)
        self._fallback_factory: Optional[Callable[[RouteMatch], Any]] = None
        self._session: Optional[MangaSession] = None
        self.screens_built: list = []
        self.screen: Any = None  # the MangaScreen on screen (one at a time)

    @classmethod
    async def install(cls, app: Any) -> "MangaFeature":
        feature = cls(app)
        feature.attach()
        return feature

    def attach(self) -> None:
        app = self.app
        app.manga = self
        shell = getattr(app, "shell", None)
        if shell is not None and self._fallback_factory is None:
            self._fallback_factory = shell.screen_factory
            shell.screen_factory = self.screen_factory
        intents = getattr(app, "intents", None)
        if intents is not None:
            try:
                from glossarion_mobile.services.intents import ACTION_MANGA

                intents.handlers[ACTION_MANGA] = self._from_intent
            except Exception:
                log.exception("registering the manga intent failed")
        chat_view = getattr(app, "chat_view", None)
        if chat_view is not None and hasattr(chat_view, "_on_tool") and not getattr(chat_view, "_manga_tool", False):
            # ＋ › Manga translator and the "Translate as manga" chip (ChatView._on_quick_chip -> _on_tool):
            # the composer's image / CBZ attachment goes to the Files tab (the glossary feature wraps
            # "extract_glossary" the same way).
            original = chat_view._on_tool

            def on_tool(tool_id: str) -> Any:
                if tool_id == "manga":
                    try:
                        chat_view.composer.set_plus_open(False)
                    except Exception:
                        pass
                    attachment = getattr(chat_view.composer, "attachment", None) or {}
                    path = attachment.get("path") if isinstance(attachment, dict) else None
                    return self.open_from_chat(str(path) if path else None)
                return original(tool_id)

            chat_view._on_tool = on_tool
            chat_view._manga_tool = True

    # ---- context / session -----------------------------------------------------------------------

    def context(self) -> Any:
        tools = getattr(self.app, "tools", None)
        if tools is None:
            from glossarion_mobile.ui.tools.feature import ToolsFeature

            tools = ToolsFeature(self.app)
        ctx = tools.context()
        ctx.extras["manga_feature"] = self
        return ctx

    def _config(self) -> dict:
        store = getattr(self.app, "config_store", None)
        try:
            return dict(store.snapshot() or {}) if store is not None else {}
        except Exception:
            return {}

    def _save(self, updates: dict) -> None:
        store = getattr(self.app, "config_store", None)
        if store is None:
            return
        try:
            store.set_many(dict(updates))
        except Exception:
            log.exception("saving the manga selection failed")

    @property
    def session(self) -> MangaSession:
        if self._session is None:
            paths = getattr(self.app, "paths", None)
            data_dir = str(getattr(paths, "data", "") or "") if paths is not None else ""
            self._session = MangaSession(data_dir=data_dir or os.getcwd(), config_source=self._config,
                                         save=self._save, prefs=getattr(self.app, "prefs", None))
        return self._session

    # ---- screens ------------------------------------------------------------------------------------

    def make_screen(self, match: RouteMatch) -> Any:
        from glossarion_mobile.ui.tools.manga.screen import MangaScreen

        return MangaScreen(match, self.context(), session=self.session, feature=self)

    def screen_factory(self, match: RouteMatch) -> Any:
        screen = None
        if match.name in SCREEN_ROUTES:
            try:
                screen = self.make_screen(match)
            except Exception:
                log.exception("building the manga screen failed")
                screen = None
            if screen is not None:
                self.screens_built.append(match.name)
        if screen is None:
            if self._fallback_factory is None:
                raise LookupError(f"no screen for {match.name}")
            screen = self._fallback_factory(match)
        return screen

    # ---- entry points -------------------------------------------------------------------------------

    def _navigate(self, tab: str = "files") -> None:
        navigate = getattr(self.app, "navigate_to", None)
        if navigate is None:
            return
        try:
            navigate("tools.manga", None, {"tab": tab})
        except TypeError:
            navigate("tools.manga")

    def _showing(self) -> bool:
        screen = self.screen
        shell = getattr(self.app, "shell", None)
        top = getattr(shell, "top_screen", None) if shell is not None else None
        return screen is not None and (top is None or top is screen)

    def open_with(self, paths: Iterable[str], *, tab: str = "files") -> int:
        """Hand files to the Files tab (Open-with, the chat attachment) and show the tool."""
        wanted = [os.fspath(p) for p in paths if accepts_path(os.fspath(p))]
        if not wanted:
            return 0
        self.session.pending.extend(wanted)
        screen = self.screen
        if screen is not None and self._showing():
            self.spawn(screen.take_pending())
            screen.select_tab(tab)
        else:
            self._navigate(tab)
        return len(wanted)

    def open_from_chat(self, attachment: Optional[str] = None) -> bool:
        """＋ › Manga translator / the "Translate as manga" quick chip: the composer's image or
        CBZ attachment goes to the Files tab (without one the tool opens as is)."""
        if attachment and accepts_path(attachment):
            self.open_with([attachment])
        else:
            self._navigate("files")
        return True

    def _from_intent(self, imp: Any) -> Any:
        imported = getattr(imp, "imported", None)
        path = getattr(imported, "path", None)
        if not path:
            return None
        return self.open_with([path])

    def spawn(self, coro: Any) -> Any:
        dispatcher = self.dispatcher
        if dispatcher is not None and getattr(dispatcher, "bound", False):
            return dispatcher.spawn(coro)
        try:
            return asyncio.ensure_future(coro)
        except RuntimeError:
            coro.close()
            return None
