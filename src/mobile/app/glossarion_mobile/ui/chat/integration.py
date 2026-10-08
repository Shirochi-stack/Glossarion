"""ChatFeature: wires the U3 chat (store, runs, sign-in, screens) into GlossarionApp.

``await ChatFeature.install(app)`` - call it in ``GlossarionApp.start`` right after
``_install_settings()`` (the history and the token store decrypt with the
SecureStorage keys, and the settings feature provides ``app.config_store``):

1. ``ChatStoreAdapter`` over the shared ``direct_text_store`` (desktop-format
   ``direct_text_chats.json`` v2 + the ``direct_text_chats.mobile.json`` sidecar),
   loaded on a worker thread; it replaces ``app.state.chats`` so the drawer's
   Pinned / Recents come from the real history (the drawer re-subscribes);
2. ``OAuthBridge`` (ChatGPT sign-in through the registered ``webbrowser``
   controller = UrlLauncher IN_APP_BROWSER_VIEW; Android short sign-in FGS);
   ``state.signed_in`` follows the token store once the backend is ready;
3. ``ChatRuns`` over JobService (``app.jobs`` / ``app.job_service`` when the
   mobile job service is installed) and ``ChatView.bind(env)``;
4. the screen factory gains ``/settings/accounts``, ``/welcome`` and
   ``/chat/<cid>/m/<mid>/edit``; ``/chat/<cid>/settings`` opens the chat settings
   sheet; ``/chat/<cid>`` selects that chat; ``/oauth/return?p=authgpt`` reaches the
   bridge; the drawer's chat long-press gets Rename / Pin / Delete;
5. lifecycle INACTIVE / HIDE / PAUSE / DETACH flush the chat history;
6. U7: ``/chat/<cid>/attachments`` (the Attachments manager), the drawer row's
   Attachments / Export chat / Duplicate as scratch (Save / Discard for a scratch chat), scratch
   chats under ``<cache>/Direct Text Scratch``, and the chat's hooks for the Progress manager
   (Retranslate chapters), Open externally and the Library hand-off after a move;
7. device fixes: a finished chat book moves into the Library by itself (``auto_migrate``, the
   desktop Migrate run for the user, UI_SPEC §2.17): when its run finishes (``ChatRuns``
   ``subscribe_finished``), whenever the job queue goes idle and once after install (books from
   earlier versions and runs whose app was killed before the move). The env gets the Library
   hooks the chat UI uses (``library_service`` / ``library_book`` / ``library_translate``).

``maybe_show_welcome()`` opens ``/welcome`` on a first run (no
``glossary_mode_dialog_shown`` in config.json and no completed mobile welcome); the
app calls it after its initial route.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
from typing import Any, Callable, Optional

from glossarion_mobile.state.chat_store_adapter import (
    SCRATCH_DIR_NAME,
    ChatStoreAdapter,
    default_history_path,
    is_scratch_cid,
)
from glossarion_mobile.ui.chat.context import ChatEnv
from glossarion_mobile.ui.chat.job_binding import TERMINAL_STATES, JobsAdapter, state_name
from glossarion_mobile.ui.chat.run_controller import RESUMABLE_ENDINGS, ChatRuns
from glossarion_mobile.ui.router import RouteMatch

__all__ = ["ADDED_TO_LIBRARY", "COLLISION_TEXT", "ChatFeature", "FLUSH_LIFECYCLE_STATES", "MIGRATE_COLLISION",
           "MIGRATE_DEFERRED", "MIGRATE_FAILED", "MIGRATE_MOVED", "MIGRATE_SKIPPED", "SCREEN_ROUTES"]

log = logging.getLogger("glossarion.chat")

FLUSH_LIFECYCLE_STATES = ("inactive", "hide", "pause", "detach")
SCREEN_ROUTES = ("settings.accounts", "welcome", "chat.message.edit", "chat.attachments", "chat.compose")
WELCOME_PREF = "welcome_completed"

#: ``auto_migrate`` outcomes: moved into the Library; left where it is for good (not a book, a run that
#: can still be resumed, ...); left for the next idle sweep (a job is running or writing it); left
#: because a different Library book has its name (the merge dialog); the shared Migrate refused.
MIGRATE_MOVED, MIGRATE_SKIPPED, MIGRATE_DEFERRED, MIGRATE_COLLISION, MIGRATE_FAILED = (
    "moved", "skipped", "deferred", "collision", "failed")
ADDED_TO_LIBRARY = "Added to the Library"
COLLISION_TEXT = "A Library book named {name} already exists"
#: How long the startup sweep waits for the Library feature (installed after the chat).
STARTUP_LIBRARY_WAIT = 30.0


def _norm(path: Any) -> str:
    try:
        return os.path.normcase(os.path.normpath(os.path.abspath(str(path))))
    except Exception:
        return str(path or "")


def _chapter_hashes(folder: Any) -> set:
    """The ``content_hash`` of every chapter in ``<folder>/translation_progress.json`` (the pipeline
    hashes each chapter's source text); empty when there is no readable progress."""
    try:
        with open(os.path.join(str(folder), "translation_progress.json"), encoding="utf-8") as stream:
            progress = json.load(stream)
    except (OSError, ValueError):
        return set()
    chapters = progress.get("chapters") if isinstance(progress, dict) else None
    if not isinstance(chapters, dict):
        return set()
    return {str(entry["content_hash"]) for entry in chapters.values()
            if isinstance(entry, dict) and entry.get("content_hash")}


def _inside(path: Any, root: Any) -> bool:
    try:
        return os.path.commonpath([_norm(path), _norm(root)]) == _norm(root)
    except ValueError:  # another drive
        return False


def _target_languages() -> tuple:
    try:
        from language_options import TARGET_LANGUAGES

        return tuple(TARGET_LANGUAGES)
    except Exception:
        return ("English",)


def profile_names(config_get: Callable[[str, Any], Any]) -> list:
    """Prompt profile names: config ``prompt_profiles`` or the built-in defaults (owner_state)."""
    profiles = config_get("prompt_profiles", None)
    if isinstance(profiles, dict) and profiles:
        return list(profiles)
    try:
        import types

        from owner_state import ConfigStateMixin

        holder = types.SimpleNamespace()
        ConfigStateMixin._init_default_prompt_profiles(holder)
        return list(getattr(holder, "default_prompts", {}) or {})
    except Exception:
        return ["Universal"]


def _reader_workspace(folder: str, source: str = "") -> str:
    """The translation workspace (``translation_progress.json``) of a chat run (blocking).

    ``folder`` is the run's pipeline output folder, or for older turns the chat's output
    folder, whose per-attachment workspaces are subfolders: the one named after the
    attachment wins, else the most recently written one.
    """
    if not folder or not os.path.isdir(folder):
        return ""
    if os.path.isfile(os.path.join(folder, "translation_progress.json")):
        return folder
    stem = os.path.splitext(os.path.basename(str(source or "")))[0].lower()
    candidates = []
    try:
        entries = list(os.scandir(folder))
    except OSError:
        return ""
    for entry in entries:
        progress = os.path.join(entry.path, "translation_progress.json")
        if entry.is_dir() and os.path.isfile(progress):
            if stem and entry.name.lower() == stem:
                return entry.path
            try:
                candidates.append((os.path.getmtime(progress), entry.path))
            except OSError:
                continue
    return max(candidates)[1] if candidates else ""


class ChatFeature:
    def __init__(self, app: Any, *, chats: Optional[ChatStoreAdapter] = None, jobs: Any = None, oauth: Any = None) -> None:
        self.app = app
        self.page = getattr(app, "page", None)
        self.dispatcher = getattr(app, "dispatcher", None)
        paths = getattr(app, "paths", None)
        platform = str(getattr(getattr(self.page, "platform", None), "value", "") or "")
        self.is_android = platform == "android"
        self.is_ios = platform == "ios"
        cache_dir = getattr(paths, "cache", None)
        scratch_dir = os.path.join(str(cache_dir), SCRATCH_DIR_NAME) if cache_dir else None
        self.chats = chats or ChatStoreAdapter(history_path=default_history_path(), scratch_dir=scratch_dir)
        self._temp_dir = str(getattr(paths, "temp", "") or "") or None
        self._url_launcher: Any = None
        self.jobs = JobsAdapter(jobs if jobs is not None else (getattr(app, "jobs", None) or getattr(app, "job_service", None)))
        if oauth is None:
            from glossarion_mobile.services.oauth import OAuthBridge

            opener = getattr(app, "opener", None)
            oauth = OAuthBridge(
                close_browser=getattr(opener, "close_in_app_view", None),
                native=getattr(app, "native", None),
                post=self._post,
                run_io=self._run_io,
                is_android=self.is_android,
            )
        self.oauth = oauth
        # Run roots live in app data (not the OS temp dir, which may vanish between launches) so an
        # interrupted run resumes from its translation_progress.json after the app was killed.
        data_dir = getattr(paths, "data", None)
        temp_dir = os.path.join(str(data_dir), "direct_text_runs") if data_dir else None
        self.runs = ChatRuns(self.chats, self.jobs, run_io=self._run_io, temp_dir=temp_dir,
                             model_name=lambda: self.env.config_get("model", None) if self.env else None)
        self.env: Optional[ChatEnv] = None
        self._fallback_factory: Optional[Callable[[RouteMatch], Any]] = None
        self._file_picker: Any = None
        self._share: Any = None
        self._unsubs: list = []
        self.welcome_shown = False
        #: Extra long-press sheet rows after Pin (U9 Series' "Move to Series…"): ``chat -> [ActionItem]``.
        self.chat_action_providers: list = []
        # auto-migrate (UI_SPEC §2.17): one move at a time; workspaces the user agreed to merge into a
        # same-named Library folder; name clashes already announced (once per session); the sweep task
        self._migrate_lock = threading.Lock()
        self._merge_approved: set = set()
        self._collisions_shown: set = set()
        self._migrate_hooked = False
        self._sweeping = False
        self._sweep_again = False
        self.startup_task: Any = None

    # ---- threading -------------------------------------------------------------------------

    def _post(self, fn: Callable[..., Any], *args: Any) -> None:
        dispatcher = self.dispatcher
        if dispatcher is not None and getattr(dispatcher, "bound", False) and not dispatcher.on_loop_thread():
            dispatcher.post(fn, *args)
        else:
            fn(*args)

    async def _run_io(self, fn: Callable[..., Any], *args: Any) -> Any:
        dispatcher = self.dispatcher
        if dispatcher is not None and getattr(dispatcher, "bound", False):
            return await dispatcher.run_in_thread(fn, *args, name="gl-chat-io")
        return await asyncio.to_thread(fn, *args)

    # ---- install ----------------------------------------------------------------------------

    @classmethod
    async def install(cls, app: Any, **kwargs: Any) -> "ChatFeature":
        feature = cls(app, **kwargs)
        await feature._run_io(feature.chats.load)
        feature.attach()
        app.chat_feature = feature
        # Chat books finished before this launch (earlier versions, or the app was killed between the
        # finish and the move) join the Library once the Library feature is installed.
        feature.startup_task = feature._spawn(feature.startup_sweep())
        return feature

    def _settings_ctx_fields(self) -> dict:
        settings = getattr(self.app, "settings", None)
        return {
            "page": self.page,
            "store": getattr(self.app, "config_store", None),
            "schema": getattr(settings, "schema", None),
            "dispatcher": self.dispatcher,
            "navigate_route": getattr(self.app, "navigate", None),
            "navigate_name": getattr(self.app, "navigate_to", None),
            "notify": getattr(self.app, "notify", None),
            "prefs": getattr(self.app, "prefs", None),
            "tablet": bool(getattr(getattr(self.app, "shell", None), "tablet", False)),
        }

    def build_env(self) -> ChatEnv:
        from glossarion_mobile.ui.theme import mono_family

        app = self.app
        env = ChatEnv(
            **self._settings_ctx_fields(),
            chats=self.chats,
            runs=self.runs,
            jobs=self.jobs,
            oauth=self.oauth,
            copy_text=getattr(app, "_copy_text", None),
            read_clipboard=getattr(getattr(app, "clipboard", None), "get", None),
            share_files=self.share_files,
            export_file=self.export_file,
            open_output=self.open_output,
            open_reader=self.open_reader,
            pick_files=self.pick_files,
            pick_folder=self.pick_folder,
            open_tool_with_source=self.open_tool_with_source,
            tools_context=self.tools_context,
            import_file=self.import_file,
            push_overlay=self.push_overlay,
            pop_overlay=self.pop_overlay,
            open_progress=self.open_progress,
            open_external=self.open_external,
            after_migrate=self.after_migrate,
            library_service=self.library_service,
            library_book=self.library_book,
            library_translate=self.library_translate,
            temp_dir=self._temp_dir,
            languages=_target_languages(),
            mono=mono_family(self.page),
            is_android=self.is_android,
            is_ios=self.is_ios,
        )
        env.profiles = lambda: profile_names(env.config_get)
        return env

    def attach(self) -> None:
        app = self.app
        state = getattr(app, "state", None)
        self.chats.post = self._post
        self.env = self.build_env()
        if state is not None and self.chats.available:
            state.chats = self.chats
            cid = self.chats.current_cid()
            if state.current_chat.value != cid:
                state.current_chat.set(cid)
            drawer = getattr(app, "drawer", None)
            if drawer is not None:
                drawer.detach()
                drawer.state = state
                drawer.attach()
                drawer.on_chat_long_press = self.chat_actions
                drawer._changed()
        chat_view = getattr(app, "chat_view", None)
        if chat_view is not None:
            chat_view.bind(self.env)
            extras = getattr(getattr(getattr(app, "settings", None), "ctx", None), "extras", None)
            if isinstance(extras, dict):
                # Settings › Direct Text › "Edit in Chat settings › All chats…" (section_page STATIC_LINKS)
                extras.setdefault("actions", {})["chat_settings_global"] = (
                    lambda: chat_view.open_chat_settings(scope="global"))
        self.runs.attach()
        self._hook_auto_migrate()
        shell = getattr(app, "shell", None)
        if shell is not None and self._fallback_factory is None:
            self._fallback_factory = shell.screen_factory
            shell.screen_factory = self.screen_factory
            original_show_sheet = shell.show_sheet
            original_show = shell.show

            def show_sheet(match: RouteMatch) -> Any:
                if match.name == "chat.settings" and chat_view is not None:
                    self._select_chat(match.params.get("cid"))
                    return chat_view.open_chat_settings()
                return original_show_sheet(match)

            def show(match: RouteMatch, **kwargs: Any) -> bool:
                if match.name in ("chat", "chat.message"):
                    self._select_chat(match.params.get("cid"))
                return original_show(match, **kwargs)

            shell.show_sheet = show_sheet
            shell.show = show
        original_return = getattr(app, "_on_oauth_return", None)

        def on_oauth_return(match: RouteMatch) -> None:
            provider = match.get("p")
            if provider == "authgpt":
                self.oauth.on_return_link(provider)
                return
            if original_return is not None:
                original_return(match)

        app._on_oauth_return = on_oauth_return
        self._hook_lifecycle()
        if state is not None:
            self._unsubs.append(state.backend.subscribe(self._on_backend))
            self._on_backend(state.backend.value)

    def _select_chat(self, cid: Any) -> None:
        state = getattr(self.app, "state", None)
        if cid and state is not None and self.chats.session(cid) is not None and state.current_chat.value != str(cid):
            state.current_chat.set(str(cid))

    def _hook_lifecycle(self) -> None:
        page = self.page
        if page is None:
            return
        original = getattr(page, "on_app_lifecycle_state_change", None)
        if getattr(original, "_glossarion_chat", False):
            return

        async def on_lifecycle(e: Any) -> None:
            state = getattr(e, "state", None)
            if str(getattr(state, "value", state)) in FLUSH_LIFECYCLE_STATES:
                try:
                    self.chats.flush()
                except Exception:
                    log.exception("flushing the chat history failed")
            oauth_lifecycle = getattr(self.oauth, "on_lifecycle", None)
            if callable(oauth_lifecycle):  # back from the browser with the sign-in still waiting
                try:
                    oauth_lifecycle(str(getattr(state, "value", state)))
                except Exception:
                    log.exception("sign-in lifecycle check failed")
            if original is not None:
                result = original(e)
                if hasattr(result, "__await__"):
                    await result

        on_lifecycle._glossarion_chat = True  # type: ignore[attr-defined]
        on_lifecycle._glossarion_settings = getattr(original, "_glossarion_settings", False)  # type: ignore[attr-defined]
        page.on_app_lifecycle_state_change = on_lifecycle

    def _on_backend(self, result: Any) -> None:
        if not result or not result.get("ok"):
            return
        self._spawn(self.refresh_sign_in())

    def _spawn(self, coro: Any) -> Any:
        dispatcher = self.dispatcher
        if dispatcher is not None and getattr(dispatcher, "bound", False):
            return dispatcher.spawn(coro)
        try:
            return asyncio.ensure_future(coro)
        except RuntimeError:
            coro.close()
            return None

    async def refresh_sign_in(self) -> dict:
        """Re-read ChatGPT slot #0 and the slot(s) the current model uses (``authgptN/`` -> #N, the
        ``authgpt0/`` pool -> every saved slot) into ``AppState.signed_in`` (slot keys). Returns
        slot #0's status."""
        from glossarion_mobile.services.oauth import pool_route_provider, provider_for_model, slot_key

        status = await self.oauth.refresh_status(0)
        found = {"authgpt": bool(status.get("signed_in"))}
        state = getattr(self.app, "state", None)
        model = state.chat_context.value.model if state is not None else None
        route = provider_for_model(model)
        try:
            if route is not None and route[0] == "authgpt":
                if pool_route_provider(model) == "authgpt" and hasattr(self.oauth, "statuses"):
                    for item in await self._run_io(self.oauth.statuses, "authgpt"):
                        found[slot_key("authgpt", int(item.get("account_id", 0) or 0))] = bool(item.get("signed_in"))
                elif route[1]:
                    other = await self.oauth.refresh_status(route[1])
                    found[slot_key("authgpt", route[1])] = bool(other.get("signed_in"))
        except Exception:
            log.debug("refreshing the model's ChatGPT slot failed", exc_info=True)
        if state is not None:
            signed = set(state.signed_in.value)
            for key, value in found.items():
                if value:
                    signed.add(key)
                else:
                    signed.discard(key)
            if frozenset(signed) != state.signed_in.value:
                state.signed_in.set(frozenset(signed))
        return status

    # ---- screens --------------------------------------------------------------------------------

    def make_screen(self, match: RouteMatch) -> Any:
        if match.name == "settings.accounts":
            from glossarion_mobile.ui.screens.accounts import AccountsScreen

            return AccountsScreen(match, oauth=self.oauth, notify=getattr(self.app, "notify", None),
                                  on_signed_in_changed=lambda _v: self._spawn(self.refresh_sign_in()))
        if match.name == "welcome":
            return self.welcome_screen(match)
        if match.name == "chat.attachments":
            chat_view = getattr(self.app, "chat_view", None)
            cid = str(match.params.get("cid"))
            if chat_view is None or not chat_view.bound or self.chats.session(cid) is None:
                return None
            self._select_chat(cid)
            if chat_view.cid != cid:
                chat_view.load_chat(cid)
            return chat_view.attachments_screen(match)
        if match.name == "chat.compose":
            chat_view = getattr(self.app, "chat_view", None)
            if chat_view is None:
                return None
            cid = str(match.params.get("cid"))
            if chat_view.bound and self.chats.session(cid) is not None and chat_view.cid != cid:
                self._select_chat(cid)
                chat_view.load_chat(cid)
            return chat_view.compose_screen(match, on_close=lambda: getattr(self.app, "back", lambda: None)())
        if match.name == "chat.message.edit":
            from glossarion_mobile.ui.screens.output_editor import OutputEditorScreen

            cid = str(match.params.get("cid"))
            index = self.chats.index_for_mid(cid, match.params.get("mid", ""))
            if index is None:
                return None
            chat_view = getattr(self.app, "chat_view", None)
            return OutputEditorScreen(
                match, chats=self.chats, cid=cid, index=index, run_io=self._run_io,
                on_saved=(lambda: chat_view.render_transcript()) if chat_view is not None else None,
                on_close=lambda: getattr(self.app, "back", lambda: None)(),
                notify=getattr(self.app, "notify", None), mono=self.env.mono if self.env else "monospace",
            )
        return None

    def screen_factory(self, match: RouteMatch) -> Any:
        screen = None
        try:
            screen = self.make_screen(match)
        except Exception:
            log.exception("building the %s screen failed", match.name)
        if screen is None:
            if self._fallback_factory is None:
                raise LookupError(f"no screen for {match.name}")
            screen = self._fallback_factory(match)
        return screen

    def welcome_screen(self, match: RouteMatch) -> Any:
        from glossarion_mobile.ui.screens.accounts import LoginPanel
        from glossarion_mobile.ui.screens.welcome import DEFAULT_GLOSSARY_MODE, WelcomeFlow, WelcomeScreen

        env = self.env
        state = getattr(self.app, "state", None)
        configured_mode = None
        if env is not None and env.config_get("glossary_mode_dialog_shown", False):
            configured_mode = env.config_get("auto_glossary_mode", None)
        flow = WelcomeFlow(
            glossary_mode=configured_mode or DEFAULT_GLOSSARY_MODE,
            target_language=(env.config_get("output_language", None) if env else None) or "English",
            signed_in="authgpt" in (state.signed_in.value if state is not None else ()),
        )

        def choose_model(query: Optional[str] = None) -> Any:
            chat_view = getattr(self.app, "chat_view", None)
            if chat_view is None:
                return None
            sheet = chat_view.open_model_sheet("model")
            if query and hasattr(sheet, "set_query"):
                sheet.set_query(query)  # step 2 after a sign-in: that provider's models
            return sheet

        def done(updates: dict) -> None:
            if env is not None:
                updates = dict(updates)
                language = updates.pop("output_language", None)
                env.config_set_many(updates)  # the desktop welcome's glossary-page writes
                if language:  # the main-window Target Language combo's fan-out
                    from glossarion_mobile.state.setting_writes import write_setting

                    write_setting(env.store, "output_language", language)
            prefs = getattr(self.app, "prefs", None)
            if prefs is not None:
                try:
                    prefs.set(WELCOME_PREF, True)
                except Exception:
                    pass
            chat_view = getattr(self.app, "chat_view", None)
            if chat_view is not None:
                chat_view.apply_settings_changed()
            navigate = getattr(self.app, "navigate_to", None)
            if navigate is not None:
                navigate("home")

        def try_action(try_id: str) -> None:
            chat_view = getattr(self.app, "chat_view", None)
            if try_id == "library":
                getattr(self.app, "navigate_to", lambda *_: None)("library")
            elif try_id == "import" and chat_view is not None:
                chat_view._spawn(chat_view.pick_and_attach())

        def save_key(key: str) -> None:
            if env is not None:
                env.config_set_many({"api_key": key})

        return WelcomeScreen(
            match,
            flow=flow,
            login_panel_factory=lambda on_done: LoginPanel(self.oauth, provider="authgpt", account_id=0,
                                                           on_done=on_done),
            languages=env.languages if env else ("English",),
            on_finish=done,
            on_skip=done,
            on_choose_model=choose_model,
            on_use_provider=choose_model,
            on_api_key=save_key,
            on_try=try_action,
            is_android=self.is_android,
            is_ios=self.is_ios,
        )

    def needs_welcome(self) -> bool:
        prefs = getattr(self.app, "prefs", None)
        try:
            if prefs is not None and prefs.get(WELCOME_PREF):
                return False
        except Exception:
            pass
        env = self.env
        return not (env is not None and env.config_get("glossary_mode_dialog_shown", False))

    def maybe_show_welcome(self) -> bool:
        if self.welcome_shown or not self.needs_welcome():
            return False
        self.welcome_shown = True
        navigate = getattr(self.app, "navigate_to", None)
        if navigate is not None:
            navigate("welcome")
        return True

    # ---- files / sharing ------------------------------------------------------------------------

    def _picker(self) -> Any:
        if self._file_picker is None:
            import flet as ft

            self._file_picker = ft.FilePicker()  # a page service: keep the reference
        return self._file_picker

    async def pick_files(self, extensions: Any, multiple: bool = False) -> list:
        """FileBridge.pick_files (copies into the Inbox) when installed, else a bare FilePicker."""
        files_service = getattr(self.app, "files", None)
        picker = getattr(files_service, "pick_files", None)
        if callable(picker):
            result = picker(allowed_extensions=list(extensions or []), allow_multiple=bool(multiple))
            if asyncio.iscoroutine(result):
                result = await result
            return [str(getattr(item, "path", item)) for item in (result or []) if getattr(item, "path", item)]
        try:
            picked = await self._picker().pick_files(allowed_extensions=list(extensions or []) or None,
                                                     allow_multiple=bool(multiple))
        except Exception as exc:
            log.warning("file picker failed: %s", exc)
            return []
        return [f.path for f in (picked or []) if getattr(f, "path", None)]

    async def pick_folder(self) -> Optional[str]:
        """FileBridge.pick_folder: the folder copied into the Inbox (``FolderPickUnavailable`` propagates)."""
        files_service = getattr(self.app, "files", None)
        picker = getattr(files_service, "pick_folder", None)
        if not callable(picker):
            from glossarion_mobile.services.files import FolderPickUnavailable

            raise FolderPickUnavailable("Folder picking is not available in this session")
        folder = await picker(dialog_title="Select Folder Containing Files to Translate")
        return str(getattr(folder, "path", folder) or "") or None

    def tools_context(self) -> Any:
        """The Tools feature's ToolsContext (installed after the chat), else None."""
        tools = getattr(self.app, "tools", None)
        context = getattr(tools, "context", None)
        if not callable(context):
            return None
        try:
            return context()
        except Exception:
            log.exception("building the tools context failed")
            return None

    def open_tool_with_source(self, route_name: str, path: str) -> Optional[str]:
        """Open a tool screen with ``path`` preselected as its source (Tools › Async batch: the Plan card's
        "Run as async batch"): the tool's per-session state gets a chat-origin ToolTarget."""
        tools = getattr(self.app, "tools", None)
        state = getattr(tools, "tool_state", None)
        if not isinstance(state, dict) or not path:
            return None
        try:
            from glossarion_mobile.ui.tools.targets import ToolTarget

            ext = os.path.splitext(path)[1].lower().lstrip(".")
            target = ToolTarget(title=os.path.basename(path), source=path, origin="chat",
                                kind=ext if ext in ("epub", "pdf", "txt") else "other")
        except Exception:
            log.exception("building the tool target failed")
            return None
        tool = route_name.rsplit(".", 1)[-1]
        state.setdefault(tool, {})["target"] = target
        navigate = getattr(self.app, "navigate_to", None)
        if callable(navigate):
            navigate(route_name)
        return route_name

    def import_file(self, path: str) -> str:
        """Blocking: an app-owned copy of a picked file (FileBridge Inbox) when the bridge exists."""
        files_service = getattr(self.app, "files", None)
        inbox = getattr(files_service, "inbox_dir", None)
        if inbox and os.path.normcase(os.path.abspath(path)).startswith(os.path.normcase(os.path.abspath(inbox)) + os.sep):
            return path  # FileBridge.pick_files already copied it
        import_paths = getattr(files_service, "import_paths", None)
        if callable(import_paths):
            try:
                imported = import_paths([path])
                if imported:
                    return str(getattr(imported[0], "path", imported[0]))
            except Exception:
                log.exception("importing %s failed", os.path.basename(path))
        return path

    def export_file(self, path: str) -> Any:
        """ExportSheet: FileBridge options (Share… · Save to… · Save to Downloads · Show in Files)."""
        files_service = getattr(self.app, "files", None)
        options = getattr(files_service, "export_options", None)
        export = getattr(files_service, "export", None)
        if not (callable(options) and callable(export)):
            return self.share_files([path])
        from glossarion_mobile.ui.components.action_sheet import ActionItem, ActionSheet

        def run(option_id: str) -> Any:
            return export(option_id, path)

        sheet = ActionSheet(
            [ActionItem(o.label, (lambda oid=o.id: run(oid)), icon=o.icon, disabled_reason=o.disabled_reason)
             for o in options(path)],
            title=os.path.basename(path),
            tablet=bool(getattr(getattr(self.app, "shell", None), "tablet", False)),
        )
        sheet.show(self.page)
        return sheet

    async def share_files(self, paths: list) -> Any:
        files_service = getattr(self.app, "files", None)
        exporter = getattr(files_service, "share", None) or getattr(files_service, "share_files", None)
        if callable(exporter):
            result = exporter(list(paths))
            return (await result) if asyncio.iscoroutine(result) else result
        import flet as ft

        if self._share is None:
            self._share = ft.Share()
        return await self._share.share_files([ft.ShareFile.from_path(p) for p in paths])

    async def open_reader(self, folder: str, source: str = "") -> Optional[str]:
        """A chat run's translation in the Reader (U5): its output workspace over the raw source.

        The row uses the Library scanner's field names, so the Reader takes the shared desktop
        decision (``library_core.plan_open_reader``: overlay for an EPUB workspace, workspace mode
        for PDF / TXT); the raw file is the chat attachment, else the shared resolvers
        (``LibraryService.raw_source``: registry, ``source_epub.txt``). Without a workspace yet, an
        EPUB or TXT attachment opens on its own (raw). Returns the Reader's route id.
        """
        app = self.app
        reader = getattr(app, "reader", None)
        notify = getattr(app, "notify", None)
        if reader is None or not hasattr(reader, "open_book"):
            if notify is not None:
                notify("The Reader is not available in this session")
            return None
        library = getattr(app, "library", None)

        def target() -> Optional[dict]:
            workspace = _reader_workspace(str(folder or ""), str(source or ""))
            if workspace:
                name = os.path.basename(os.path.normpath(workspace))
                progress = os.path.join(workspace, "translation_progress.json")
                book = {"name": name, "folder_name": name, "path": workspace, "output_folder": workspace,
                        "type": "in_progress", "is_in_progress": True, "in_library": False,
                        "progress_file": progress if os.path.isfile(progress) else ""}
                raw = str(source or "") if source and os.path.isfile(str(source)) else ""
                if not raw and library is not None and hasattr(library, "raw_source"):
                    raw = library.raw_source(book)
                if raw:
                    book["raw_source_path"] = raw
                return {"book": book}
            if source and os.path.isfile(str(source)) and str(source).lower().endswith((".epub", ".txt")):
                return {"path": str(source)}
            return None

        found = await self._run_io(target)
        if not found:
            if notify is not None:
                notify("Nothing to read yet: the first chapter is not saved")
            return None
        if "book" in found:
            return reader.open_book(found["book"])
        return reader.open_book(path=found["path"])

    async def open_progress(self, folder: str, source: str = "", select: Optional[tuple] = None) -> Optional[str]:
        """A chat workspace in the Progress manager (``/tools/progress?out=<bid>``, Chapters tab): the
        Book page over that output folder, where Retranslate / Resolve QA run (U7). Returns the route id.
        ``select`` (U9 ``/retranslate <range>``): ``(start, end)`` chapters selected when the list loads."""
        app = self.app
        library = getattr(app, "library", None)
        navigate = getattr(app, "navigate_to", None)
        notify = getattr(app, "notify", None)
        if library is None or not hasattr(library, "bid_for") or navigate is None:
            if notify is not None:
                notify("The Progress manager is not available in this session")
            return None

        def target() -> Optional[dict]:
            workspace = _reader_workspace(str(folder or ""), str(source or ""))
            if not workspace:
                return None
            name = os.path.basename(os.path.normpath(workspace))
            progress = os.path.join(workspace, "translation_progress.json")
            book = {"name": name, "folder_name": name, "path": workspace, "output_folder": workspace,
                    "type": "in_progress", "is_in_progress": True, "in_library": False,
                    "progress_file": progress if os.path.isfile(progress) else ""}
            if source and os.path.isfile(str(source)):
                book["raw_source_path"] = str(source)
            return book

        book = await self._run_io(target)
        if book is None:
            if notify is not None:
                notify("No translation progress in this chat's workspace yet")
            return None
        bid = library.bid_for(book)
        if select:
            from glossarion_mobile.ui.library.chapters_tab import request_range_selection

            request_range_selection(bid, *select)
        navigate("tools.progress", None, {"out": bid})
        return bid

    async def open_external(self, path: str) -> Any:
        """Open externally: the system share / Open-in sheet on Android and iOS (no file:// launches
        there); the default app through ``UrlLauncher`` on desktop dev, else the share sheet."""
        if self.is_android or self.is_ios:
            return await self.share_files([path])
        try:
            import pathlib

            import flet as ft

            if self._url_launcher is None:
                self._url_launcher = ft.UrlLauncher()  # a page service: keep the reference
            await self._url_launcher.launch_url(pathlib.Path(path).resolve().as_uri())
            return True
        except Exception as exc:
            log.info("open externally failed (%s); sharing instead", exc)
            return await self.share_files([path])

    @staticmethod
    def _record_raw_blocking(source: str) -> None:
        """The raw file of a book that moved into the Library goes into the raw-inputs registry (the run
        set-up's ``library_core.record_library_raw_inputs``), so the Library always finds it. The file
        stays where it is (a copy into Library/Raw would rename it and break the raw-stem -> workspace
        rule Translate / Resume use)."""
        if not source or not os.path.isfile(str(source)):
            return
        try:
            import library_core  # shared (U5)

            library_core.record_library_raw_inputs([str(source)])
        except Exception:
            log.debug("recording the raw input failed", exc_info=True)

    async def after_migrate(self, target: str, source: str = "") -> None:
        """After a move the user confirmed (the Attachments merge dialog): the raw file is recorded in
        the Library and the snackbar offers "Open book"."""
        await self._run_io(self._record_raw_blocking, source)
        self._announce_moved(target, source)

    # ---- auto-migrate (UI_SPEC §2.17: finished chat books join the Library by themselves) -------

    def library_service(self) -> Any:
        """The LibraryService (``app.library``; installed after the chat), or None."""
        return getattr(self.app, "library", None)

    def _hook_auto_migrate(self) -> None:
        if self._migrate_hooked:
            return
        self._migrate_hooked = True
        self._unsubs.append(self.runs.subscribe_finished(self._on_run_finished))
        service = self.jobs.jobs
        on_transition = getattr(service, "on_transition", None) if service is not None else None
        if callable(on_transition):
            try:
                self._unsubs.append(on_transition(self._on_job_transition))
            except Exception:
                log.exception("subscribing to job transitions failed")

    def _on_run_finished(self, cid: str, run: Any, state: str) -> None:
        """ChatRuns finish thread, once the run is committed: a run that finished moves its book."""
        folder = str(getattr(run, "output_folder", "") or "")
        if str(state or "").upper() != "DONE" or not folder:
            return
        outcome = self.auto_migrate_blocking(cid, folder)
        if outcome["status"] in (MIGRATE_MOVED, MIGRATE_COLLISION):
            self._post(self._announce, outcome)

    def _on_job_transition(self, snap: Any, previous: Any = None) -> Any:
        """A job ended: once nothing runs, sweep (moves that waited for the jobs, and workspaces a job
        was writing). Returns the sweep task (None when there is nothing to do yet)."""
        if state_name(snap) not in TERMINAL_STATES or self._active_job() is not None:
            return None
        return self._spawn(self.sweep("idle"))

    async def startup_sweep(self, wait: float = STARTUP_LIBRARY_WAIT) -> list:
        """Once after install: waits (bounded) for the Library feature, then sweeps every chat."""
        deadline = time.monotonic() + max(0.0, float(wait))
        while self.library_service() is None and time.monotonic() < deadline:
            await asyncio.sleep(0.25)
        return await self.sweep("startup")

    async def sweep(self, reason: str = "") -> list:
        """Move every eligible ``Attachments/<stem>`` workspace of every chat (idempotent); a sweep
        asked for while one runs runs again after it. Returns the outcomes."""
        if self._sweeping:
            self._sweep_again = True
            return []
        self._sweeping = True
        outcomes: list = []
        try:
            while True:
                self._sweep_again = False
                batch = await self._run_io(self.sweep_blocking)
                moved = [o for o in batch if o.get("status") == MIGRATE_MOVED]
                for outcome in batch:
                    # several books at once (the first launch after an update): one summary snackbar
                    self._announce(outcome, quiet=len(moved) > 1)
                if len(moved) > 1:
                    self._announce_many(len(moved))
                outcomes.extend(batch)
                if not self._sweep_again:
                    break
        except Exception:
            log.exception("the Library auto-migrate sweep failed (%s)", reason or "sweep")
        finally:
            self._sweeping = False
        return outcomes

    def sweep_blocking(self) -> list:
        """Blocking: ``auto_migrate_blocking`` over every saved chat's attachment workspaces."""
        outcomes: list = []
        try:
            rows = list(self.chats.all() or [])
        except Exception:
            log.debug("listing the chats failed", exc_info=True)
            return outcomes
        for row in rows:
            cid = str(getattr(row, "cid", "") or "")
            if not cid or is_scratch_cid(cid):
                continue
            try:
                folders = list(self.chats.attachment_folders(cid) or [])
            except Exception:
                continue
            for folder in folders:
                outcome = self.auto_migrate_blocking(cid, folder)
                outcomes.append(outcome)
                if outcome["status"] == MIGRATE_DEFERRED and outcome.get("jobs_running"):
                    return outcomes  # nothing else can move before the jobs end either
        return outcomes

    def _job_lock(self) -> Any:
        """``JobService.backend.job_lock`` (``job_runner.JOB_LOCK``): every job holds it for its whole
        run, while its ``OUTPUT_DIRECTORY`` points at the job's own temporary run root."""
        service = self.jobs.jobs
        backend = getattr(service, "backend", None) if service is not None else None
        lock = getattr(backend, "job_lock", None) if backend is not None else None
        if lock is None:
            from glossarion_mobile.ui.chat.attachments import jobs_lock

            lock = jobs_lock()
        return lock

    def _active_job(self) -> Any:
        """``JobService.view().active`` (a job that has not ended, for services without ``view``)."""
        service = self.jobs.jobs
        view = getattr(service, "view", None) if service is not None else None
        if callable(view):
            try:
                return getattr(view(), "active", None)
            except Exception:
                return None
        snap = self.jobs.snapshot()
        return snap if snap is not None and state_name(snap) not in TERMINAL_STATES else None

    def _resumable_here(self, cid: str, folder: str) -> bool:
        """The chat's last job (the one Resume / Retry failed resubmit) ended resumable on ``folder``:
        moving it would leave the job card's Resume / Retry failed working on a folder that is gone."""
        ending, snapshot, workspace, source = self.runs.last_job_ending(cid)
        if ending not in RESUMABLE_ENDINGS:
            return False
        if not workspace and snapshot is not None:
            try:
                workspace = self.job_workspace(snapshot)
            except Exception:
                workspace = ""
        if workspace:
            return _norm(workspace) == _norm(folder)
        stem = os.path.splitext(os.path.basename(source))[0]
        return bool(stem) and stem.casefold() == os.path.basename(os.path.normpath(folder)).casefold()

    @staticmethod
    def _same_book(target: str, folder: str, source: str) -> bool:
        """The Library folder ``target`` is this book's own workspace: its ``source_epub.txt`` points at
        the attachment (same file, or the same content), and the translation it already holds is of the
        same chapters (``_same_chapters``). A path alone proves nothing: FileBridge reuses a freed Inbox
        name, so a different book can sit at the path a Library book's pointer names."""
        try:
            import library_core  # shared (U5)
        except Exception:
            return False
        read_pointer = getattr(library_core, "_read_source_epub_pointer", None)
        same_content = getattr(library_core, "_same_file_content", None)
        if read_pointer is None:
            return False
        pointed = read_pointer(target)
        if not pointed:
            return False
        for raw in (source, read_pointer(folder)):
            if not raw or not os.path.isfile(raw):
                continue
            if _norm(raw) == _norm(pointed) or (same_content is not None and same_content(raw, pointed)):
                return ChatFeature._same_chapters(folder, target)
        return False

    @staticmethod
    def _same_chapters(folder: str, target: str) -> bool:
        """``target`` holds no translation yet, or at least half of the chapters either workspace
        translated carry the same pipeline ``content_hash`` (``translation_progress.json``): the two
        translations are of the same book, so merging replaces nothing of another book's."""
        theirs = _chapter_hashes(target)
        if not theirs:
            return True
        mine = _chapter_hashes(folder)
        shared = len(mine & theirs)
        return bool(mine) and shared > 0 and shared * 2 >= min(len(mine), len(theirs))

    def auto_migrate_blocking(self, cid: Any, folder: str) -> dict:
        """Blocking: move one chat book's ``Attachments/<stem>`` workspace into the Library when it is
        safe (the desktop Migrate, ``ChatStoreAdapter.migrate_attachment``). Returns the outcome dict
        (``status`` one of the ``MIGRATE_*`` values, ``reason``, ``target``, ``source``).

        Only a managed attachment workspace of a saved chat that holds a translation of a book
        (``library_core.RAW_IMPORT_EXTENSIONS``) moves, while no job writes into it and the chat's last
        job does not offer Resume / Retry failed on it. The move itself runs with the jobs' process
        lock taken without waiting and no job active: the shared Migrate picks its destination from the
        live ``OUTPUT_DIRECTORY``, which a running job points at its temporary run root (deleted when
        that run ends). An existing Library folder of the same name is merged into only when it is the
        same book (its ``source_epub.txt`` points at the same raw file and any translation it holds is of
        the same chapters, ``_same_book``); otherwise the workspace stays
        and the user decides (the desktop "Attachment folder already exists" dialog).
        """
        cid = str(cid)
        folder = str(folder or "")
        outcome = {"cid": cid, "folder": folder, "status": MIGRATE_SKIPPED, "reason": "", "target": "", "source": ""}

        def done(status: str, reason: str = "", **extra: Any) -> dict:
            outcome.update(status=status, reason=reason, **extra)
            return outcome

        if not folder:
            return done(MIGRATE_SKIPPED, "no workspace")
        if is_scratch_cid(cid):
            return done(MIGRATE_SKIPPED, "a scratch chat is never saved implicitly")
        with self._migrate_lock:
            session = self.chats.session(cid)
            if session is None:
                return done(MIGRATE_SKIPPED, "the chat no longer exists")
            try:
                managed = self.chats.binding_for(cid).is_managed_attachment_workspace(session, folder)
            except Exception:
                managed = False
            if not managed:
                return done(MIGRATE_SKIPPED, "not an attachment workspace of this chat")
            if not os.path.isfile(os.path.join(folder, "translation_progress.json")):
                return done(MIGRATE_SKIPPED, "no translation in this workspace")
            from glossarion_mobile.ui.chat.attachments import attachment_source, workspace_busy

            try:
                import library_core  # shared (U5)
            except Exception:
                return done(MIGRATE_SKIPPED, "the Library is not available")
            source = attachment_source(self.chats.messages(cid), folder)
            if not source:
                read_pointer = getattr(library_core, "_read_source_epub_pointer", None)
                source = str((read_pointer(folder) if read_pointer is not None else "") or "")
            outcome["source"] = source
            book_extensions = tuple(getattr(library_core, "RAW_IMPORT_EXTENSIONS", ()) or ())
            if not source or os.path.splitext(source)[1].lower() not in book_extensions:
                return done(MIGRATE_SKIPPED, "not a book")
            if workspace_busy(self.runs, self.jobs, cid, folder):
                return done(MIGRATE_DEFERRED, "a job is writing this workspace")
            if self._resumable_here(cid, folder):
                return done(MIGRATE_SKIPPED, "its run can still be resumed")
            lock = self._job_lock()
            if lock is not None and not lock.acquire(blocking=False):
                return done(MIGRATE_DEFERRED, "a job is running", jobs_running=True)
            try:
                if self._active_job() is not None:
                    return done(MIGRATE_DEFERRED, "a job is running", jobs_running=True)
                target = self.chats.migration_target(cid, folder)
                outcome["target"] = target
                if not target:
                    return done(MIGRATE_SKIPPED, "no Library output folder")
                run_roots = [r for r in (self.runs.temp_dir, self._temp_dir) if r]
                if any(_inside(target, root) for root in run_roots):
                    # OUTPUT_DIRECTORY still names a run root: not the real output folder
                    return done(MIGRATE_DEFERRED, "the output folder points at a run root", jobs_running=True)
                merge = os.path.exists(target)
                if merge and _norm(folder) not in self._merge_approved and not self._same_book(target, folder, source):
                    return done(MIGRATE_COLLISION, "a different Library book has this name")
                result = self.chats.migrate_attachment(cid, folder, (lambda _target: True) if merge else None)
            finally:
                if lock is not None:
                    lock.release()
            if not result.get("ok"):
                notices = list(result.get("notices") or [])
                text = f"{notices[-1].get('title')}: {notices[-1].get('text')}" if notices else "the move failed"
                return done(MIGRATE_FAILED, text)
            from glossarion_mobile.ui.chat.attachments import migrated_target

            target = migrated_target(result) or target
            self._merge_approved.discard(_norm(folder))
            self._record_raw_blocking(source)
            self.runs.relocate_output(cid, folder, target)
            return done(MIGRATE_MOVED, "merged" if merge else "moved", target=target)

    async def auto_migrate(self, cid: Any, folder: str) -> dict:
        """``auto_migrate_blocking`` on the io pool, then its snackbar."""
        outcome = await self._run_io(self.auto_migrate_blocking, cid, folder)
        self._announce(outcome)
        return outcome

    def _announce(self, outcome: dict, *, quiet: bool = False) -> None:
        """UI loop: a moved book's snackbar ("Added to the Library" · Open book; ``quiet``: the chat and
        the Library are refreshed but the caller shows one summary), a name clash's (its action opens
        the merge dialog; once per workspace and session)."""
        status = outcome.get("status")
        if status == MIGRATE_MOVED:
            self._announce_moved(outcome.get("target", ""), outcome.get("source", ""), outcome.get("cid"),
                                 quiet=quiet)
        elif status == MIGRATE_COLLISION:
            key = _norm(outcome.get("folder", ""))
            if key in self._collisions_shown:
                return
            self._collisions_shown.add(key)
            notify = getattr(self.app, "notify", None)
            name = os.path.basename(os.path.normpath(str(outcome.get("target") or outcome.get("folder") or "")))
            if notify is not None:
                cid, folder = outcome.get("cid"), outcome.get("folder")
                notify(COLLISION_TEXT.format(name=name), "Merge…", lambda: self.open_merge_dialog(cid, folder))

    def _workspace_bid(self, target: str, source: str = "") -> Optional[str]:
        library = self.library_service()
        if library is None or not target or not hasattr(library, "bid_for"):
            return None
        name = os.path.basename(os.path.normpath(target))
        return library.bid_for({"name": name, "folder_name": name, "path": target, "output_folder": target,
                                "type": "in_progress", "is_in_progress": True, "in_library": True,
                                **({"raw_source_path": str(source)} if source else {})})

    def _announce_many(self, count: int) -> None:
        notify = getattr(self.app, "notify", None)
        navigate = getattr(self.app, "navigate_to", None)
        if notify is None:
            return
        text = f"Added {count} books to the Library"
        if navigate is not None:
            notify(text, "Library", lambda: navigate("library"))
        else:
            notify(text)

    def _announce_moved(self, target: str, source: str = "", cid: Any = None, *, quiet: bool = False) -> None:
        app = self.app
        library = self.library_service()
        notify = getattr(app, "notify", None)
        navigate = getattr(app, "navigate_to", None)
        if library is not None:
            try:
                library.mark_dirty()
            except Exception:
                pass
        try:
            bid = self._workspace_bid(target, source)
        except Exception:
            log.debug("the Library id of %s failed", target, exc_info=True)
            bid = None
        chat_view = getattr(app, "chat_view", None)
        if chat_view is not None:
            hook = getattr(chat_view, "on_workspace_migrated", None) if cid is not None else None
            try:
                if callable(hook):
                    hook(str(cid), target, source)
                else:
                    if cid is not None and str(getattr(chat_view, "cid", "")) == str(cid):
                        chat_view.render_transcript()  # the turn's stored paths now point into the Library
                    chat_view.header.set_attachments(chat_view._attachment_count(chat_view.cid))
            except Exception:
                log.debug("refreshing the chat after a move failed", exc_info=True)
        if notify is not None and not quiet:
            if bid and navigate is not None:
                notify(ADDED_TO_LIBRARY, "Open book", lambda: navigate("library.book", {"bid": bid}))
            else:
                notify(ADDED_TO_LIBRARY)

    def open_merge_dialog(self, cid: Any, folder: str) -> Any:
        """The name-clash snackbar's action: the desktop "Attachment folder already exists" dialog."""
        from glossarion_mobile.ui.chat.attachments import show_merge_dialog

        target = self.chats.migration_target(cid, folder)
        return show_merge_dialog(self.page, target, lambda: self.merge_into_library(cid, folder))

    async def merge_into_library(self, cid: Any, folder: str) -> dict:
        """"Merge and replace": the workspace moves into the same-named Library folder (now, or when
        the running jobs end)."""
        self._merge_approved.add(_norm(folder))
        outcome = await self._run_io(self.auto_migrate_blocking, cid, folder)
        notify = getattr(self.app, "notify", None)
        if outcome["status"] == MIGRATE_MOVED:
            self._announce(outcome)
        elif notify is not None:
            if outcome["status"] == MIGRATE_DEFERRED:
                notify("The book joins the Library when the running job finishes")
            else:
                notify(f"Not added to the Library: {outcome.get('reason') or 'unknown reason'}")
        return outcome

    async def library_book(self, folder: str) -> Optional[str]:
        """The Library id of the book whose workspace is ``folder`` (its ``output_folder``, or the one
        ``LibraryService.workspace_for`` resolves); a quiet rescan first when the Library is stale."""
        library = self.library_service()
        if library is None or not folder or not hasattr(library, "bid_for"):
            return None
        snap = getattr(library, "snapshot", None)
        if getattr(library, "dirty", False) or not getattr(snap, "scanned_at", 0):
            try:
                await library.refresh(quiet=True, reason="chat")
            except Exception:
                log.debug("refreshing the Library failed", exc_info=True)
            snap = getattr(library, "snapshot", None)
        workspace_for = getattr(library, "workspace_for", None)
        wanted = _norm(folder)
        for book in (snap.all_books() if snap is not None else ()):
            workspace = str(book.get("output_folder") or "")
            if not workspace and callable(workspace_for):
                try:
                    workspace = str(workspace_for(book) or "")
                except Exception:
                    workspace = ""
            if workspace and _norm(workspace) == wanted:
                return library.bid_for(book)
        return None

    async def library_translate(self, folder: str, source: str = "") -> Any:
        """Resume / Retry failed of a chat book that moved into the Library: the Library translate sheet
        for it (the Library resumes from the moved workspace's progress file)."""
        notify = getattr(self.app, "notify", None)
        library = self.library_service()
        feature = getattr(self.app, "library_feature", None)
        bid = await self.library_book(folder)
        book = library.book_for_bid(bid) if bid and library is not None and hasattr(library, "book_for_bid") else None
        if book is None or feature is None or not hasattr(feature, "context"):
            if notify is not None:
                notify("This book is not in the Library yet")
            return None
        if source and not book.get("raw_source_path") and os.path.isfile(str(source)):
            book = dict(book, raw_source_path=str(source))
        from glossarion_mobile.ui.library.translate_sheet import open_translate_sheet

        return await open_translate_sheet(feature.context(), [book])

    def job_workspace(self, snap: Any) -> str:
        """A chat job's persisted ``Direct Text/<chat>/Attachments/<stem>`` workspace (Jobs › job › Files): the
        folder of the responses of the job's user turn (``params['user_index']``, ``chat_ops.turn_workspace``),
        or the Library folder it moved to (auto-migrate, ``chat_ops.moved_workspace``), else the chat's
        attachment folder named after the attachment; '' when there is none."""
        from glossarion_mobile.ui.chat.chat_ops import moved_workspace, turn_span, turn_workspace

        env = self.env
        chats = getattr(env, "chats", None) if env is not None else None
        spec = getattr(snap, "spec", None)
        origin = dict(getattr(spec, "origin", None) or {})
        params = dict(getattr(spec, "params", None) or {})
        cid = origin.get("cid")
        if chats is None or cid is None:
            return ""
        try:
            messages = list(chats.messages(cid) or [])
        except Exception:
            return ""
        try:
            user_index = int(params.get("user_index"))
        except (TypeError, ValueError):
            user_index = -1
        span = turn_span(messages, user_index) if 0 <= user_index < len(messages) else []
        folder = turn_workspace(messages, span[1:]) or moved_workspace(messages, span[1:])
        if folder:
            return folder
        if span:
            message = messages[user_index]
            if len(message) > 2 and str(message[0]) == "user_file":
                stem = os.path.splitext(os.path.basename(str(message[2] or "")))[0].lower()
                try:
                    folders = list(chats.attachment_folders(cid) or [])
                except Exception:
                    folders = []
                for candidate in folders:
                    if stem and os.path.basename(os.path.normpath(candidate)).lower() == stem:
                        return str(candidate)
        return ""

    def open_output(self, folder: str) -> Optional[str]:
        """A chat output folder in Files (``tools.files.folder`` with the folder's file ref, like Jobs ›
        Files and the Book page): under the "chats" root (``Output/Direct Text``) when it lies there,
        else the "output" root; the root itself when the folder is outside both. Returns the route."""
        navigate = getattr(self.app, "navigate_to", None)
        if navigate is None:
            return None
        jobs = getattr(self.app, "jobs", None)
        roots_fn = getattr(jobs, "file_roots", None)
        try:
            roots = dict(roots_fn() or {}) if callable(roots_fn) else {}
        except Exception:
            roots = {}
        prefs = getattr(self.app, "prefs", None)
        folder = str(folder or "")
        from glossarion_mobile.ui.screens.files import root_for

        root = root_for(folder, roots, ("chats", "output")) if folder else None
        if root is None or prefs is None or not hasattr(prefs, "file_ref"):
            navigate("tools.files", {"root": root or "output"})
            return "tools.files"
        navigate("tools.files.folder", {"root": root, "fid": prefs.file_ref(folder)})
        return "tools.files.folder"

    def push_overlay(self, view: Any) -> None:
        shell = getattr(self.app, "shell", None)
        if shell is not None:
            shell.push_overlay(view)
            try:
                self.page.update()
            except Exception:
                pass

    def pop_overlay(self, view: Any) -> None:
        shell = getattr(self.app, "shell", None)
        if shell is None:
            return
        shell.pop_view(view)
        try:
            self.page.update()
        except Exception:
            pass

    # ---- drawer row actions -------------------------------------------------------------------

    def _extra_chat_actions(self, chat: Any) -> list:
        items: list = []
        for provider in list(self.chat_action_providers):
            try:
                items.extend(provider(chat) or ())
            except Exception:
                log.exception("chat action provider failed")
        return items

    def chat_actions(self, chat: Any) -> Any:
        from glossarion_mobile.ui.components.action_sheet import ActionItem, ActionSheet

        app = self.app
        chat_view = getattr(app, "chat_view", None)

        def rename() -> None:
            self._select_chat(chat.cid)
            if chat_view is not None:
                chat_view.open_rename()

        def delete() -> None:
            self._select_chat(chat.cid)
            if chat_view is not None:
                chat_view.confirm_delete()

        def attachments() -> None:
            navigate = getattr(app, "navigate_to", None)
            if navigate is not None:
                navigate("chat.attachments", {"cid": chat.cid})

        def export() -> None:
            if chat_view is not None:
                chat_view.open_export(chat.cid)

        no_view = None if chat_view is not None else "The chat is not available"
        scratch = bool(getattr(chat, "scratch", False))
        items = [ActionItem("Rename", rename, icon="DRIVE_FILE_RENAME_OUTLINE")]
        if scratch:
            items += [
                ActionItem("Save", (lambda: chat_view.save_scratch(chat.cid)) if chat_view is not None else None,
                           icon="SAVE", disabled_reason=no_view),
                ActionItem(f"Attachments ({chat.attachments})", attachments, icon="ATTACH_FILE"),
                ActionItem("Export chat", export, icon="IOS_SHARE", disabled_reason=no_view),
                ActionItem("Discard", delete, icon="DELETE_OUTLINE", destructive=True),
            ]
        else:
            items += [
                ActionItem("Unpin" if chat.pinned else "Pin", lambda: self.chats.set_pinned(chat.cid, not chat.pinned),
                           icon="PUSH_PIN"),
                # U9 Series: "Move to Series…" (SeriesFeature; none without it)
                *self._extra_chat_actions(chat),
                ActionItem(f"Attachments ({chat.attachments})", attachments, icon="ATTACH_FILE"),
                ActionItem("Export chat", export, icon="IOS_SHARE", disabled_reason=no_view),
                ActionItem("Duplicate as scratch",
                           (lambda: chat_view.duplicate_as_scratch(chat.cid)) if chat_view is not None else None,
                           icon="CONTENT_COPY", disabled_reason=no_view),
                ActionItem("Delete", delete, icon="DELETE_OUTLINE", destructive=True,
                           disabled_reason=None if self.chats.can_delete(chat.cid) else "Nothing to delete"),
            ]
        sheet = ActionSheet(
            items,
            title=chat.title,
            tablet=bool(getattr(getattr(app, "shell", None), "tablet", False)),
        )
        sheet.show(self.page)
        return sheet

    def close(self) -> None:
        for unsub in self._unsubs:
            try:
                unsub()
            except Exception:
                pass
        self._unsubs = []
        self.runs.detach()
        self.chats.close()
