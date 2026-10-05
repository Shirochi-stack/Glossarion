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
5. lifecycle INACTIVE / HIDE / PAUSE / DETACH flush the chat history.

``maybe_show_welcome()`` opens ``/welcome`` on a first run (no
``glossary_mode_dialog_shown`` in config.json and no completed mobile welcome); the
app calls it after its initial route.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Callable, Optional

from glossarion_mobile.state.chat_store_adapter import ChatStoreAdapter, default_history_path
from glossarion_mobile.ui.chat.context import ChatEnv
from glossarion_mobile.ui.chat.job_binding import JobsAdapter
from glossarion_mobile.ui.chat.run_controller import ChatRuns
from glossarion_mobile.ui.router import RouteMatch

__all__ = ["ChatFeature", "FLUSH_LIFECYCLE_STATES", "SCREEN_ROUTES"]

log = logging.getLogger("glossarion.chat")

FLUSH_LIFECYCLE_STATES = ("inactive", "hide", "pause", "detach")
SCREEN_ROUTES = ("settings.accounts", "welcome", "chat.message.edit")
WELCOME_PREF = "welcome_completed"


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


class ChatFeature:
    def __init__(self, app: Any, *, chats: Optional[ChatStoreAdapter] = None, jobs: Any = None, oauth: Any = None) -> None:
        self.app = app
        self.page = getattr(app, "page", None)
        self.dispatcher = getattr(app, "dispatcher", None)
        paths = getattr(app, "paths", None)
        platform = str(getattr(getattr(self.page, "platform", None), "value", "") or "")
        self.is_android = platform == "android"
        self.is_ios = platform == "ios"
        self.chats = chats or ChatStoreAdapter(history_path=default_history_path())
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
            pick_files=self.pick_files,
            import_file=self.import_file,
            push_overlay=self.push_overlay,
            pop_overlay=self.pop_overlay,
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
        self.runs.attach()
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

            def show(match: RouteMatch) -> bool:
                if match.name in ("chat", "chat.message"):
                    self._select_chat(match.params.get("cid"))
                return original_show(match)

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
                env.config_set_many(updates)
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

    def open_output(self, folder: str) -> None:
        opener = getattr(getattr(self.app, "files", None), "open_folder", None)
        if callable(opener):
            opener(folder)
            return
        navigate = getattr(self.app, "navigate_to", None)
        if navigate is not None:
            navigate("tools.files", {"root": "output"})

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

        sheet = ActionSheet(
            [
                ActionItem("Rename", rename, icon="DRIVE_FILE_RENAME_OUTLINE"),
                ActionItem("Unpin" if chat.pinned else "Pin", lambda: self.chats.set_pinned(chat.cid, not chat.pinned),
                           icon="PUSH_PIN"),
                ActionItem(f"Attachments ({chat.attachments})", disabled_reason="Arrives in U7", icon="ATTACH_FILE"),
                ActionItem("Export chat", disabled_reason="Arrives in U7", icon="IOS_SHARE"),
                ActionItem("Duplicate as scratch", disabled_reason="Arrives in U7", icon="CONTENT_COPY"),
                ActionItem("Delete", delete, icon="DELETE_OUTLINE", destructive=True,
                           disabled_reason=None if self.chats.can_delete(chat.cid) else "Nothing to delete"),
            ],
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
