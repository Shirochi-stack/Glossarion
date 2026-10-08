"""SeriesFeature: the optional, mobile-only Series (UI_SPEC §2.15, Appendix B), wired into the app.

``await SeriesFeature.install(app)`` - call it in ``GlossarionApp.start`` after the chat, the
Library and the Glossary Manager (``_install_series`` after the last ``_install_*``):

1. loads ``mobile_series.json`` (beside the chat sidecar, ``state.series.SeriesStore``) on the
   io pool and exposes the feature as ``app.series`` (``current()`` for the Library screens);
2. layers each chat's series defaults beneath its own overrides
   (``ChatStoreAdapter.override_layers``): Global config -> Series defaults -> per-chat
   overrides; runs, the header subtitle / "custom" chip and the option pills read the result;
3. the drawer "Series" section + "Series" search group (``ChatDrawer.series``), the chat ⋯ menu
   and the drawer long-press sheet "Move to Series…", the Chat settings "Series" section and its
   "Inherited from: Series <name>" labels (``chat_settings.SERIES_HOOKS``);
4. the ``/series/<sid>`` page (``series_page.SeriesScreen``) in the shell's screen factory;
5. the Library: "Add to Series" (Book page ⋯, selection › More) and the Series filter.

No backend behaviour is added: series defaults are the per-chat override keys, applied exactly
like a chat override. Scratch chats never join a series.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Callable, Optional, Sequence

from glossarion_mobile.state.chat_store_adapter import is_scratch_cid
from glossarion_mobile.state.series import (
    SERIES_FILE,
    SeriesDefaultsChats,
    SeriesStore,
    chat_series_id,
    member_chats,
    move_chat,
    series_rows,
    series_search,
)
from glossarion_mobile.ui.router import RouteMatch

__all__ = ["IMPLEMENTED_ROUTES", "SCREEN_ROUTES", "SeriesFeature", "current"]

log = logging.getLogger("glossarion.series")

SCREEN_ROUTES = ("series",)
#: Routes this feature ships (Integrate merges them into the implemented sets).
IMPLEMENTED_ROUTES = frozenset(SCREEN_ROUTES)

_CURRENT: dict = {"feature": None}


def current() -> Optional["SeriesFeature"]:
    """The installed SeriesFeature (None without Series): the Library asks it for series."""
    return _CURRENT["feature"]


class SeriesFeature:
    def __init__(self, app: Any, *, store: Optional[SeriesStore] = None) -> None:
        self.app = app
        self.page = getattr(app, "page", None)
        self.dispatcher = getattr(app, "dispatcher", None)
        self.store = store or SeriesStore(self._default_path())
        self._fallback_factory: Optional[Callable[[RouteMatch], Any]] = None
        self._unsubs: list = []
        self._layer: Optional[Callable[[Any], dict]] = None
        self.sheet: Any = None
        self.dialog: Any = None
        self.screens_built: list = []
        # The series glossary this feature last pre-filled on the chat view (``_prefill_glossary``): the
        # view's ``last_manual_glossary`` is one value for every chat, so it is ours only while its path
        # still equals this one; a glossary the user picked in the sheet is never replaced.
        self._prefilled_view: Any = None
        self._prefilled_path: Optional[str] = None

    # ---- install ------------------------------------------------------------------------------

    def _default_path(self) -> str:
        paths = getattr(self.app, "paths", None)
        data = getattr(paths, "data", None) if paths is not None else None
        if data:
            return os.path.join(str(data), SERIES_FILE)
        chats = self.chats
        sidecar = getattr(chats, "sidecar_path", None) if chats is not None else None
        if sidecar:
            return os.path.join(os.path.dirname(str(sidecar)), SERIES_FILE)
        return os.path.join(os.getcwd(), SERIES_FILE)

    @classmethod
    async def install(cls, app: Any, **kwargs: Any) -> "SeriesFeature":
        feature = cls(app, **kwargs)
        await feature.run_io(feature.store.load)
        feature.attach()
        return feature

    def attach(self) -> None:
        app = self.app
        app.series = self
        _CURRENT["feature"] = self
        self.store.post = self._post
        chats = self.chats
        if chats is not None and hasattr(chats, "override_layers") and self._layer is None:
            self._layer = self.defaults_for_chat
            chats.override_layers.append(self._layer)
        drawer = getattr(app, "drawer", None)
        if drawer is not None:
            drawer.series = self
        header = self._header()
        if header is not None and hasattr(header, "set_series_handler"):
            header.set_series_handler(self.move_current_chat)
        chat_feature = getattr(app, "chat_feature", None)
        providers = getattr(chat_feature, "chat_action_providers", None)
        if isinstance(providers, list) and self.chat_actions not in providers:
            providers.append(self.chat_actions)
        try:
            from glossarion_mobile.ui.sheets import chat_settings

            chat_settings.SERIES_HOOKS["inherited_label"] = self.inherited_label
            chat_settings.SERIES_HOOKS["section"] = self.settings_section
        except Exception:
            log.exception("hooking the Chat settings sheet failed")
        shell = getattr(app, "shell", None)
        if shell is not None and self._fallback_factory is None:
            self._fallback_factory = shell.screen_factory
            shell.screen_factory = self.screen_factory
        self._unsubs.append(self.store.subscribe(self._on_series_changed))
        state = getattr(app, "state", None)
        if state is not None:
            try:
                self._unsubs.append(state.current_chat.subscribe(lambda cid: self._prefill_glossary(cid)))
            except Exception:
                pass
        self._refresh_drawer()

    def close(self) -> None:
        for unsub in self._unsubs:
            try:
                unsub()
            except Exception:
                pass
        self._unsubs = []
        chats = self.chats
        if chats is not None and self._layer is not None:
            try:
                chats.override_layers.remove(self._layer)
            except (AttributeError, ValueError):
                pass
        self._layer = None
        try:
            from glossarion_mobile.ui.sheets import chat_settings

            chat_settings.SERIES_HOOKS.update({"inherited_label": None, "section": None})
        except Exception:
            pass
        drawer = getattr(self.app, "drawer", None)
        if drawer is not None and getattr(drawer, "series", None) is self:
            drawer.series = None
        if _CURRENT["feature"] is self:
            _CURRENT["feature"] = None

    # ---- app access ---------------------------------------------------------------------------

    @property
    def chats(self) -> Any:
        feature = getattr(self.app, "chat_feature", None)
        if feature is not None and getattr(feature, "chats", None) is not None:
            return feature.chats
        state = getattr(self.app, "state", None)
        chats = getattr(state, "chats", None) if state is not None else None
        return chats if chats is not None and hasattr(chats, "set_meta") else None

    @property
    def chat_view(self) -> Any:
        return getattr(self.app, "chat_view", None)

    def _header(self) -> Any:
        view = self.chat_view
        return getattr(view, "header", None) if view is not None else None

    @property
    def tablet(self) -> bool:
        return bool(getattr(getattr(self.app, "shell", None), "tablet", False))

    def _post(self, fn: Callable[[], Any]) -> None:
        dispatcher = self.dispatcher
        if dispatcher is not None and getattr(dispatcher, "bound", False) and not dispatcher.on_loop_thread():
            dispatcher.post(fn)
        else:
            fn()

    async def run_io(self, fn: Callable[..., Any], *args: Any) -> Any:
        dispatcher = self.dispatcher
        if dispatcher is not None and getattr(dispatcher, "bound", False):
            return await dispatcher.run_in_thread(fn, *args, name="gl-series-io")
        return await asyncio.to_thread(fn, *args)

    def spawn(self, coro: Any) -> Any:
        dispatcher = self.dispatcher
        if dispatcher is not None and getattr(dispatcher, "bound", False):
            return dispatcher.spawn(coro)
        try:
            return asyncio.ensure_future(coro)
        except RuntimeError:
            coro.close()
            return None

    @staticmethod
    def push(*controls: Any) -> None:
        for control in controls:
            if control is None:
                continue
            try:
                control.update()
            except Exception:
                pass

    def say(self, message: str) -> None:
        notify = getattr(self.app, "notify", None)
        if callable(notify):
            notify(message)
        else:
            log.info("series: %s", message)

    def go(self, name: str, params: Optional[dict] = None, *, reset: bool = False) -> None:
        navigate = getattr(self.app, "navigate_to", None)
        if callable(navigate):
            navigate(name, params, reset=reset)

    def show(self, dialog: Any) -> Any:
        if self.page is not None:
            dialog.show(self.page)
        return dialog

    # ---- layering -----------------------------------------------------------------------------

    def defaults_for_chat(self, cid: Any) -> dict:
        """The override layer: the chat's series defaults ({} outside a series)."""
        sid = chat_series_id(self.chats, cid, self.store)
        return self.store.defaults(sid) if sid else {}

    def series_of(self, cid: Any) -> Any:
        sid = chat_series_id(self.chats, cid, self.store)
        return self.store.get(sid) if sid else None

    def inherited_label(self, chats: Any, cid: Any, field_name: str) -> Optional[str]:
        """Chat settings: "Series <name>" when the chat's series sets ``field_name``."""
        item = self.series_of(cid)
        if item is not None and item.defaults.get(field_name) is not None:
            return f"Series {item.name}"
        return None

    # ---- drawer provider ----------------------------------------------------------------------

    def rows(self) -> list:
        return series_rows(self.store, self.chats)

    def search(self, query: str) -> list:
        return series_search(self.store, self.chats, query)

    def open_series(self, sid: str) -> None:
        self.go("series", {"sid": sid}, reset=True)

    def open_chat(self, cid: str) -> None:
        opener = getattr(self.app, "_open_chat", None)
        if callable(opener):
            opener(str(cid))
        else:
            self.go("chat", {"cid": str(cid)})

    def new_chat_in_series(self, sid: str) -> Optional[str]:
        """"＋ New chat in series": a new (or the reused empty) chat, put in the series, opened."""
        chats = self.chats
        if chats is None or not self.store.has(sid):
            self.say("New chats need the chat store")
            return None
        cid = str(chats.new_chat())
        move_chat(chats, cid, sid, self.store)
        self.open_chat(cid)
        return cid

    def _refresh_drawer(self) -> None:
        drawer = getattr(self.app, "drawer", None)
        changed = getattr(drawer, "_changed", None) if drawer is not None else None
        if callable(changed):
            changed()

    def _on_series_changed(self) -> None:
        self._refresh_drawer()
        self._refresh_chat()

    def _refresh_chat(self) -> None:
        view = self.chat_view
        apply = getattr(view, "apply_settings_changed", None) if view is not None else None
        if callable(apply) and getattr(view, "bound", False):
            try:
                apply()
            except Exception:
                log.exception("refreshing the chat after a series change failed")
        self._prefill_glossary()

    def _prefill_glossary(self, cid: Any = None) -> None:
        """A series' manual glossary prefills the chat's "Provide Manual Glossary" sheet.

        Runs on every chat switch and series change. The sheet's file is replaced only while it is
        empty or still the one this feature put there: another series' chat gets its own file, a
        chat without a series glossary gets an empty sheet, and a file the user picked stays."""
        view = self.chat_view
        if view is None or not getattr(view, "bound", False):
            return
        cid = cid if cid is not None else getattr(view, "cid", None)
        item = self.series_of(cid)
        path = str(item.defaults.get("manual_glossary_path") or "") if item is not None else ""
        last = getattr(view, "last_manual_glossary", None)
        ours = self._prefilled_path if self._prefilled_view is view else None
        if last is not None and not (ours and getattr(last, "path", None) == ours):
            return  # picked in the sheet (or "Use as manual glossary" for this chat): not the series' to replace
        if not path:
            if last is not None:
                view.last_manual_glossary = None
            self._prefilled_view, self._prefilled_path = None, None
            return
        try:
            from glossarion_mobile.ui.chat.direct_text_rules import ManualGlossarySource

            view.last_manual_glossary = ManualGlossarySource("path", path=path,
                                                             extension=os.path.splitext(path)[1].lower())
            self._prefilled_view, self._prefilled_path = view, path
        except Exception:
            log.exception("prefilling the series glossary failed")

    # ---- move to series -----------------------------------------------------------------------

    def move_current_chat(self) -> Any:
        view = self.chat_view
        cid = getattr(view, "cid", None) if view is not None else None
        if cid is None:
            state = getattr(self.app, "state", None)
            cid = state.current_chat.value if state is not None else None
        return self.move_chat_sheet(cid)

    def move_chat_sheet(self, cid: Any, title: Optional[str] = None) -> Any:
        """"Move to Series…" for one chat (⋯ menu, long-press sheet, Chat settings)."""
        from glossarion_mobile.ui.chat.series_sheets import SeriesPickerSheet

        chats = self.chats
        if chats is None or cid is None:
            self.say("Series need the chat store")
            return None
        summary = chats.get(cid) if hasattr(chats, "get") else None
        if getattr(summary, "scratch", False) or is_scratch_cid(cid):
            self.say("Save the scratch chat first: scratch chats cannot join a series")
            return None
        current_sid = chat_series_id(chats, cid, self.store)

        def pick(sid: Optional[str]) -> None:
            self._move(cid, sid)

        def new() -> None:
            self.edit_series(None, on_created=lambda item: self._move(cid, item.id))

        self.sheet = SeriesPickerSheet(series=self.store.all(), current=current_sid, title="Move to Series",
                                       subtitle=title or getattr(summary, "title", None), on_pick=pick, on_new=new,
                                       allow_remove=True, tablet=self.tablet)
        return self.show(self.sheet)

    def _move(self, cid: Any, sid: Optional[str]) -> None:
        if move_chat(self.chats, cid, sid, self.store):
            item = self.store.get(sid) if sid else None
            self.say(f"Moved to {item.name}" if item is not None else "Removed from the series")
        self._refresh_drawer()
        self._refresh_chat()

    def chat_actions(self, chat: Any) -> list:
        """Drawer long-press sheet row (after Pin)."""
        from glossarion_mobile.ui.components.action_sheet import ActionItem

        if getattr(chat, "scratch", False):
            return []
        return [ActionItem("Move to Series…", lambda c=chat: self.move_chat_sheet(c.cid, c.title),
                           icon="DRIVE_FILE_MOVE_OUTLINE")]

    def settings_section(self, sheet: Any) -> Any:
        """Chat settings section 5 "Series": current series · Move to Series… · Series page."""
        import flet as ft

        from glossarion_mobile.ui.chat.series_sheets import color_dot

        cid = getattr(sheet, "cid", None)
        item = self.series_of(cid)
        if item is not None:
            current = ft.Row([color_dot(item.color_hex, 14), ft.Text(item.name, expand=True, max_lines=2,
                                                                       overflow=ft.TextOverflow.ELLIPSIS)],
                             spacing=8)
            note = "Its defaults sit between All chats and this chat."
        else:
            current = ft.Text("Not in a series", theme_style=ft.TextThemeStyle.BODY_MEDIUM)
            note = "A series gives its chats shared defaults (model, profile, language, glossary)."

        def move(e: Any = None) -> None:
            close = getattr(sheet, "close", None)
            if callable(close):
                close()
            self.move_chat_sheet(cid)

        def page(e: Any = None) -> None:
            close = getattr(sheet, "close", None)
            if callable(close):
                close()
            if item is not None:
                self.go("series", {"sid": item.id})

        buttons = [ft.TextButton(content="Move to Series…", on_click=move, key="settings-move-series")]
        if item is not None:
            buttons.append(ft.TextButton(content="Series page ›", on_click=page, key="settings-series-page"))
        return ft.ExpansionTile(
            title="Series",
            controls=[current, ft.Text(note, theme_style=ft.TextThemeStyle.BODY_SMALL, color=ft.Colors.ON_SURFACE_VARIANT),
                      ft.Row(buttons, wrap=True, spacing=4)],
        )

    # ---- series edit --------------------------------------------------------------------------

    def edit_series(self, sid: Optional[str], *, on_created: Optional[Callable[[Any], Any]] = None,
                    book_ids: Sequence[str] = ()) -> Any:
        """New series (``sid`` None; ``book_ids`` linked on create) or edit an existing one."""
        from glossarion_mobile.ui.chat.series_sheets import SeriesEditorDialog

        item = self.store.get(sid) if sid else None
        linked = list(item.book_ids) if item is not None else list(book_ids)
        books = [(bid, self.book_title(bid)) for bid in linked]

        def save(values: dict) -> None:
            if item is None:
                created = self.store.create(values["name"], color=values["color"], book_ids=linked,
                                            cover_bid=values["cover_bid"])
                if on_created is not None:
                    on_created(created)
                else:
                    self.say(f"Created {created.name}")
            else:
                self.store.update(item.id, name=values["name"], color=values["color"], cover_bid=values["cover_bid"])

        self.dialog = SeriesEditorDialog(
            title="Edit series" if item is not None else "New series",
            name=item.name if item is not None else "",
            color=item.color if item is not None else None,
            books=books,
            cover_bid=item.cover_bid if item is not None else (linked[0] if linked else ""),
            on_save=save,
            on_delete=(lambda: self.delete_series(item.id)) if item is not None else None,
            save_label="Save" if item is not None else "Create",
        )
        return self.show(self.dialog)

    def delete_series(self, sid: str) -> bool:
        """Delete a series: its chats stay (they leave it), linked books stay in the Library."""
        item = self.store.get(sid)
        if item is None:
            return False
        chats = self.chats
        for chat in member_chats(chats, sid, self.store):
            move_chat(chats, chat.cid, None)
        self.store.delete(sid)
        self.say(f"Deleted {item.name}")
        shell = getattr(self.app, "shell", None)
        top = getattr(shell, "top_screen", None) if shell is not None else None
        if getattr(top, "sid", None) == sid:  # leave the deleted series' page
            self.go("home", reset=True)
        return True

    def open_defaults_sheet(self, sid: str) -> Any:
        """Series page › Defaults: the Chat settings sheet in series scope."""
        from glossarion_mobile.ui.sheets.chat_settings import ChatSettingsSheet

        item = self.store.get(sid)
        if item is None:
            return None
        view = self.chat_view
        env = getattr(view, "env", None) if view is not None else None
        config = getattr(env, "store", None) or getattr(self.app, "config_store", None)
        if config is None:
            self.say("Series defaults need the settings store")
            return None
        profiles = []
        languages = []
        try:
            profiles = list(view._profiles()) if view is not None and hasattr(view, "_profiles") else []
        except Exception:
            profiles = []
        languages = list(getattr(env, "languages", None) or [])
        sheet = ChatSettingsSheet(
            cid=sid, config=config, chats=SeriesDefaultsChats(self.store, sid), profiles=profiles,
            languages=languages, subject="series", title=f"{item.name} · defaults",
            on_choose_model=lambda scope: self._choose_model(sid, scope, config),
        )
        self.sheet = sheet
        return self.show(sheet)

    def _choose_model(self, sid: str, scope: str, config: Any) -> Any:
        """Series defaults › Model › Choose…: the ModelSheet as a field picker."""
        try:
            from glossarion_mobile.ui.sheets.model_sheet import ModelSheet, sheet_env
        except Exception:
            self.say("The model picker is not available")
            return None
        if scope != "chat":  # "All chats": the global model, like the chat's sheet
            current = str(config.get("model", "") or "")
            on_select = lambda _f, value, _c: config.set_many({"model": value})  # noqa: E731
        else:
            current = str(self.store.defaults(sid).get("model") or config.get("model", "") or "")
            on_select = lambda _f, value, _c: self.store.set_default(sid, "model", value)  # noqa: E731
        sheet = ModelSheet(current_model=current, env=sheet_env(), field_mode=True, on_select=on_select,
                           title="Series model" if scope == "chat" else "Model")
        return self.show(sheet)

    # ---- glossary -----------------------------------------------------------------------------

    def _glossary(self) -> Any:
        return getattr(self.app, "glossary", None)

    def can_open_glossary(self) -> bool:
        return self._glossary() is not None

    def open_glossary(self, path: str) -> Any:
        feature = self._glossary()
        if feature is None or not path:
            self.say("The Glossary Manager is not available")
            return None
        return feature.open_editor_for_path(path)

    async def count_terms(self, path: str) -> Optional[int]:
        feature = self._glossary()
        service = getattr(feature, "service", None) if feature is not None else None
        counter = getattr(service, "count_entries", None)
        if not callable(counter) or not os.path.isfile(path):
            return None
        try:
            return await self.run_io(counter, path)
        except Exception:
            return None

    async def use_book_glossary(self, sid: str) -> Optional[str]:
        """"Use book glossary": the first linked book's glossary becomes the series' manual glossary."""
        item = self.store.get(sid)
        feature = self._glossary()
        service = getattr(feature, "service", None) if feature is not None else None
        finder = getattr(service, "glossaries_for_book", None)
        if item is None or not callable(finder):
            self.say("Book glossaries need the Glossary Manager")
            return None
        for bid in item.book_ids:
            book = self.book(bid)
            if book is None:
                continue
            try:
                rows = await self.run_io(finder, book)
            except Exception:
                rows = []
            if rows:
                path = str(rows[0].path)
                self.store.set_default(sid, "glossary_override_mode", "manual")
                self.store.set_default(sid, "manual_glossary_path", path)
                self.say(f"{item.name} uses {os.path.basename(path)} (Force Manual Glossary)")
                return path
        self.say("No linked book has a glossary file yet — extract one first")
        return None

    def clear_glossary(self, sid: str) -> None:
        item = self.store.get(sid)
        if item is None:
            return
        self.store.set_default(sid, "manual_glossary_path", None)
        if item.defaults.get("glossary_override_mode") == "manual":
            self.store.set_default(sid, "glossary_override_mode", None)

    # ---- Library ------------------------------------------------------------------------------

    def _library(self) -> Any:
        return getattr(self.app, "library", None)

    def book(self, bid: str) -> Optional[dict]:
        service = self._library()
        finder = getattr(service, "book_for_bid", None) if service is not None else None
        if not callable(finder):
            return None
        try:
            return finder(bid)
        except Exception:
            return None

    def book_title(self, bid: str) -> str:
        book = self.book(bid)
        return str(book.get("name") or bid) if book else f"Book {bid}"

    def cover_src(self, bid: str) -> Optional[str]:
        service = self._library()
        book = self.book(bid)
        if service is None or book is None:
            return None
        try:
            from glossarion_mobile.services.library import book_key

            return (getattr(service, "covers", {}) or {}).get(book_key(book))
        except Exception:
            return None

    def book_row(self, bid: str, on_more: Optional[Callable[[], Any]] = None) -> Any:
        """A linked book as a Library list row with its progress (None when it left the Library)."""
        service = self._library()
        book = self.book(bid)
        if service is None or book is None:
            return None
        try:
            from glossarion_mobile.services.library import book_key
            from glossarion_mobile.ui.library.book_card import BookListRow
            from glossarion_mobile.ui.library.models import build_card

            key = book_key(book)
            snapshot = getattr(service, "snapshot", None)
            views = getattr(snapshot, "views", {}) or {}
            badge, size = service.card_badge(book)
            model = build_card(book, key=key, bid=bid, view=views.get(key), badge_text=badge, size_label=size,
                               dark=self._dark())
            row = BookListRow(model, cover_src=(getattr(service, "covers", {}) or {}).get(key), dark=self._dark(),
                              on_open=lambda _m: self.go("library.book", {"bid": bid}),
                              on_long_press=lambda _m: on_more() if on_more is not None else None,
                              on_more=lambda _m: on_more() if on_more is not None else None)
            return row.control
        except Exception:
            log.exception("building the series book row failed")
            return None

    def _dark(self) -> bool:
        try:
            from glossarion_mobile.ui.theme import is_dark

            return bool(is_dark(self.page))
        except Exception:
            return False

    def series_for_book(self, bid: str) -> list:
        return self.store.series_for_book(bid)

    def add_books_sheet(self, bids: Sequence[str], title: Optional[str] = None) -> Any:
        """Library "Add to Series" (Book page ⋯, selection › More): link books to a series."""
        from glossarion_mobile.ui.chat.series_sheets import SeriesPickerSheet

        bids = [str(b) for b in bids if b]
        if not bids:
            self.say("No book to add")
            return None

        def pick(sid: Optional[str]) -> None:
            if sid:
                self._link(sid, bids)

        def new() -> None:
            self.edit_series(None, book_ids=bids, on_created=lambda item: self._linked_message(item, len(bids)))

        current = None
        if len(bids) == 1:
            owners = self.store.series_for_book(bids[0])
            current = owners[0].id if owners else None
        count = len(bids)
        subtitle = title or (f"{count} books" if count != 1 else self.book_title(bids[0]))
        self.sheet = SeriesPickerSheet(series=self.store.all(), current=current, title="Add to Series",
                                       subtitle=subtitle, on_pick=pick, on_new=new, tablet=self.tablet)
        return self.show(self.sheet)

    def _link(self, sid: str, bids: Sequence[str]) -> int:
        added = self.store.link_books(sid, bids)
        item = self.store.get(sid)
        if item is not None:
            self._linked_message(item, added, already=len(bids) - added)
        return added

    def _linked_message(self, item: Any, added: int, already: int = 0) -> None:
        if added:
            self.say(f"Added {added} book{'s' if added != 1 else ''} to {item.name}")
        elif already:
            self.say(f"Already in {item.name}")

    def book_ids(self, sid: Optional[str]) -> Optional[frozenset]:
        """The Library Series filter: the book ids of a series (None = no filter)."""
        if not sid:
            return None
        item = self.store.get(sid)
        return frozenset(item.book_ids) if item is not None else frozenset()

    def choices(self) -> list:
        """(sid, name, colour hex) for the Library filter chips."""
        return [(s.id, s.name, s.color_hex) for s in self.store.all()]

    # ---- screens ------------------------------------------------------------------------------

    def make_screen(self, match: RouteMatch) -> Any:
        from glossarion_mobile.ui.chat.series_page import SeriesScreen

        return SeriesScreen(match, self)

    def screen_factory(self, match: RouteMatch) -> Any:
        screen = None
        if match.name in SCREEN_ROUTES:
            try:
                screen = self.make_screen(match)
            except Exception:
                log.exception("building the %s screen failed", match.name)
                screen = None
            if screen is not None:
                self.screens_built.append(match.name)
        if screen is None:
            if self._fallback_factory is None:
                raise LookupError(f"no screen for {match.name}")
            screen = self._fallback_factory(match)
        return screen
