"""Phone-size regressions for the owner's reports "chat settings doesn't scroll down" and "the add key
button works, but there is no feedback and the menu doesn't close, so I end up adding the same key 20
times", and for the same bug classes in other sheets and dialogs.

Run from src/mobile (Flet venv or the 3.13 review venv):
    python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic tests_host/test_phone_sheets.py

* Scrolling. Flet 1.0.3 ``BottomSheet(scrollable=True)`` only lifts Flutter's 9/16 height cap; content
  taller than the sheet scrolls only when the sheet body has a vertical scroller (``Column(scroll=…)`` /
  ``ListView``), or is a Column whose expanded child is one (an action row pinned below a scrolling
  area). The tests walk the real control trees: every row the owner needs (the last one included) must
  sit inside such a scroller or in the pinned row under it, at 360x740 and 412x915, with the app text
  scale at 1.3 and 2.0. Short menus may stay compact only when they surely fit at 200 % text.
* Closing. ``page.pop_dialog()`` closes the most recently opened dialog, which after a save handler's
  snackbar is the snackbar, not the sheet. The tests use the real ``show_snackbar`` as ``notify`` (the
  older host tests used a list appender, which hid the bug) and check what is open afterwards.
"""

from __future__ import annotations

import asyncio
import importlib.util
import types
from pathlib import Path

import pytest

MOBILE_DIR = Path(__file__).resolve().parents[1]

_MK_SPEC = importlib.util.spec_from_file_location("_glossarion_mk_helpers_phone", Path(__file__).with_name("test_models_keys.py"))
MK = importlib.util.module_from_spec(_MK_SPEC)
_MK_SPEC.loader.exec_module(MK)
_fake_session = MK._fake_session
needs_flet = MK.needs_flet

_TC_SPEC = importlib.util.spec_from_file_location("_glossarion_tc_helpers_phone", Path(__file__).with_name("test_chat.py"))
_TC = importlib.util.module_from_spec(_TC_SPEC)
_TC_SPEC.loader.exec_module(_TC)
app_env = _TC.app_env
storage = _TC.storage
desktop_store_cls = _TC.desktop_store_cls
#: the chat sheets run on the shared Direct Text code (the flet-only venv lacks bs4 & co.)
needs_backend = pytest.mark.skipif(bool(_TC._BACKEND_ERROR), reason=_TC._BACKEND_ERROR or "backend importable")

PHONES = [(360, 740), (412, 915)]
SCALES = [1.3, 2.0]


# ==========================================================================
# control-tree helpers
# ==========================================================================


def _children(control) -> list:
    from flet.controls.base_control import BaseControl

    out = []
    for name in ("content", "title", "subtitle", "leading", "trailing"):
        child = getattr(control, name, None)
        if isinstance(child, BaseControl):
            out.append(child)
    for name in ("controls", "actions"):
        items = getattr(control, name, None)
        if isinstance(items, list):
            out.extend(c for c in items if isinstance(c, BaseControl))
    return out


def _path(root, target, trail=()):
    trail = trail + (root,)
    if root is target:
        return trail
    for child in _children(root):
        found = _path(child, target, trail)
        if found:
            return found
    return None


def _is_scroller(control) -> bool:
    name = type(control).__name__
    if name in ("ListView", "GridView", "ReorderableListView"):
        return True
    return name == "Column" and getattr(control, "scroll", None) is not None


def _pinned_layout(control) -> bool:
    """A Column (not scrolling) whose expanded child scrolls: the rows after it stay in view."""
    if type(control).__name__ != "Column" or getattr(control, "scroll", None) is not None:
        return False
    return any(getattr(c, "expand", None) and _is_scroller(c) for c in control.controls or [])


def reachable(root, target) -> bool:
    """``target`` can be scrolled into view (or is pinned under a scroller) inside ``root``."""
    path = _path(root, target)
    assert path, f"{type(target).__name__} is not in the sheet"
    return any(_is_scroller(c) or _pinned_layout(c) for c in path[:-1])


def assert_phone_sheet(sheet, *targets) -> None:
    """Sized to the viewport (no 9/16 cap), every target reachable, inset above the navigation bar."""
    assert sheet.scrollable, "the sheet keeps Flutter's 9/16 height cap"
    for target in targets:
        assert reachable(sheet.content, target), f"{type(target).__name__} cannot be scrolled into view"
    assert type(sheet.content).__name__ == "SafeArea" and sheet.content.avoid_intrusions_bottom, \
        "the last row would sit under the navigation bar (no bottom inset)"


def _column_of(root, target):
    """The Column that holds ``target`` (its nearest Column ancestor)."""
    path = _path(root, target)
    assert path, f"{type(target).__name__} is not in the sheet"
    return next(c for c in reversed(path[:-1]) if type(c).__name__ == "Column")


def _tile(sheet, title):
    """A chat-settings section (ExpansionTile) by its title."""
    return next(c for c in sheet.body.controls if type(c).__name__ == "ExpansionTile" and c.title == title)


def _open_dialogs(page) -> list:
    return [d for d in page._dialogs.controls if getattr(d, "open", False)]


def _tap(button):
    """What a tap does: the button's on_click with an event (sync or coroutine handler)."""
    return button.on_click(types.SimpleNamespace(control=button))


def _phone(width, height):
    conn, session = _fake_session("android")
    session.apply_page_patch({"width": width, "height": height})
    return conn, session, session.page


def _snackbar_notify(page, bars):
    """``GlossarionApp.notify``: a real SnackBar through ``page.show_dialog``."""
    from glossarion_mobile.ui.components.dialogs import show_snackbar

    def notify(message, action_label=None, on_action=None):
        bar = show_snackbar(page, message, action_label=action_label, on_action=on_action)
        bars.append(bar)
        return bar

    return notify


def _text(bar) -> str:
    return str(getattr(bar.content, "value", ""))


# ==========================================================================
# owner report 1: chat settings
# ==========================================================================


_LIVE_SESSION: list = []  # Flet's Page holds its session weakly: keep the fake one alive


async def _start_phone(tf, width, height):
    main_module = tf._load_main_module()
    conn, session = tf._fake_session("android")
    # without a strong reference the next cyclic GC destroys the session mid-test ("An attempt to
    # fetch destroyed session"); when that happens depends on how many objects the app allocates
    _LIVE_SESSION[:] = [(conn, session)]
    session.apply_page_patch({"width": width, "height": height})
    page = session.page
    await main_module.main(page)
    await session.after_event(page)
    return page, page.data


def _find(root, predicate):
    if predicate(root):
        return root
    for child in _children(root):
        found = _find(child, predicate)
        if found is not None:
            return found
    return None


@needs_flet
@needs_backend
@pytest.mark.parametrize("scale", SCALES)
@pytest.mark.parametrize("size", PHONES)
def test_chat_settings_sheet_scrolls_to_its_last_row_on_a_phone(app_env, desktop_store_cls, tmp_path, size, scale):
    """Chat › header ⋮ › Chat settings, and ＋ › This chat › Chat settings…: every section and the
    "Reset chat overrides" footer can be scrolled to."""
    from glossarion_mobile.ui.chat.integration import ChatFeature

    tf = _TC._load_foundations()
    _TC._desktop_history(tmp_path)

    async def scenario():
        page, app = await _start_phone(tf, *size)
        try:
            await tf._wait(lambda: app.state.engine_ready)
            app.state.text_scale.set(scale)
            adapter = _TC._adapter(desktop_store_cls, tmp_path, save_delay=0.01)
            await ChatFeature.install(app, chats=adapter, jobs=_TC.FakeJobService(), oauth=_TC._FakeOAuth())
            view = app.chat_view
            view.header._menu("chat_settings")  # the header ⋮ row
            sheet = view.settings_sheet
            assert sheet is not None and sheet.dialog in _open_dialogs(page)
            reset = _find(sheet.dialog.content, lambda c: getattr(c, "content", None) == "Reset chat overrides")
            text_size = _find(sheet.dialog.content, lambda c: type(c).__name__ == "Slider" and getattr(c, "max", None) == 1.5)
            assert reset is not None and text_size is not None
            assert_phone_sheet(sheet.dialog, _tile(sheet, "Run behaviour"), _tile(sheet, "Conversation"), text_size, reset)
            sheet.close()
            assert not sheet.dialog.open
            # ＋ › This chat › Chat settings… (the ＋ sheet's last row) opens the same sheet
            from glossarion_mobile.ui.sheets.plus_sheet import PlusSheet

            plus = PlusSheet()
            assert_phone_sheet(plus.dialog, plus.chat_tiles["chat_settings"], plus.tool_tiles["retranslate"])
            view._on_this_chat("chat_settings")
            assert view.settings_sheet is not sheet and view.settings_sheet.dialog in _open_dialogs(page)
        finally:
            await tf._stop(app)

    asyncio.run(scenario())


class _Chats:
    def __init__(self):
        self.values, self.metas = {}, {}

    def overrides(self, cid):
        return dict(self.values)

    def meta(self, cid):
        return dict(self.metas)

    def set_override(self, cid, name, value):
        if value is None:
            self.values.pop(name, None)
        else:
            self.values[name] = value

    def set_meta(self, cid, name, value):
        self.metas[name] = value

    def reset_overrides(self, cid):
        self.values.clear()


class _Config(dict):
    def set_many(self, updates):
        self.update(updates)


@needs_flet
@needs_backend
def test_chat_settings_sections_keep_their_expanded_state_after_a_change():
    """Toggling a switch inside "Run behaviour" rebuilt the rows with the defaults, so the section the
    user was in collapsed under their finger."""
    from glossarion_mobile.ui.sheets.chat_settings import ChatSettingsSheet

    sheet = ChatSettingsSheet(cid="2", config=_Config(), chats=_Chats())
    assert _tile(sheet, "Run behaviour").expanded is False and _tile(sheet, "Model & prompt").expanded is True
    _tile(sheet, "Run behaviour").expanded = True  # the client reports the tap (ExpansionTile on_change)
    _tile(sheet, "Model & prompt").expanded = False
    sheet.set_value("force_multipass_off", True)
    assert _tile(sheet, "Run behaviour").expanded is True and _tile(sheet, "Model & prompt").expanded is False
    assert sheet.value_of("force_multipass_off") is True


class _Prefs:
    def __init__(self):
        self.data = {}

    def get(self, key, default=None):
        return self.data.get(key, default)

    def set(self, key, value):
        self.data[key] = value


def _auto_accept_row(sheet):
    return next((c for c in _tile(sheet, "Glossary").controls if getattr(c, "key", None) == "setting-auto_accept_glossary"),
                None)


def _flip(row, value):
    """A tap on the row's Switch (its on_change with the new value)."""
    switch = _find(row, lambda c: type(c).__name__ == "Switch")
    switch.on_change(types.SimpleNamespace(control=types.SimpleNamespace(value=value)))


@needs_flet
@needs_backend
def test_chat_settings_auto_accept_switch_writes_prefs_and_chat_meta():
    """Owner report #6: "a toggle to always auto-accept the generated glossary". Chat settings › Glossary ›
    "Always accept generated glossaries": All chats writes Prefs ``chat_auto_accept_glossary`` (never
    config.json), This chat the chat's sidecar meta; ↺ / Reset chat overrides clear the chat's value."""
    from glossarion_mobile.ui.chat.direct_text_rules import AUTO_ACCEPT_GLOSSARY_PREF
    from glossarion_mobile.ui.sheets.chat_settings import DESCRIPTIONS, ChatSettingsSheet

    prefs, config, chats = _Prefs(), _Config(), _Chats()
    sheet = ChatSettingsSheet(cid="2", config=config, chats=chats, prefs=prefs)
    row = _auto_accept_row(sheet)
    assert row is not None and sheet.value_of("auto_accept_glossary") is False  # default off (desktop: always asks)
    switch = _find(row, lambda c: type(c).__name__ == "Switch")
    assert switch.label == "Always accept generated glossaries" and switch.value is False
    assert _find(row, lambda c: getattr(c, "value", None) == DESCRIPTIONS["auto_accept_glossary"]) is not None
    assert "Desktop always asks" in DESCRIPTIONS["auto_accept_glossary"]

    # This chat: the sidecar meta, nothing global
    _flip(row, True)
    assert chats.metas == {"auto_accept_glossary": True} and chats.values == {}
    assert prefs.data == {} and dict(config) == {}
    assert sheet.value_of("auto_accept_glossary") is True and sheet.is_overridden("auto_accept_glossary")
    assert _find(_auto_accept_row(sheet), lambda c: getattr(c, "value", None) == "custom") is not None
    sheet.reset("auto_accept_glossary")  # ↺
    assert chats.metas.get("auto_accept_glossary") is None and sheet.value_of("auto_accept_glossary") is False

    # All chats: Prefs only
    sheet.scope = "global"
    sheet.rebuild()
    _flip(_auto_accept_row(sheet), True)
    assert prefs.data == {AUTO_ACCEPT_GLOSSARY_PREF: True} and dict(config) == {}
    assert chats.metas.get("auto_accept_glossary") is None
    sheet.scope = "chat"
    sheet.rebuild()
    assert sheet.value_of("auto_accept_glossary") is True and not sheet.is_overridden("auto_accept_glossary")
    assert _find(_auto_accept_row(sheet), lambda c: str(getattr(c, "value", "")).startswith("Inherited from")) is not None
    _flip(_auto_accept_row(sheet), False)  # this chat asks again
    assert chats.metas["auto_accept_glossary"] is False and sheet.effective().auto_accept_glossary is False
    sheet.reset()  # Reset chat overrides
    assert chats.metas.get("auto_accept_glossary") is None and sheet.effective().auto_accept_glossary is True

    # Series defaults (built without Prefs) and a sheet without Prefs: no row
    series = ChatSettingsSheet(cid="s1", config=_Config(), chats=_Chats(), prefs=prefs, subject="series")
    assert _auto_accept_row(series) is None
    assert _auto_accept_row(ChatSettingsSheet(cid="2", config=_Config(), chats=_Chats())) is None


# ==========================================================================
# the same scroll bug in other sheets
# ==========================================================================


@needs_flet
@pytest.mark.parametrize("size", PHONES)
def test_long_action_sheet_scrolls_and_a_short_one_stays_compact(size):
    from glossarion_mobile.ui.components.action_sheet import ActionItem, ActionSheet

    async def scenario():
        _conn, _session, page = _phone(*size)
        rows = [ActionItem(f"Chapter action {i}", lambda: None, icon="EDIT",
                           disabled_reason="No audio file" if i in (5, 6, 7, 8) else None) for i in range(14)]
        chapter_menu = ActionSheet(rows, title="Chapter 12", subtitle="✅ Completed")  # library chapters_tab
        chapter_menu.show(page)
        assert_phone_sheet(chapter_menu.dialog, chapter_menu.tiles[-1], chapter_menu.cancel_tile)
        short = ActionSheet([ActionItem("Rename", lambda: None), ActionItem("Delete", lambda: None, destructive=True)],
                            title="My chat")
        short.show(page)
        # surely fits at 200 % text: no full-height sheet for two rows
        assert _column_of(short.dialog.content, short.cancel_tile).scroll is None
        tablet = ActionSheet(rows, title="Chapter 12", tablet=True)
        tablet.show(page)
        assert reachable(tablet.dialog.content, tablet.cancel_tile)

    asyncio.run(scenario())


@needs_flet
@pytest.mark.parametrize("size", PHONES)
def test_show_full_translation_info_sheet_scrolls(size):
    from glossarion_mobile.ui.chat.direct_text_rules import LONG_OUTPUT_CHARS
    from glossarion_mobile.ui.components.info_sheet import InfoSheet

    async def scenario():
        _conn, _session, page = _phone(*size)
        body = ("A translated sentence of a long chapter. " * 400)[:LONG_OUTPUT_CHARS + 1]
        sheet = InfoSheet(title="Request 1", body=body)
        sheet.show(page)
        text = _find(sheet.dialog.content, lambda c: getattr(c, "value", None) == body)
        assert_phone_sheet(sheet.dialog, text)
        note = InfoSheet(title="Why", body="Needs the TTS SDK.")
        note.show(page)
        assert _column_of(note.dialog.content, _find(note.dialog.content, lambda c: getattr(c, "value", None) == "Why")
                          ).scroll is None

    asyncio.run(scenario())


@needs_flet
@pytest.mark.parametrize("size", PHONES)
def test_keys_request_contexts_sheet_keeps_apply_in_reach(size):
    from key_contexts import CONTEXT_LABELS

    from glossarion_mobile.ui.screens.keys import ContextSheet

    async def scenario():
        _conn, _session, page = _phone(*size)
        applied = []
        sheet = ContextSheet(states={k: True for k in CONTEXT_LABELS}, labels=CONTEXT_LABELS, on_apply=applied.append)
        sheet.show(page)
        apply = _find(sheet.dialog.content, lambda c: getattr(c, "content", None) == "Apply")
        assert_phone_sheet(sheet.dialog, list(sheet.chips.values())[-1], apply)
        _tap(apply)
        _tap(apply)  # a second tap while it closes
        assert applied == [{}] and not sheet.dialog.open

    asyncio.run(scenario())


@needs_flet
@needs_backend
def test_mode_options_model_poe_manual_glossary_and_picker_sheets_reach_their_last_row():
    from glossarion_mobile.ui.chat.mode_options_sheet import ModeOptionsSheet
    from glossarion_mobile.ui.sheets.manual_glossary import ManualGlossarySheet
    from glossarion_mobile.ui.sheets.model_sheet import ModelSheet, PoeSetupSheet, SheetEnv
    from glossarion_mobile.ui.tools.source_picker import SourcePicker

    vision = ModeOptionsSheet("vision")
    assert_phone_sheet(vision.dialog, vision.content.column)
    assert vision.content.column.scroll is None  # the ＋ sheet inlines the same column inside its own scroll
    model = ModelSheet(current_model="gpt-6", env=SheetEnv())
    assert_phone_sheet(model.dialog, model.list_view, model.thinking, model.footer)
    poe = PoeSetupSheet(env=SheetEnv())
    assert_phone_sheet(poe.dialog, poe.field)
    manual = ManualGlossarySheet()
    assert_phone_sheet(manual.dialog, manual.editor, manual.use_button)
    picker = SourcePicker(types.SimpleNamespace(spawn=lambda c: None, push=lambda *a: None), title="QA", multi=True)
    assert picker.list_view.expand and picker.list_view.height is None
    assert_phone_sheet(picker.sheet, picker.list_view, picker.done_button)


@needs_flet
@pytest.mark.parametrize("size", PHONES)
def test_review_file_order_and_display_sheets_scroll(size):
    from glossarion_mobile.ui.tools.review import ReviewScreen

    async def scenario():
        _conn, _session, page = _phone(*size)
        fake = types.SimpleNamespace(
            ctx=types.SimpleNamespace(page=page, cfg=lambda k, d=None: d), _on_reorder=lambda e: None,
            set_display=lambda *a: None, reset_display=lambda *a: None,
            targets=[types.SimpleNamespace(source_name=f"Volume {i + 1}.epub", title=f"Volume {i + 1}") for i in range(12)])
        order = ReviewScreen.open_order_sheet(fake)
        assert fake.order_listing.expand  # bounded: the list scrolls itself (and auto-scrolls while dragging)
        assert_phone_sheet(order, fake.order_listing.controls[-1])
        fake.targets = fake.targets[:2]
        assert not ReviewScreen.open_order_sheet(fake).content.content.content.controls[-1].expand  # two rows fit
        display = ReviewScreen.open_display_sheet(fake)
        reset = _find(display.content, lambda c: getattr(c, "key", None) == "review-display-reset")
        assert_phone_sheet(display, fake.display_fields["review_list_gap"], reset)

    asyncio.run(scenario())


@needs_flet
def test_refusal_patterns_header_scrolls_with_the_list(tmp_path):
    from glossarion_mobile.ui.screens.refusal_patterns import RefusalModel, RefusalPatternsScreen

    async def scenario():
        _conn, _session, page = _phone(360, 740)
        store = MK._store(tmp_path, {})
        screen = RefusalPatternsScreen(None, model=RefusalModel(store), page=page, notify=lambda *a: None)
        body = screen.get_body()
        page.views[0].controls.append(body)
        page.update()
        # with the keyboard up the page body is ~380 dp: a fixed header left the list 0 dp
        assert reachable(body, screen.search) and reachable(body, screen.new_field)
        assert _path(screen.list_view, screen.search) and _path(screen.list_view, screen.delete_button)
        screen.set_query("as an ai")  # the filtered rows follow the header in the same list
        assert _path(screen.list_view.controls[0], screen.search)
        assert len(screen.list_view.controls) == 1 + len(screen.visible_patterns()) < 1 + len(screen.model.patterns())
        store._saver.close()

    asyncio.run(scenario())


# ==========================================================================
# owner report 2: Add key
# ==========================================================================


def _keys_screen(tmp_path, page, notify, data=None):
    import key_pool_service as real

    from glossarion_mobile.ui.router import parse_route
    from glossarion_mobile.ui.screens.keys import KeysScreen

    module = MK._fake_key_service(find_duplicate_key=real.find_duplicate_key,
                                  added_key_extra_info=real.added_key_extra_info)
    store, keys = MK._keys(tmp_path, {"use_multi_api_keys": True, "multi_api_keys": [], **(data or {})}, module=module)
    ctx = MK._settings_ctx(store, page)
    ctx.notify = notify
    screen = KeysScreen(parse_route("/settings/keys/translation"), controller=keys, ctx=ctx, page=page, notify=notify,
                        run_io=lambda fn, *a: asyncio.to_thread(fn, *a), spawn=lambda c: asyncio.ensure_future(c))
    page.views[0].controls.append(screen.get_body())
    page.update()
    screen.did_show()
    return store, keys, screen


@needs_flet
@pytest.mark.parametrize("size", PHONES)
def test_add_key_closes_confirms_once_and_refuses_a_duplicate(tmp_path, size):
    async def scenario():
        _conn, _session, page = _phone(*size)
        bars = []
        store, keys, screen = _keys_screen(tmp_path, page, _snackbar_notify(page, bars))
        editor = screen.open_editor(None)
        assert editor.sheet in _open_dialogs(page) and editor.save_button.content == "Add"
        editor.key_field.set_value("sk-dup-key-1234567890", notify=False)
        editor.model_picker.set_value("gpt-6")
        _tap(editor.save_button)
        # one key, the sheet closed, the confirmation names the key and the pool and stays on screen
        assert keys.count("main") == 1
        assert not editor.sheet.open and editor.closed
        assert [_text(b) for b in bars] == ["Added key sk-…7890 · gpt-6 to Translation Keys"]
        assert _open_dialogs(page) == [bars[0]]
        assert editor.save_button.disabled and editor.close_button.disabled
        for _ in range(19):  # the owner's repeated taps while the sheet animates away
            _tap(editor.save_button)
        assert keys.count("main") == 1 and len(bars) == 1
        # the same key again: refused with a message, the editor stays open and usable
        again = screen.open_editor(None)
        again.key_field.set_value("sk-dup-key-1234567890", notify=False)
        again.model_picker.set_value("gpt-6")
        _tap(again.save_button)
        assert keys.count("main") == 1 and again.sheet.open and not again.closed
        assert again.error_text.visible and "already in Translation Keys (#1: sk-…7890 · gpt-6)" in again.error_text.value
        assert not again.save_button.disabled and bars[0].open
        # another model is another key
        again.model_picker.set_value("gpt-6-mini")
        _tap(again.save_button)
        assert keys.count("main") == 2 and not again.sheet.open
        # Edit key saves and says so after closing
        edit = screen.open_editor(0)
        _tap(edit.save_button)
        assert not edit.sheet.open and _text(bars[-1]) == "Saved key sk-…7890 · gpt-6" and bars[-1].open
        screen.dispose()
        store._saver.close()

    asyncio.run(scenario())


@needs_flet
def test_editor_save_is_disabled_while_an_async_save_runs(tmp_path):
    from glossarion_mobile.ui.settings.editors import SecretEditor

    async def scenario():
        _conn, _session, page = _phone(360, 740)
        store = MK._store(tmp_path, {})
        gate = asyncio.Event()
        calls, saved = [], []

        async def on_save(value):
            calls.append(value)
            await gate.wait()
            return None

        editor = SecretEditor(MK._settings_ctx(store, page), title="Key", value="", on_save=on_save,
                              ).show()
        editor.on_saved = saved.append
        editor.field.value = "secret"
        task = editor.save()
        await asyncio.sleep(0)
        assert editor.save_button.disabled and editor.close_button.disabled
        assert editor.save() is False and calls == ["secret"]  # a tap while saving is ignored
        gate.set()
        assert await task is True
        assert not editor.sheet.open and saved == ["secret"] and calls == ["secret"]
        # an error keeps the editor open and usable
        failing = SecretEditor(MK._settings_ctx(store, page), title="Key", value="", on_save=lambda v: "Invalid").show()
        assert failing.save() is False and failing.sheet.open and not failing.save_button.disabled
        assert failing.error_text.value == "Invalid"
        store._saver.close()

    asyncio.run(scenario())


# ==========================================================================
# confirm dialogs and sheets that show feedback while closing
# ==========================================================================


@needs_flet
def test_confirm_dialog_closes_itself_and_keeps_the_actions_snackbar(tmp_path):
    from glossarion_mobile.ui.components.dialogs import ConfirmDialog, show_snackbar

    async def scenario():
        _conn, _session, page = _phone(360, 740)
        runs, bars = [], []

        def action():
            runs.append(1)
            bars.append(show_snackbar(page, "Removed 2 key(s)", action_label="Undo", on_action=lambda: None))

        dialog = ConfirmDialog(title="Clear all keys", body="Remove all?", destructive=True, on_confirm=action)
        dialog.show(page)
        await dialog._on_confirm()
        assert runs == [1] and not dialog.dialog.open and bars[0].open and dialog.state == "closed"
        await dialog._on_confirm()  # a second tap while it closes never runs the action again
        assert runs == [1]

    asyncio.run(scenario())


@needs_flet
def test_clear_all_keys_and_legacy_import_close_and_keep_undo(tmp_path):
    async def scenario():
        _conn, _session, page = _phone(412, 915)
        bars = []
        store, keys, screen = _keys_screen(tmp_path, page, _snackbar_notify(page, bars), data={"multi_api_keys": [
            {"api_key": "sk-aaaaaaaaaaaa1111", "model": "gpt-6"}, {"api_key": "sk-bbbbbbbbbbbb2222", "model": "gpt-6"}]})
        dialog = screen.confirm_clear()
        await dialog._on_confirm()
        assert keys.count("main") == 0 and not dialog.dialog.open
        assert _text(bars[-1]) == "Removed 2 key(s)" and bars[-1].open
        await dialog._on_confirm()
        assert _text(bars[-1]) == "Removed 2 key(s)"  # no "Removed 0 key(s)" over the Undo
        bars[-1].on_action(None)  # Undo
        await asyncio.sleep(0)
        assert keys.count("main") == 2
        # a legacy list appends, but never a key the pool already holds
        plan = keys.import_plan([{"api_key": "sk-aaaaaaaaaaaa1111", "model": "gpt-6"},
                                 {"api_key": "sk-cccccccccccc3333", "model": "gpt-6"}])
        assert keys.apply_import(plan) == 1 and keys.count("main") == 3 and plan.duplicates == 1
        assert "1 duplicate key(s) skipped" in keys.import_result_message(plan, 1)
        screen.dispose()
        store._saver.close()

    asyncio.run(scenario())


@needs_flet
@needs_backend
def test_model_sheet_closes_before_its_choice_opens_the_next_sheet():
    """Long-press Send › "Translate once with another model…": with Force Manual Glossary the choice
    opens the Manual glossary sheet, which the model sheet's close used to pop instead of itself."""
    from glossarion_mobile.ui.sheets.manual_glossary import ManualGlossarySheet
    from glossarion_mobile.ui.sheets.model_sheet import ModelSheet, SheetEnv

    async def scenario():
        _conn, _session, page = _phone(360, 740)
        opened = []

        def on_select(field, value, chat_scope):
            manual = ManualGlossarySheet()
            manual.show(page)
            opened.append(manual)

        sheet = ModelSheet(current_model="gpt-6", env=SheetEnv(), one_shot=True, on_select=on_select)
        sheet.show(page)
        sheet.select("model", "gpt-6-mini")
        assert not sheet.dialog.open and opened and opened[0].dialog.open

    asyncio.run(scenario())


@needs_flet
def test_qa_custom_sheet_save_closes_and_keeps_its_snackbar(tmp_path):
    from glossarion_mobile.ui.tools.qa_custom import CustomModeSheet

    async def scenario():
        _conn, _session, page = _phone(360, 740)
        bars, cfg = [], {}
        notify = _snackbar_notify(page, bars)
        ctx = types.SimpleNamespace(page=page, cfg=lambda k, d=None: cfg.get(k, d), set_cfg=cfg.__setitem__,
                                    say=lambda m, *a: notify(m), push=lambda *c: None)
        sheet = CustomModeSheet(ctx).show(page)
        sheet.save()
        assert not sheet.sheet.open and bars and bars[0].open

    asyncio.run(scenario())
