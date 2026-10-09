"""Device fixes, owner requests #15 / #16 (+ the Chat-settings half of #17): prompts in Chat settings.

What the owner reported on the U8 APK (2026-10-08):

* #16 "why is the only prompt the universal prompt. the rest are all missing from the mobile version":
  the chat's profile pickers (Chat settings, the ModelSheet Profile tab, ``/profile``, Series defaults)
  came from ``integration.profile_names``, which built the built-ins on a bare SimpleNamespace, hit an
  AttributeError and fell back to ``["Universal"]`` on every fresh config. They now read the Settings ›
  Profiles & prompts listing (the shared ``prompt_profiles`` desktop start-up): all 18 built-ins,
  translation profiles first and the task-specific ones under "Specialised" (owner decision).
* #15 "we should be able to modify prompts in the chat settings and add prompts": Chat settings ›
  Model & prompt has the profile's prompt card: Edit prompt (``PromptEditorSheet``; edits the shared
  profile like the desktop), New profile… (a copy of the current profile) and Manage… (Settings ›
  Profiles & prompts). Edits from This chat never switch the global/desktop active profile
  (``ProfileService(keep_active=True)``); renamed profiles are followed by chats and series, deleted
  ones fall back to the inherited profile (never a silent Universal).
* #17: no API-key settings in Chat settings (keys get their own Keys button in the drawer footer).

Everything runs on the real shared ``prompt_profiles`` core and a real ``MobileConfigStore`` in a temp
folder; HOME / USERPROFILE / APPDATA / the Library, data and output folders point at scratch dirs.

Run from src/mobile with the mobile venv:
    python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic tests_host/test_chat_prompts.py
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

MOBILE_DIR = Path(__file__).resolve().parents[1]
APP_DIR = MOBILE_DIR / "app"
SRC_DIR = MOBILE_DIR.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))


def _has(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


needs_flet = pytest.mark.skipif(not (_has("flet") and _has("msgpack")), reason="flet/msgpack not installed")

_TC_SPEC = importlib.util.spec_from_file_location("_glossarion_tc_helpers_prompts", Path(__file__).with_name("test_chat.py"))
_TC = importlib.util.module_from_spec(_TC_SPEC)
_TC_SPEC.loader.exec_module(_TC)
desktop_store_cls = _TC.desktop_store_cls
#: the profile listing replays the desktop start-up (owner_state & co.); the flet-only venv lacks the backend
needs_backend = pytest.mark.skipif(bool(_TC._BACKEND_ERROR), reason=_TC._BACKEND_ERROR or "backend importable")
pytestmark = [needs_flet, needs_backend]

#: The desktop's built-in profiles (owner_state.ConfigStateMixin._init_default_prompt_profiles), desktop order.
BUILTINS = ["Universal", "Refinement", "Korean_BeautifulSoup", "Japanese_BeautifulSoup", "Chinese_BeautifulSoup",
            "Korean_html2text", "Japanese_html2text", "Chinese_html2text", "Manga_JP", "Manga_KR", "Manga_CN",
            "Glossary_Editor", "RPGMaker_GTool", "RPGMaker_GTool_Image", "NanoBanana_Image", "Original",
            "SDLXLIFF Editing v2", "Subtitle Translation"]
TRANSLATION = ["Universal", "Korean_BeautifulSoup", "Japanese_BeautifulSoup", "Chinese_BeautifulSoup",
               "Korean_html2text", "Japanese_html2text", "Chinese_html2text"]
SPECIALISED = [n for n in BUILTINS if n not in TRANSLATION]


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    """No test here reads or writes the user's home, app data, Library or output folders."""
    for name, sub in (("HOME", "home"), ("USERPROFILE", "home"), ("APPDATA", "appdata"),
                      ("GLOSSARION_LIBRARY_DIR", "Library"), ("GLOSSARION_DATA_DIR", "data")):
        folder = tmp_path / "_env" / sub
        folder.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv(name, str(folder))
    monkeypatch.setenv("OUTPUT_DIRECTORY", str(tmp_path / "_env" / "Output"))
    monkeypatch.setenv("GLOSSARION_HTTP_LOG", "0")
    from glossarion_mobile.ui.screens import profiles, prompt_editor

    # the editor's debounced token count needs a running UI loop; these tests drive it synchronously
    monkeypatch.setattr(prompt_editor.TokenCounter, "schedule", lambda self, text: None)

    profiles._RENAMED.clear()
    yield tmp_path
    profiles._RENAMED.clear()


def _store(tmp_path: Path, config=None, name: str = "config.json"):
    from glossarion_mobile.state.config_store import MobileConfigStore

    path = tmp_path / name
    path.write_text(json.dumps(config or {}), encoding="utf-8")
    store = MobileConfigStore(str(path), debounce=10,
                              reader=lambda p, decrypt=True: json.loads(Path(p).read_text(encoding="utf-8")),
                              writer=lambda disk, p, backup=False: Path(p).write_text(json.dumps(disk), encoding="utf-8"))
    store.load()
    return store


def _desktop_defaults() -> dict:
    import prompt_profiles

    return dict(prompt_profiles.profile_state_from_config({}).default_prompts)


class Chats:
    """The ChatStoreAdapter surface the sheet uses (one chat)."""

    def __init__(self, overrides=None):
        self.values = dict(overrides or {})
        self.metas: dict = {}

    def overrides(self, cid):
        return dict(self.values)

    own_overrides = overrides

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


class Page:
    def __init__(self):
        self.dialogs: list = []
        self.width, self.height = 412, 900

    def show_dialog(self, dialog):
        dialog.open = True
        self.dialogs.append(dialog)

    def update(self, *a):
        pass


class Ctx:
    """The SettingsContext surface the prompt editor uses."""

    def __init__(self, store=None):
        self.store = store
        self.page = Page()
        self.messages: list = []
        self.routes: list = []

    def show_dialog(self, dialog):
        self.page.show_dialog(dialog)

    def push(self, *controls):
        pass

    def say(self, message, *_a):
        self.messages.append(message)

    async def run_io(self, fn, *args):
        return fn(*args)

    def go(self, name, params=None, *, fragment=None):
        self.routes.append((name, params))
        return name


def _walk(control, out=None):
    from flet.controls.base_control import BaseControl

    out = [] if out is None else out
    if isinstance(control, (list, tuple)):
        for item in control:
            _walk(item, out)
        return out
    if not isinstance(control, BaseControl):
        return out
    out.append(control)
    for attr in ("controls", "content", "title", "subtitle", "leading", "trailing", "options", "segments",
                 "label", "actions"):
        value = getattr(control, attr, None)
        if isinstance(value, (list, tuple, BaseControl)):
            _walk(value, out)
    return out


def _texts(root) -> list:
    texts = []
    for control in _walk(root):
        for attr in ("value", "label", "text", "tooltip", "content", "hint_text"):
            value = getattr(control, attr, None)
            if isinstance(value, str) and value:
                texts.append(value)
    return texts


def _by_key(root, key):
    return next((c for c in _walk(root) if getattr(c, "key", None) == key), None)


def _profile_dropdown(sheet):
    import flet as ft

    row = _by_key(sheet.sections["model"], "setting-profile")
    return next(c for c in row.controls if isinstance(c, ft.Dropdown))


def _card(sheet):
    return _by_key(sheet.sections["model"], "setting-profile-prompt")


def _card_text(sheet, suffix):
    control = _by_key(_card(sheet), f"setting-profile-prompt-{suffix}")
    return None if control is None else getattr(control, "value", control)


# ==========================================================================
# #16: every built-in, in the chat's order
# ==========================================================================


def test_owner_report_16_a_fresh_config_lists_every_builtin_not_only_universal(tmp_path):
    """The owner's exact complaint: on a fresh mobile config the chat offered only "Universal"."""
    from glossarion_mobile.ui.chat.integration import ChatFeature, profile_names
    from glossarion_mobile.ui.sheets.chat_settings import GROUP_OPTION_PREFIX, ChatSettingsSheet

    store = _store(tmp_path)
    names = profile_names(store)
    assert names != ["Universal"] and len(names) == 18
    assert sorted(names) == sorted(_desktop_defaults()) == sorted(BUILTINS)
    # translation profiles first, then the task-specific built-ins (each group in desktop order)
    assert names == TRANSLATION + SPECIALISED
    assert store.keys() == [] and not store.dirty  # listing writes nothing (UI_SPEC Appendix B)

    # the chat env the app builds (ChatFeature.build_env) hands the same list to every picker
    app = types.SimpleNamespace(page=None, dispatcher=None, paths=None, config_store=store)
    feature = ChatFeature(app, chats=Chats(), jobs=None, oauth=types.SimpleNamespace())
    env = feature.build_env()
    assert env.profiles() == names and env.profile_service is not None

    # Chat settings › Prompt profile: 18 profiles + the disabled "Specialised" heading after the translation ones
    sheet = ChatSettingsSheet(cid="2", config=store, chats=Chats(), profiles=env.profiles())
    options = _profile_dropdown(sheet).options
    keys = [o.key for o in options]
    assert [k for k in keys if not k.startswith(GROUP_OPTION_PREFIX)] == names
    heading = options[len(TRANSLATION)]
    assert heading.key == GROUP_OPTION_PREFIX + "Specialised" and heading.disabled and heading.text == "Specialised"
    assert _profile_dropdown(sheet).value == "Universal"
    store._saver.close()


CONFIGS = {
    # a current desktop config (every built-in) + a custom profile
    "desktop": lambda d: ({**d, "My style": "mine"}, 19),
    # an older desktop config without the newest built-ins: desktop start-up adds them back
    "old_desktop": lambda d: ({k: v for k, v in d.items()
                               if k not in ("NanoBanana_Image", "SDLXLIFF Editing v2", "Subtitle Translation")}
                              | {"My style": "mine"}, 19),
    # only Universal stored: the always-include built-ins come back (the deletable ones do not)
    "only_universal": lambda d: ({"Universal": "x"}, 13),
    "empty": lambda d: ({}, 13),
}


@pytest.mark.parametrize("label", list(CONFIGS))
def test_chat_list_is_the_profiles_screen_listing(tmp_path, label):
    from glossarion_mobile.ui.chat.integration import profile_names
    from glossarion_mobile.ui.screens.profiles import ProfileService, chat_profile_order

    stored, count = CONFIGS[label](_desktop_defaults())
    store = _store(tmp_path, {"prompt_profiles": stored})
    listing = ProfileService(store).listing()
    names = profile_names(store)
    assert len(names) == count and sorted(names) == sorted(listing.names)
    assert names == chat_profile_order(listing.names)
    if "My style" in stored:  # a custom profile is a translation profile: before "Specialised"
        assert names.index("My style") < names.index("Refinement")
    assert set(SPECIALISED) & set(names) <= set(names[-len(set(SPECIALISED) & set(names)):])
    store._saver.close()


def test_profile_names_fallbacks(tmp_path, monkeypatch):
    from glossarion_mobile.ui.chat.integration import profile_names

    # a hand-edited prompt_profiles that is not an object: the desktop start-up rejects it -> no crash
    store = _store(tmp_path, {"prompt_profiles": ["Universal", "Mine"]})
    assert profile_names(store) == ["Universal"]
    store._saver.close()
    # a build whose desktop start-up cannot import: the stored profiles, in the chat order
    import prompt_profiles

    def broken(config):
        raise ImportError("no backend here")

    monkeypatch.setattr(prompt_profiles, "profile_state_from_config", broken)
    store = _store(tmp_path, {"prompt_profiles": {"Manga_JP": "m", "Mine": "x"}}, name="c2.json")
    assert profile_names(store) == ["Mine", "Manga_JP"]
    store._saver.close()


# ==========================================================================
# ProfileService(keep_active=True): edits that do not choose the profile
# ==========================================================================


def test_keep_active_edits_leave_the_global_profile_alone(tmp_path):
    from glossarion_mobile.ui.screens.profiles import PROFILE_CONFIG_KEYS, ProfileService

    store = _store(tmp_path, {"prompt_profiles": {"Universal": "u", "Mine": "m", "Spare": "s"},
                              "active_profile": "Universal", "model": "authgpt/gpt-6-luna"})
    service = ProfileService(store)
    written: list = []
    store.observe_all(lambda key, value: written.append(key))

    service.save("Korean_html2text", "Korean_html2text", "edited KR", keep_active=True)
    assert store.get("prompt_profiles")["Korean_html2text"] == "edited KR"
    assert store.get("active_profile") == "Universal" and store.get("text_extraction_method") is None
    assert service.save_as("Chat copy", "copy text", keep_active=True) == "Chat copy"
    assert service.duplicate("Mine", keep_active=True) == "Mine (copy)"
    assert service.new(keep_active=True) == "New Profile #1"
    assert store.get("active_profile") == "Universal"
    # a NON-active built-in reset / custom delete keeps the active profile and the extraction method
    assert service.delete_or_reset("Korean_html2text", keep_active=True) == "reset"
    assert store.get("prompt_profiles")["Korean_html2text"] == _desktop_defaults()["Korean_html2text"]
    assert service.delete_or_reset("Spare", keep_active=True) == "deleted"
    assert store.get("active_profile") == "Universal" and store.get("text_extraction_method") is None
    assert set(written) <= set(PROFILE_CONFIG_KEYS)

    # renaming / deleting the profile in use follows the desktop rule even with keep_active
    service.select("Mine")
    assert service.save("Mine", "Mine renamed", "m2", keep_active=True) == "Mine renamed"
    assert store.get("active_profile") == "Mine renamed"
    service.delete_or_reset("Mine renamed", keep_active=True)
    assert store.get("active_profile") == "Universal"

    # without keep_active: the desktop Save Profile semantics (the saved profile becomes active)
    service.save("Japanese_html2text", "Japanese_html2text", "edited JP")
    assert store.get("active_profile") == "Japanese_html2text"
    store._saver.close()


def test_keep_active_with_a_dangling_or_absent_active_profile(tmp_path):
    from glossarion_mobile.ui.screens.profiles import ProfileService

    store = _store(tmp_path, {"prompt_profiles": {"Universal": "u", "Mine": "m"}, "active_profile": "Gone"})
    ProfileService(store).save("Mine", "Mine", "m2", keep_active=True)
    assert store.get("active_profile") == "Gone" and store.get("prompt_profiles")["Mine"] == "m2"
    store._saver.close()
    fresh = _store(tmp_path, {}, name="fresh.json")
    ProfileService(fresh).save("Universal", "Universal", "my universal", keep_active=True)
    assert "active_profile" not in fresh.keys() and "text_extraction_method" not in fresh.keys()
    assert fresh.get("prompt_profiles")["Universal"] == "my universal" and len(fresh.get("prompt_profiles")) == 18
    fresh._saver.close()


# ==========================================================================
# #15: Chat settings › Edit prompt / New profile… / Manage…
# ==========================================================================


def _sheet(store, overrides=None, *, scope="chat", **kwargs):
    from glossarion_mobile.ui.sheets.chat_settings import ChatSettingsSheet

    chats = kwargs.pop("chats", None) or Chats(overrides)
    ctx = kwargs.pop("ctx", None) or Ctx(store)
    sheet = ChatSettingsSheet(cid="2", config=store, chats=chats, scope=scope, ctx=ctx, **kwargs)
    return sheet, chats, ctx


def test_edit_prompt_changes_the_shared_profile_only(tmp_path):
    from glossarion_mobile.ui.screens.prompt_editor import PromptEditorSheet

    store = _store(tmp_path, {"prompt_profiles": {"Universal": "u", "Mine": "mine text"}, "active_profile": "Universal"})
    sheet, chats, ctx = _sheet(store, {"profile": "Mine"})
    assert _card_text(sheet, "preview") == "mine text"
    assert not _by_key(_card(sheet), "setting-profile-prompt-edit").disabled
    editor = sheet.edit_prompt()
    assert isinstance(editor, PromptEditorSheet) and editor.title == "Mine" and editor.field.value == "mine text"
    assert ctx.page.dialogs[-1] is editor.sheet
    editor.field.value = "new text {target_lang}"
    assert editor.save() is True
    assert store.get("prompt_profiles")["Mine"] == "new text {target_lang}"
    assert store.get("active_profile") == "Universal" and chats.values == {"profile": "Mine"}
    assert _card_text(sheet, "preview") == "new text {target_lang}"  # the card was rebuilt after the save

    # saving the editor unchanged writes nothing (the shared save strips: no "modified" for nothing)
    written: list = []
    store.observe_all(lambda key, value: written.append(key))
    unchanged = sheet.edit_prompt()
    assert unchanged.save() is True and written == []

    # a built-in edited, then put back with the editor's Reset to default: the shared reset (exact text)
    sheet.set_value("profile", "Korean_html2text")
    edit = sheet.edit_prompt()
    edit.field.value = "changed"
    edit.save()
    assert sheet.listing.is_modified("Korean_html2text")
    again = sheet.edit_prompt()
    assert again.default == _desktop_defaults()["Korean_html2text"]
    again._on_reset()
    again.save()
    assert store.get("prompt_profiles")["Korean_html2text"] == _desktop_defaults()["Korean_html2text"]
    assert not sheet.listing.is_modified("Korean_html2text")
    assert store.get("active_profile") == "Universal" and store.get("text_extraction_method") is None

    # All chats: the card and Edit act on the global active profile
    sheet._on_scope(types.SimpleNamespace(control=types.SimpleNamespace(selected=["global"])))
    assert sheet.value_of("profile") == "Universal"
    editor = sheet.edit_prompt()
    editor.field.value = "global universal"
    editor.save()
    assert store.get("prompt_profiles")["Universal"] == "global universal"
    store._saver.close()


def _confirm(dialog):
    dialog.actions[1].on_click(None)


def test_new_profile_copies_the_current_profile(tmp_path):
    store = _store(tmp_path, {"prompt_profiles": {"Universal": "u", "Mine": "mine text"}, "active_profile": "Universal"})
    sheet, chats, ctx = _sheet(store, {"profile": "Mine"})
    dialog = sheet.new_profile()
    _dialog, field = sheet.new_profile_dialog
    assert ctx.page.dialogs[-1] is dialog and field.value == "Mine (copy)"
    assert "Starts as a copy of 'Mine'." in _texts(dialog)
    field.value = "Mine v2"
    _confirm(dialog)
    assert store.get("prompt_profiles")["Mine v2"] == "mine text"
    assert chats.values["profile"] == "Mine v2"  # This chat: the new profile is the chat's own
    assert store.get("active_profile") == "Universal"  # the global/desktop profile is untouched
    assert "Mine v2" in [o.key for o in _profile_dropdown(sheet).options] and _profile_dropdown(sheet).value == "Mine v2"
    assert sheet.prompt_editor.title == "Mine v2" and dialog.open is False  # then it opens in the editor

    # a taken name keeps the dialog open with the desktop message
    dialog = sheet.new_profile()
    _dialog, field = sheet.new_profile_dialog
    field.value = "Universal"
    _confirm(dialog)
    assert field.error == "A profile with this name already exists. Choose another name." and dialog.open

    # All chats: the new profile becomes the active one, with the desktop extraction-method switch
    sheet._on_scope(types.SimpleNamespace(control=types.SimpleNamespace(selected=["global"])))
    dialog = sheet.new_profile()
    sheet.new_profile_dialog[1].value = "My_html2text"
    _confirm(dialog)
    assert store.get("active_profile") == "My_html2text" and store.get("text_extraction_method") == "enhanced"
    assert store.get("prompt_profiles")["My_html2text"] == "u"  # a copy of the active Universal
    store._saver.close()


def test_series_defaults_new_profile_sets_the_series_default(tmp_path):
    from glossarion_mobile.state.series import SeriesDefaultsChats, SeriesStore

    store = _store(tmp_path, {"prompt_profiles": {"Universal": "u"}, "active_profile": "Universal"})
    series = SeriesStore(str(tmp_path / "mobile_series.json"))
    saga = series.create("Saga")
    sheet, _chats, _ctx = _sheet(store, chats=SeriesDefaultsChats(series, saga.id), subject="series")
    dialog = sheet.new_profile()
    sheet.new_profile_dialog[1].value = "Saga style"
    _confirm(dialog)
    assert series.defaults(saga.id)["profile"] == "Saga style" and store.get("active_profile") == "Universal"
    store._saver.close()


def test_prompt_card_role_skip_missing_and_manage(tmp_path):
    from glossarion_mobile.ui.sheets.chat_settings import ChatSettingsSheet

    store = _store(tmp_path, {"prompt_profiles": {"Universal": "u"}, "system_prompt_to_user": True,
                              "direct_text_skip_prompt_profile": True})
    managed: list = []
    sheet, chats, ctx = _sheet(store, {"profile": "Gone"}, on_manage_profiles=lambda: managed.append(1))
    card = _card(sheet)
    assert "User prompt" in _texts(card) and "System prompt" not in _texts(card)  # desktop ↕️/🔀 role
    assert _card_text(sheet, "skip") == "Skip prompt profile is on: runs ignore this prompt."
    # a chat whose profile no longer exists: "(missing)", Edit off, the missing note
    options = _profile_dropdown(sheet).options
    assert options[0].key == "Gone" and options[0].text == "Gone (missing)"
    assert _by_key(card, "setting-profile-prompt-edit").disabled and _card_text(sheet, "missing")
    assert sheet.edit_prompt() is None and ctx.messages[-1] == "The prompt profile 'Gone' no longer exists"
    _by_key(card, "setting-profile-prompt-manage").on_click(None)
    assert managed == [1]
    # the default Manage…: close the sheet, open Settings › Profiles & prompts
    plain, _c, ctx2 = _sheet(store, {"profile": "Universal"})
    plain.manage_profiles()
    assert ctx2.routes == [("settings.profiles", None)]
    # a heading picked by a client that ignores ``disabled`` changes nothing
    plain._on_select("profile", "__group__:Specialised")
    assert _c.values == {"profile": "Universal"}
    store._saver.close()

    # no settings store (a plain dict config): read-only card, actions off with a reason, no crash
    class Config(dict):
        def set_many(self, updates):
            self.update(updates)

    bare = ChatSettingsSheet(cid="2", config=Config(), chats=Chats(), profiles=["Universal"])
    card = _card(bare)
    assert _by_key(card, "setting-profile-prompt-edit").disabled and _by_key(card, "setting-profile-prompt-new").disabled
    assert "Needs the settings store" in _texts(card) and bare.new_profile() is None


def test_owner_report_17_chat_settings_has_no_key_settings(tmp_path):
    """Keys get their own button in the drawer footer: Chat settings (This chat, All chats, Series
    defaults) shows no API key, key pool or Keys control."""
    from glossarion_mobile.state.series import SeriesDefaultsChats, SeriesStore

    store = _store(tmp_path, {"prompt_profiles": {"Universal": "u"}})
    series = SeriesStore(str(tmp_path / "mobile_series.json"))
    saga = series.create("Saga")
    sheets = [_sheet(store)[0], _sheet(store, scope="global")[0],
              _sheet(store, chats=SeriesDefaultsChats(series, saga.id), subject="series")[0]]
    for sheet in sheets:
        for tile in sheet.sections.values():
            tile.expanded = True
        texts = " | ".join(_texts(sheet.column)).casefold()
        for needle in ("api key", "api keys", "key pool", "multi-key", "keys"):
            assert needle not in texts, (needle, sheet.scope, sheet.subject)
        keys = [str(getattr(c, "key", "") or "") for c in _walk(sheet.column)]
        assert not [k for k in keys if "keys" in k.casefold() or "api_key" in k.casefold()]
    store._saver.close()


# ==========================================================================
# renamed profiles are followed, deleted ones fall back to the inherited profile
# ==========================================================================


def test_chats_and_series_follow_a_rename_and_inherit_after_a_delete(tmp_path, desktop_store_cls):
    from glossarion_mobile.state.series import SeriesStore
    from glossarion_mobile.ui.chat.integration import ChatFeature
    from glossarion_mobile.ui.chat.run_request import config_overrides
    from glossarion_mobile.ui.screens.profiles import ProfileService

    _TC._desktop_history(tmp_path)
    adapter = _TC._adapter(desktop_store_cls, tmp_path)
    store = _store(tmp_path, {"prompt_profiles": {"Universal": "u", "Mine": "m", "Other": "o"},
                              "active_profile": "Universal"})
    series = SeriesStore(str(tmp_path / "mobile_series.json"))
    saga = series.create("Saga", defaults={"profile": "Mine"})
    notes: list = []
    app = types.SimpleNamespace(page=None, dispatcher=None, paths=None, config_store=store,
                                series=types.SimpleNamespace(store=series),
                                notify=lambda message, *a: notes.append(message))
    feature = ChatFeature(app, chats=adapter, jobs=None, oauth=types.SimpleNamespace())
    feature.env = feature.build_env()
    feature._hook_profile_overrides()
    try:
        adapter.set_override("2", "profile", "Mine")
        service = ProfileService(store)
        # renamed in Settings › Profiles & prompts (the desktop Save Profile rename): chats and series follow
        service.save("Mine", "Mine renamed", "m2")
        assert adapter.own_overrides("2")["profile"] == "Mine renamed"
        assert series.defaults(saga.id)["profile"] == "Mine renamed"
        assert config_overrides(adapter.overrides("2"))["active_profile"] == "Mine renamed"
        # deleted: they inherit again (never a silent first profile) and the user is told
        service.delete_or_reset("Mine renamed")
        assert "profile" not in adapter.own_overrides("2") and "profile" not in series.defaults(saga.id)
        assert "active_profile" not in config_overrides(adapter.overrides("2"))
        assert notes and "no longer exists" in notes[-1]
        # gone through an import / restore (no rename known): inherit
        adapter.set_override("2", "profile", "Other")
        store.set_many({"prompt_profiles": {"Universal": "u"}})
        assert "profile" not in adapter.own_overrides("2")
        # stale names from before this launch: the startup check clears them too
        adapter.set_override("2", "profile", "Ghost")
        series.set_default(saga.id, "profile", "Ghost")
        changed = feature.reconcile_profile_overrides()
        assert changed == {"chats": {"2": None}, "series": {saga.id: None}}
        # names that exist are left alone (always-include built-ins the desktop start-up adds back included;
        # Manga_JP is a deletable built-in, gone with this import like on the desktop)
        adapter.set_override("2", "profile", "Korean_html2text")
        assert feature.reconcile_profile_overrides() == {"chats": {}, "series": {}}
        assert adapter.own_overrides("2")["profile"] == "Korean_html2text"
        adapter.set_override("2", "profile", "Manga_JP")
        assert feature.reconcile_profile_overrides() == {"chats": {"2": None}, "series": {}}
    finally:
        feature.close()
        store._saver.close()


# ==========================================================================
# the real chat view on a fresh config
# ==========================================================================


def test_chat_view_pickers_on_a_fresh_config(tmp_path, desktop_store_cls):
    from glossarion_mobile.state.app_state import AppState
    from glossarion_mobile.ui.chat.chat_view import ChatView
    from glossarion_mobile.ui.chat.integration import ChatFeature
    from glossarion_mobile.ui.sheets.chat_settings import ChatSettingsSheet

    async def scenario():
        _TC._desktop_history(tmp_path)
        adapter = _TC._adapter(desktop_store_cls, tmp_path)
        store, schema = _TC._config_store(tmp_path)
        page = types.SimpleNamespace(width=412, height=900, show_dialog=lambda d: None, update=lambda *a: None)
        notes: list = []
        app = types.SimpleNamespace(page=page, dispatcher=None, paths=None, config_store=store,
                                    settings=types.SimpleNamespace(schema=schema))
        feature = ChatFeature(app, chats=adapter, jobs=_TC.FakeJobService(), oauth=types.SimpleNamespace())
        env = feature.build_env()
        env.oauth = None  # no sign-in service in this view
        state = AppState()
        state.current_chat.set("2")
        routes: list = []
        view = ChatView(page, state=state, navigate=lambda *a, **k: routes.append(a),
                        notify=lambda message, *a, **k: notes.append(message), env=env)
        try:
            sheet = view.open_chat_settings()
            assert isinstance(sheet, ChatSettingsSheet)
            assert len([o for o in _profile_dropdown(sheet).options if not o.disabled]) == 18
            assert _card(sheet) is not None and not _by_key(_card(sheet), "setting-profile-prompt-new").disabled
            # Integrate: the chat's sheet edits on the chat env and its profile service; Manage… leaves the chat
            assert sheet.ctx is env and sheet.profile_service is env.profile_service
            closed: list = []
            sheet.close = lambda: closed.append(True)
            sheet.manage_profiles()
            assert closed == [True] and routes[-1] == ("settings.profiles",)
            model_sheet = view.open_model_sheet("profile")
            assert list(model_sheet.profiles) == TRANSLATION + SPECIALISED
            # ModelSheet › Profile: the Specialised heading, and previews from the shared listing (built-ins
            # have no stored text on a fresh config). The app installs a SheetEnv with the settings store; a
            # fresh one here, so neither another test's installed SheetEnv nor this store leaks either way.
            from glossarion_mobile.ui.sheets.model_sheet import SheetEnv

            model_sheet.env = SheetEnv(store=store)
            rows = model_sheet.profile_rows()
            keys = [str(getattr(row, "key", "") or "") for row in rows]
            assert keys.count("profile-group-specialised") == 1
            assert model_sheet._profile_preview("Manga_JP") and model_sheet._profile_preview("Universal")
            assert len(model_sheet._profile_texts()) == 18
            view.run_slash("/profile manga_jp")
            assert notes[-1] == "Prompt profile: Manga_JP"
        finally:
            feature.runs.detach()
            adapter.close()
            store._saver.close()

    asyncio.run(scenario())


@needs_flet
def test_profile_section_links_to_the_profile_editors():
    """Integrate: Settings › Profile & System Prompt (main.prompt, whose prompt_profiles tile is raw JSON) links
    to Profiles & prompts and Assistant prefill, real routes with real icons."""
    import flet as ft

    from glossarion_mobile.ui.router import ROUTES_BY_NAME
    from glossarion_mobile.ui.settings.section_page import STATIC_LINKS

    links = STATIC_LINKS["main.prompt"]
    assert [link[2] for link in links] == ["settings.profiles", "settings.prefill"]
    assert all(link[2] in ROUTES_BY_NAME and getattr(ft.Icons, link[1], None) is not None for link in links)


@needs_flet
def test_profile_section_tiles_only_choose_real_profiles_and_keep_the_profiles(tmp_path):
    """Device fixes 2 (#16, review): Settings › Profile & System Prompt's "Profile" is the desktop combo over the
    shared listing (a name that is not a profile is never stored: every run would silently use the first
    profile), and the raw "Prompt profiles" JSON is read-only on mobile - nor does "Reset this section" wipe the
    custom profiles behind it."""
    from glossarion_mobile.state.setting_writes import write_setting
    from glossarion_mobile.ui.router import parse_route
    from glossarion_mobile.ui.screens.profiles import ProfileService
    from glossarion_mobile.ui.settings.context import SettingsContext
    from glossarion_mobile.ui.settings.schema_access import SchemaAccess
    from glossarion_mobile.ui.settings.section_page import SectionPage
    from glossarion_mobile.ui.settings.tiles import ProfileTile

    profiles = {**_desktop_defaults(), "Wuxia House Style": "wuxia prompt"}
    store = _store(tmp_path, {"prompt_profiles": profiles, "active_profile": "Wuxia House Style"})
    ctx = SettingsContext(page=None, store=store, schema=SchemaAccess(), notify=lambda *args: None)
    page = SectionPage(parse_route("/settings/s/main.prompt"), ctx)
    page.get_body()

    tile = page.tile("active_profile")
    names = list(ProfileService(store).listing().names)
    assert isinstance(tile, ProfileTile) and tile.editable
    assert [value for value, _label in tile.options()] == names and "Wuxia House Style" in names
    assert tile.value() == "Wuxia House Style"
    assert tile.choose(names.index("Korean_html2text"))  # "Use this profile": the extraction method follows
    assert store.get("active_profile") == "Korean_html2text" and store.get("text_extraction_method") == "enhanced"
    assert write_setting(store, "active_profile", "wuxia house style") == []  # not a profile: refused
    assert store.get("active_profile") == "Korean_html2text"
    assert tile.apply("wuxia house style") and store.get("active_profile") == "Korean_html2text"

    json_tile = page.tile("prompt_profiles")
    assert not json_tile.editable and json_tile.readonly_reason == "Edited in Settings › Profiles & prompts"
    assert json_tile.activate() is None
    removed = page.reset_section()
    assert "active_profile" in removed and "prompt_profiles" not in removed
    assert store.get("prompt_profiles")["Wuxia House Style"] == "wuxia prompt"
    store._saver.close()
