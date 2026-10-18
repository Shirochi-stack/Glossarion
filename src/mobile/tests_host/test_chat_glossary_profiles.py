"""Owner request: "chat settings is missing the glossary prompt profile like chapter translation does. Add it
in the glossary section and add it to the Auto glossary pop up in the response card".

* Chat settings › Glossary has a "Glossary prompt profile" dropdown listing the Glossary Manager bucket of
  the effective glossary mode (Minimal -> minimal, Balanced / Full -> balanced_full); modes without an AI
  extraction prompt show a note instead.
* All chats selects the profile globally (``GlossaryPromptProfiles.select``: the bucket's current prompt keys
  change); This chat stores a per-chat override that only that chat's runs apply as config overrides.
* The Plan card's Glossary sheet has the same dropdown (global write, like its mode row) and stays only as
  tall as its rows.

Run from src/mobile with the mobile venv:
    .venv/Scripts/python.exe -m pytest -p no:cacheprovider -W ignore tests_host/test_chat_glossary_profiles.py -q
"""

from __future__ import annotations

import asyncio
import importlib.util
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


def _gd():
    try:
        import glossary_document as gd
        import prompt_profiles  # noqa: F401
    except Exception:
        return None
    return gd if hasattr(gd, "GlossaryPromptProfiles") else None


gd = _gd()
needs_flet = pytest.mark.skipif(not (_has("flet") and _has("msgpack")), reason="flet/msgpack not installed")
needs_profiles = pytest.mark.skipif(gd is None, reason="glossary_document.GlossaryPromptProfiles not importable")

_MK_SPEC = importlib.util.spec_from_file_location("_glossarion_mk_helpers_gprof", Path(__file__).with_name("test_models_keys.py"))
MK = importlib.util.module_from_spec(_MK_SPEC)
_MK_SPEC.loader.exec_module(MK)


class _Config(dict):
    def set_many(self, updates):
        self.update(updates)


class _Chats:
    """Two chats' sidecar overrides (ChatStoreAdapter surface the sheet uses)."""

    def __init__(self):
        self.values = {"2": {}, "3": {}}

    def overrides(self, cid):
        return dict(self.values.setdefault(str(cid), {}))

    own_overrides = overrides

    def meta(self, cid):
        return {}

    def set_override(self, cid, name, value):
        from glossarion_mobile.state.chat_store_adapter import OVERRIDE_KEYS

        assert name in OVERRIDE_KEYS
        values = self.values.setdefault(str(cid), {})
        if value is None:
            values.pop(name, None)
        else:
            values[name] = value

    def set_meta(self, cid, name, value):
        return None

    def reset_overrides(self, cid):
        self.values[str(cid)] = {}


def _config(mode="balanced"):
    return _Config({
        "auto_glossary_mode": mode,
        "direct_text_glossary_override_mode": "none",
        "manual_glossary_prompt3": "BAL default", "manual_glossary_prompt": "BAL default",
        "unified_auto_glosary_prompt3": "MIN default",
        "glossary_prompt_profiles": {"balanced_full": {"Strict": "BAL strict", "Loose": "BAL loose"},
                                     "minimal": {"Tiny": "MIN tiny"}},
        "active_glossary_prompt_profiles": {},
        "glossary_prompt_profile_defaults": {"balanced_full": "BAL default", "minimal": "MIN default"},
    })


def _sheet(config, chats=None, scope="chat", cid="2"):
    from glossarion_mobile.ui.sheets.chat_settings import ChatSettingsSheet

    return ChatSettingsSheet(cid=cid, config=config, chats=chats or _Chats(), scope=scope)


def _find(root, predicate):
    from flet.controls.base_control import BaseControl

    if root is None:
        return None
    if predicate(root):
        return root
    for name in ("content", "controls", "title", "subtitle", "leading", "trailing"):
        value = getattr(root, name, None)
        items = value if isinstance(value, list) else [value]
        for item in items:
            if isinstance(item, BaseControl):
                found = _find(item, predicate)
                if found is not None:
                    return found
    return None


def _glossary_dropdown(sheet):
    tile = sheet.sections["glossary"]
    return _find(tile, lambda c: type(c).__name__ == "Dropdown" and c.label == "Glossary prompt profile")


def _glossary_note(sheet):
    return _find(sheet.sections["glossary"], lambda c: getattr(c, "key", None) == "setting-glossary-profile-note")


# ==========================================================================
# the bucket of the effective mode
# ==========================================================================


@needs_profiles
def test_prompt_bucket_follows_the_effective_glossary_mode():
    from glossarion_mobile.ui.chat.direct_text_rules import glossary_prompt_bucket
    from glossarion_mobile.ui.chat.run_request import GLOSSARY_PROMPT_CONFIG_KEYS

    get = lambda mode: (lambda key, default=None: {"auto_glossary_mode": mode}.get(key, default))
    assert glossary_prompt_bucket("none", get("minimal")) == "minimal"
    assert glossary_prompt_bucket("none", get("balanced")) == "balanced_full"
    assert glossary_prompt_bucket("attachments_only", get("full")) == "balanced_full"
    for mode in ("off", "off_no_automap", "no_glossary", "single_pass"):
        assert glossary_prompt_bucket("none", get(mode)) is None, mode
    assert glossary_prompt_bucket("no_glossary", get("balanced")) is None  # the chat forces No glossary
    assert glossary_prompt_bucket("manual", get("balanced")) is None
    # the run-override keys mirror the shared bucket metadata
    keys = {k for b in ("balanced_full", "minimal") for k in (gd.glossary_prompt_profile_meta(b)["config_key"],
                                                              gd.glossary_prompt_profile_meta(b)["legacy_config_key"]) if k}
    assert keys == set(GLOSSARY_PROMPT_CONFIG_KEYS)


@needs_flet
@needs_profiles
def test_chat_settings_dropdown_lists_the_bucket_of_the_effective_mode():
    config = _config("balanced")
    sheet = _sheet(config)
    dropdown = _glossary_dropdown(sheet)
    assert dropdown is not None and [o.key for o in dropdown.options] == ["Default", "Strict", "Loose", "On-device"]  # + the built-in profile
    assert dropdown.value == "Default"
    assert _find(sheet.sections["glossary"], lambda c: getattr(c, "value", None) == "Inherited from: All chats")

    config["auto_glossary_mode"] = "minimal"
    sheet.rebuild()
    assert [o.key for o in _glossary_dropdown(sheet).options] == ["Default", "Tiny", "On-device"]

    config["auto_glossary_mode"] = "off"
    sheet.rebuild()
    assert _glossary_dropdown(sheet) is None and "not used" in _glossary_note(sheet).value
    # This chat forcing No glossary hides it as well
    config["auto_glossary_mode"] = "balanced"
    sheet.set_value("glossary_override_mode", "no_glossary")
    assert _glossary_dropdown(sheet) is None and _glossary_note(sheet) is not None


# ==========================================================================
# All chats vs This chat
# ==========================================================================


@needs_flet
@needs_profiles
def test_all_chats_selection_changes_the_global_prompt_keys():
    config = _config("balanced")
    chats = _Chats()
    sheet = _sheet(config, chats, scope="global")
    sheet._on_select("glossary_prompt_profile_balanced_full", "Strict")
    assert config["active_glossary_prompt_profiles"] == {"balanced_full": "Strict"}
    assert config["manual_glossary_prompt3"] == "BAL strict" and config["manual_glossary_prompt"] == "BAL strict"
    assert config["unified_auto_glosary_prompt3"] == "MIN default"  # the Minimal bucket is untouched
    assert chats.values == {"2": {}, "3": {}} and _glossary_dropdown(sheet).value == "Strict"
    sheet._on_select("glossary_prompt_profile_balanced_full", "Default")
    assert config["manual_glossary_prompt3"] == "BAL default" and config["active_glossary_prompt_profiles"] == {}


@needs_flet
@needs_profiles
def test_this_chat_selection_overrides_only_that_chats_job_config():
    from glossarion_mobile.ui.chat.chat_view import ChatView
    from glossarion_mobile.ui.chat.run_request import (
        GLOSSARY_PROMPTS,
        config_overrides,
        run_config_of,
        with_run_config,
    )

    config = _config("balanced")
    chats = _Chats()
    sheet = _sheet(config, chats, scope="chat")
    sheet._on_select("glossary_prompt_profile_balanced_full", "Loose")
    assert chats.values["2"] == {"glossary_prompt_profile_balanced_full": "Loose"}
    assert config["manual_glossary_prompt3"] == "BAL default"  # the global prompt is unchanged
    assert config["active_glossary_prompt_profiles"] == {}
    dropdown = _glossary_dropdown(sheet)
    assert dropdown.value == "Loose" and sheet.is_overridden("glossary_prompt_profile_balanced_full")

    view = types.SimpleNamespace(env=types.SimpleNamespace(store=config))
    mine = ChatView._with_glossary_prompts(view, chats.overrides("2"))
    other = ChatView._with_glossary_prompts(view, chats.overrides("3"))
    job = config_overrides(mine)
    assert job["manual_glossary_prompt3"] == "BAL loose" and job["manual_glossary_prompt"] == "BAL loose"
    assert "unified_auto_glosary_prompt3" not in job
    assert config_overrides(other) == {} and GLOSSARY_PROMPTS not in other
    # a Resume with other Run options keeps the chat's glossary prompts (they are chat keys, not Run options)
    assert run_config_of(job) == {} and with_run_config(job, {"batch_size": 3})["manual_glossary_prompt3"] == "BAL loose"
    # a deleted profile is left out: the run keeps the global prompt
    assert config_overrides(ChatView._with_glossary_prompts(view, {"glossary_prompt_profile_minimal": "Gone"})) == {}

    # ↺ goes back to All chats
    sheet.reset("glossary_prompt_profile_balanced_full")
    assert chats.values["2"] == {} and _glossary_dropdown(sheet).value == "Default"


# ==========================================================================
# the Plan card's Glossary sheet
# ==========================================================================


def _plan_sheet(page, config, **kwargs):
    from glossarion_mobile.services.glossary import GlossaryService
    from glossarion_mobile.ui.glossary.common import GlossaryContext
    from glossarion_mobile.ui.glossary.sheets import PlanGlossarySheet

    ctx = GlossaryContext(service=GlossaryService(config=config), page=page, platform="android",
                          notify=lambda *a, **k: None, navigate=lambda *a, **k: None)
    noop = lambda: None
    return PlanGlossarySheet(ctx, on_load_file=noop, on_clear=noop, on_review=noop, on_mode=noop,
                             effective="Glossary: Balanced (auto)", on_settings=noop, **kwargs)


@needs_flet
@needs_profiles
def test_plan_glossary_sheet_profile_dropdown_writes_globally():
    config = _config("balanced")
    changed = []

    async def scenario():
        _conn, session = MK._fake_session("android")
        session.apply_page_patch({"width": 412, "height": 915})
        sheet = _plan_sheet(session.page, config, prompt_bucket="balanced_full", chat_profile="Loose",
                            on_profile_changed=lambda: changed.append(True))
        assert [o.key for o in sheet.profile_dropdown.options] == ["Default", "Strict", "Loose", "On-device"]  # + the built-in profile
        assert "this chat uses “Loose”" in sheet.profile_row.controls[-1].value
        assert sheet.select_profile("Strict") == "BAL strict" and changed == [True]
        assert config["manual_glossary_prompt3"] == "BAL strict"
        assert config["active_glossary_prompt_profiles"] == {"balanced_full": "Strict"}
        # a mode without an extraction prompt: a note, no dropdown
        off = _plan_sheet(session.page, config, prompt_bucket=None)
        assert off.profile_dropdown is None and "not used" in off.profile_row.value

    asyncio.run(scenario())


# ==========================================================================
# the glossary prompt card (owner: "the in built minimal and balanced/full glossary prompts to be configured")
# ==========================================================================

_CP_SPEC = importlib.util.spec_from_file_location("_glossarion_cp_helpers_gprof", Path(__file__).with_name("test_chat_prompts.py"))
CP = importlib.util.module_from_spec(_CP_SPEC)
_CP_SPEC.loader.exec_module(CP)

#: the prompt keys' schema defaults (what Settings › Glossary shows when config.json has none)
BUILTIN = {"manual_glossary_prompt3": "BUILTIN BAL\nsecond line", "unified_auto_glosary_prompt3": "BUILTIN MIN"}


class _GCtx(CP.Ctx):
    """The SettingsContext surface: the prompt editor, the schema defaults and Manage… (open_setting)."""

    def __init__(self, store=None):
        super().__init__(store)
        self.schema = types.SimpleNamespace(effective_default=lambda key: BUILTIN.get(key))
        self.opened: list = []

    def open_setting(self, section_id, key=None):
        self.opened.append(section_id)
        return section_id


def _carded_sheet(config, chats=None, scope="chat"):
    from glossarion_mobile.ui.sheets.chat_settings import ChatSettingsSheet

    ctx = _GCtx(config)
    return ChatSettingsSheet(cid="2", config=config, chats=chats or _Chats(), scope=scope, ctx=ctx), ctx


def _glossary_card_text(sheet, suffix):
    control = CP._by_key(sheet.sections["glossary"], f"setting-glossary-prompt-{suffix}")
    return None if control is None else getattr(control, "value", control)


@needs_flet
@needs_profiles
def test_glossary_card_shows_the_selected_profile_of_the_modes_bucket():
    from glossarion_mobile.ui.screens.profiles import prompt_preview

    config = _config("balanced")
    sheet, _ctx = _carded_sheet(config)
    card = CP._by_key(sheet.sections["glossary"], "setting-glossary-prompt")
    texts = CP._texts(card)
    assert "Extraction prompt" in texts and "Built-in" in texts  # Default is the bucket's built-in prompt
    assert any(t.startswith("Profiles are shared") for t in texts)
    assert any(t.startswith("11 chars") for t in texts)  # "BAL default": N chars · ≈ N tokens
    assert _glossary_card_text(sheet, "preview") == "BAL default"
    for suffix in ("edit", "new", "manage"):
        assert not CP._by_key(card, f"setting-glossary-prompt-{suffix}").disabled, suffix
    sheet._on_select("glossary_prompt_profile_balanced_full", "Strict")
    assert _glossary_card_text(sheet, "preview") == "BAL strict"
    sheet._on_select("glossary_prompt_profile_balanced_full", "On-device")
    assert _glossary_card_text(sheet, "preview") == prompt_preview(gd.ON_DEVICE_GLOSSARY_PROMPTS["balanced_full"])

    # Minimal mode: the Minimal bucket's prompts
    config["auto_glossary_mode"] = "minimal"
    sheet.rebuild()
    assert _glossary_card_text(sheet, "preview") == "MIN default"
    sheet._on_select("glossary_prompt_profile_minimal", "Tiny")
    assert _glossary_card_text(sheet, "preview") == "MIN tiny"
    # a mode without an extraction prompt: the note, no card
    config["auto_glossary_mode"] = "off"
    sheet.rebuild()
    assert sheet.glossary_card is None and _glossary_note(sheet) is not None
    assert CP._by_key(sheet.sections["glossary"], "setting-glossary-prompt") is None


@needs_flet
@needs_profiles
def test_glossary_card_default_shows_the_builtin_prompt():
    config = _Config({"auto_glossary_mode": "full", "direct_text_glossary_override_mode": "none"})
    sheet, _ctx = _carded_sheet(config, scope="global")
    assert _glossary_card_text(sheet, "preview") == "BUILTIN BAL\nsecond line"
    editor = sheet.glossary_card.edit_prompt()
    assert editor.title == "Default" and editor.field.value == "BUILTIN BAL\nsecond line"
    assert editor.default == "BUILTIN BAL\nsecond line"  # Reset to default: the built-in text
    config["auto_glossary_mode"] = "minimal"
    sheet.rebuild()
    assert _glossary_card_text(sheet, "preview") == "BUILTIN MIN"


@needs_flet
@needs_profiles
def test_glossary_edit_prompt_saves_the_profile_without_switching_it():
    from glossarion_mobile.services.glossary import PROMPT_PLACEHOLDERS
    from glossarion_mobile.ui.screens.prompt_editor import PromptEditorSheet

    config = _config("balanced")
    chats = _Chats()
    sheet, ctx = _carded_sheet(config, chats)
    sheet._on_select("glossary_prompt_profile_balanced_full", "Loose")
    editor = sheet.glossary_card.edit_prompt()
    assert isinstance(editor, PromptEditorSheet) and editor.title == "Loose" and editor.field.value == "BAL loose"
    assert editor.placeholders == PROMPT_PLACEHOLDERS["balanced_full"] and "{entries}" in editor.placeholders
    assert ctx.page.dialogs[-1] is editor.sheet
    editor.field.value = "BAL loose v2 {language}"
    assert editor.save() is True
    assert config["glossary_prompt_profiles"]["balanced_full"]["Loose"] == "BAL loose v2 {language}"
    # This chat: the profile itself changed; the global selection and the run prompt did not
    assert config["active_glossary_prompt_profiles"] == {} and config["manual_glossary_prompt3"] == "BAL default"
    assert chats.values["2"] == {"glossary_prompt_profile_balanced_full": "Loose"}
    assert _glossary_card_text(sheet, "preview") == "BAL loose v2 {language}"  # rebuilt after the save
    written = dict(config)
    assert sheet.glossary_card.edit_prompt().save() is True and dict(config) == written  # unchanged: nothing

    # All chats: Default is the bucket's Default prompt; as the selected one it is the run prompt too
    sheet._on_scope(types.SimpleNamespace(control=types.SimpleNamespace(selected=["global"])))
    assert sheet.glossary_card.name == "Default"
    editor = sheet.glossary_card.edit_prompt()
    editor.field.value = "BAL default v2"
    editor.save()
    assert config["glossary_prompt_profile_defaults"]["balanced_full"] == "BAL default v2"
    assert config["manual_glossary_prompt3"] == "BAL default v2" and config["manual_glossary_prompt"] == "BAL default v2"
    assert config["active_glossary_prompt_profiles"] == {}
    assert config["unified_auto_glosary_prompt3"] == "MIN default"  # the Minimal bucket is untouched

    # Manage…: Settings › Glossary on the bucket's tab
    CP._by_key(sheet.sections["glossary"], "setting-glossary-prompt-manage").on_click(None)
    assert ctx.opened == ["glossary.balanced_full"]


@needs_flet
@needs_profiles
def test_glossary_new_profile_copies_and_selects_it():
    config = _config("balanced")
    chats = _Chats()
    sheet, ctx = _carded_sheet(config, chats)
    sheet._on_select("glossary_prompt_profile_balanced_full", "Strict")
    card = sheet.glossary_card
    dialog = card.new_profile()
    _dialog, field = card.name_dialog
    assert ctx.page.dialogs[-1] is dialog and field.value == "Strict (copy)"
    assert "Starts as a copy of 'Strict'." in CP._texts(dialog)
    field.value = "Strict v2"
    CP._confirm(dialog)
    assert config["glossary_prompt_profiles"]["balanced_full"]["Strict v2"] == "BAL strict"
    # This chat: the chat's own profile; the global selection is untouched
    assert chats.values["2"] == {"glossary_prompt_profile_balanced_full": "Strict v2"}
    assert config["active_glossary_prompt_profiles"] == {} and config["manual_glossary_prompt3"] == "BAL default"
    assert _glossary_dropdown(sheet).value == "Strict v2" and "Strict v2" in [o.key for o in _glossary_dropdown(sheet).options]
    assert card.editor.title == "Strict v2" and dialog.open is False  # then it opens in the editor

    # a taken name keeps the dialog open
    dialog = sheet.glossary_card.new_profile()
    field = sheet.glossary_card.name_dialog[1]
    field.value = "Default"
    CP._confirm(dialog)
    assert field.error and dialog.open

    # All chats: the new profile becomes the active one (the bucket's run prompt)
    sheet._on_scope(types.SimpleNamespace(control=types.SimpleNamespace(selected=["global"])))
    dialog = sheet.glossary_card.new_profile()
    sheet.glossary_card.name_dialog[1].value = "Mine"
    CP._confirm(dialog)
    assert config["active_glossary_prompt_profiles"] == {"balanced_full": "Mine"}
    assert config["glossary_prompt_profiles"]["balanced_full"]["Mine"] == "BAL default"  # a copy of Default
    assert config["manual_glossary_prompt3"] == "BAL default"


@needs_flet
@needs_profiles
def test_plan_glossary_sheet_prompt_card_writes_globally():
    from glossarion_mobile.services.glossary import GlossaryService
    from glossarion_mobile.ui.glossary.common import GlossaryContext
    from glossarion_mobile.ui.glossary.sheets import PlanGlossarySheet

    config = _config("minimal")
    changed = []

    async def scenario():
        _conn, session = MK._fake_session("android")
        session.apply_page_patch({"width": 412, "height": 915})
        settings = _GCtx(config)
        ctx = GlossaryContext(service=GlossaryService(config=config), page=session.page, platform="android",
                              notify=lambda *a, **k: None, navigate=lambda *a, **k: None, settings=settings)
        noop = lambda: None
        sheet = PlanGlossarySheet(ctx, on_load_file=noop, on_clear=noop, on_review=noop, on_mode=noop,
                                  effective="Glossary: Minimal (auto)", on_settings=noop, prompt_bucket="minimal",
                                  on_profile_changed=lambda: changed.append(True))
        assert sheet.prompt_card in sheet.tiles and sheet.tiles.index(sheet.prompt_card) == 2  # under the dropdown

        def preview():
            return CP._by_key(sheet.prompt_card, "pg-prompt-preview").value

        assert preview() == "MIN default"
        assert any(t.startswith("Profiles are shared") for t in CP._texts(sheet.prompt_card))
        assert sheet.select_profile("Tiny") == "MIN tiny" and preview() == "MIN tiny" and changed == [True]
        # Edit prompt: the active profile's text, the run prompt follows
        editor = sheet.glossary_card.edit_prompt()
        assert settings.page.dialogs[-1] is editor.sheet and "{max_names}" in editor.placeholders
        editor.field.value = "MIN tiny v2"
        editor.save()
        assert config["glossary_prompt_profiles"]["minimal"]["Tiny"] == "MIN tiny v2"
        assert config["unified_auto_glosary_prompt3"] == "MIN tiny v2" and preview() == "MIN tiny v2"
        # New profile…: the active one, like the dropdown
        dialog = sheet.glossary_card.new_profile()
        assert sheet.glossary_card.name_dialog[1].value == "Tiny (copy)"
        sheet.glossary_card.name_dialog[1].value = "Tinier"
        CP._confirm(dialog)
        assert config["active_glossary_prompt_profiles"] == {"minimal": "Tinier"}
        assert sheet.profile_dropdown.value == "Tinier" and preview() == "MIN tiny v2"
        # Manage…: Settings › Glossary on the Minimal tab
        sheet.glossary_card.manage()
        assert settings.opened == ["glossary.minimal"]
        # no extraction prompt: no card
        off = PlanGlossarySheet(ctx, on_load_file=noop, on_clear=noop, on_review=noop, on_mode=noop,
                                prompt_bucket=None)
        assert off.glossary_card is None and off.prompt_card not in off.tiles

    asyncio.run(scenario())
