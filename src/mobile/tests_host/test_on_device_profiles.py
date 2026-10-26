"""Built-in "On-device" prompt profiles for small on-device models (mobile llama.cpp, ``local/<stem>``).

* the translation profiles "On-device_BeautifulSoup" (html tags) and "On-device_html2text" (markdown) are
  desktop built-ins (always included, protected), so a fresh and an existing config both list them;
* the Balanced/Full and Minimal glossary buckets get an "On-device" profile (``glossary_document``),
  seeded without overwriting an edited text and protected from deletion like Default;
* choosing a ``local/`` model (``setting_writes.write_setting(store, "model", ...)``, the path every
  global model writer takes) selects the On-device translation profile of the text extraction method
  (switching the method while one is active switches to the other) and the glossary and review generator profiles
  and the current prompt keys runs read; choosing a model that is not ``local/`` again restores the
  remembered profiles where On-device is still the active one;
* the review generator's "On-device" profile holds both of its prompts (the review prompt and the Full
  Review Mode final prompt), and the review job sends the active profile's.

Runs on the real shared ``prompt_profiles`` / ``glossary_document`` and a real ``MobileConfigStore``
in a temp folder. Run from src/mobile:
    python -m pytest -p no:cacheprovider -W ignore tests_host/test_on_device_profiles.py
"""

from __future__ import annotations

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


def _backend_error() -> str:
    try:
        import glossary_document  # noqa: F401
        import prompt_profiles  # noqa: F401
        prompt_profiles.profile_state_from_config({})
    except Exception as exc:  # the flet-only venv lacks the backend dependencies
        return f"the shared backend is not importable here ({exc})"
    return ""


_BACKEND_ERROR = _backend_error()
pytestmark = pytest.mark.skipif(bool(_BACKEND_ERROR), reason=_BACKEND_ERROR or "backend importable")

LOCAL = "local/qwen3-0.6b-q4"
CLOUD = "authchatgpt/auto"
ON_DEVICE = "On-device"
ON_DEVICE_BS = "On-device_BeautifulSoup"
ON_DEVICE_H2T = "On-device_html2text"
TRANSLATION_TEXT = "You must only output {target_lang} translations while retaining the html tags of the following text."
HTML2TEXT_TEXT = "You must only output {target_lang} translations with markdown."


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    for name, sub in (("HOME", "home"), ("USERPROFILE", "home"), ("APPDATA", "appdata"),
                      ("GLOSSARION_LIBRARY_DIR", "Library"), ("GLOSSARION_DATA_DIR", "data")):
        folder = tmp_path / "_env" / sub
        folder.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv(name, str(folder))
    monkeypatch.setenv("OUTPUT_DIRECTORY", str(tmp_path / "_env" / "Output"))
    yield tmp_path


def _store(tmp_path: Path, config=None):
    from glossarion_mobile.state.config_store import MobileConfigStore

    path = tmp_path / "config.json"
    path.write_text(json.dumps(config or {}), encoding="utf-8")
    store = MobileConfigStore(str(path), debounce=10,
                              reader=lambda p, decrypt=True: json.loads(Path(p).read_text(encoding="utf-8")),
                              writer=lambda disk, p, backup=False: Path(p).write_text(json.dumps(disk), encoding="utf-8"))
    store.load()
    return store


def _on_device_glossary(bucket: str) -> str:
    import glossary_document as gd

    return gd.ON_DEVICE_GLOSSARY_PROMPTS[bucket]


def _existing_config() -> dict:
    """A config from before the On-device profiles: a custom translation profile active, Balanced/Full on
    Default with its own prompt, Minimal on a custom profile."""
    import prompt_profiles

    defaults = dict(prompt_profiles.profile_state_from_config({}).default_prompts)
    profiles = {name: text for name, text in defaults.items() if not name.startswith(ON_DEVICE)}
    profiles["My style"] = "Translate to {target_lang}, my way."
    return {
        "prompt_profiles": profiles,
        "active_profile": "Korean_html2text",
        "text_extraction_method": "enhanced",
        "glossary_prompt_profiles": {"balanced_full": {"Alpha": "alpha balanced"},
                                     "minimal": {"Mini": "mini prompt"}},
        "active_glossary_prompt_profiles": {"minimal": "Mini"},
        "glossary_prompt_profile_defaults": {"balanced_full": "my balanced default"},
        "manual_glossary_prompt3": "my balanced default",
        "manual_glossary_prompt": "my balanced default",
        "unified_auto_glosary_prompt3": "mini prompt",
        "model": "gemini-2.5-flash",
    }


def _assert_on_device(store, translation: str = ON_DEVICE_BS) -> None:
    assert store.get("active_profile") == translation
    assert store.get("active_glossary_prompt_profiles") == {"balanced_full": ON_DEVICE, "minimal": ON_DEVICE}
    balanced = _on_device_glossary("balanced_full").strip()
    assert store.get("manual_glossary_prompt3") == balanced and store.get("manual_glossary_prompt") == balanced
    assert store.get("unified_auto_glosary_prompt3") == _on_device_glossary("minimal").strip()


# ---- the built-in profiles ------------------------------------------------------------------------------


@pytest.mark.parametrize("config", [{}, "existing"], ids=["fresh", "existing"])
def test_builtin_on_device_translation_profile_is_listed_and_protected(tmp_path, config):
    import prompt_profiles
    from glossarion_mobile.ui.screens.profiles import ProfileService

    config = _existing_config() if config == "existing" else config
    state = prompt_profiles.profile_state_from_config(config)
    assert state.prompt_profiles[ON_DEVICE_BS] == TRANSLATION_TEXT
    assert state.prompt_profiles[ON_DEVICE_H2T] == HTML2TEXT_TEXT
    assert ON_DEVICE not in state.prompt_profiles
    assert {ON_DEVICE_BS, ON_DEVICE_H2T} <= prompt_profiles.protected_profiles(state)
    assert prompt_profiles.extraction_method_for_profile(ON_DEVICE_BS) == "standard"
    assert prompt_profiles.extraction_method_for_profile(ON_DEVICE_H2T) == "enhanced"
    store = _store(tmp_path, config)
    assert {ON_DEVICE_BS, ON_DEVICE_H2T} <= set(ProfileService(store).listing().names)
    store._saver.close()


def test_mobile_on_device_names_match_the_shared_core():
    import prompt_profiles
    from glossarion_mobile.state import setting_writes as sw

    assert sw.ON_DEVICE_TRANSLATION_PROFILES == prompt_profiles.ON_DEVICE_TRANSLATION_PROFILES


@pytest.mark.parametrize("name,text", [(ON_DEVICE_BS, TRANSLATION_TEXT), (ON_DEVICE_H2T, HTML2TEXT_TEXT)])
def test_on_device_translation_profiles_cannot_be_deleted_and_reset_restores_them(tmp_path, name, text):
    import prompt_profiles

    state = prompt_profiles.profile_state_from_config({})
    state.prompt_profiles[name] = "edited"
    outcome = prompt_profiles.delete_or_reset_profile(state, name, confirm=lambda *a, **k: True)
    assert outcome.kind == "reset" and state.prompt_profiles[name] == text


@pytest.mark.parametrize("config", [{}, "existing", "edited"])
def test_builtin_on_device_glossary_profiles_are_seeded(config):
    import glossary_document as gd

    if config == "existing":
        config = _existing_config()
    elif config == "edited":
        config = {"glossary_prompt_profiles": {"balanced_full": {ON_DEVICE: "my own small prompt"}}}
    owner = types.SimpleNamespace(config=json.loads(json.dumps(config)))
    profiles, _active = gd.ensure_glossary_prompt_profiles(owner)
    for bucket in ("balanced_full", "minimal"):
        expected = "my own small prompt" if config.get("glossary_prompt_profiles", {}).get(bucket, {}).get(ON_DEVICE) \
            else _on_device_glossary(bucket)
        assert profiles[bucket][ON_DEVICE] == expected
    if "Alpha" in str(config):
        assert profiles["balanced_full"]["Alpha"] == "alpha balanced" and profiles["minimal"]["Mini"] == "mini prompt"


def test_on_device_glossary_prompts_ask_only_for_names():
    """Very short prompts a 0.8B model follows: name,{language} name,gender lines, {language} the only
    placeholder; the example rows are what both extractors read as characters."""
    import glossary_document as gd

    for text in gd.ON_DEVICE_GLOSSARY_PROMPTS.values():
        assert len(text) < 300 and "name,{language} name,gender" in text and "{language}" in text
        assert "{fields}" not in text and "{entries}" not in text and "description" not in text
        assert text.strip().splitlines()[-2:] == ["미라,Mira,female", "レン,Ren,male"]


@pytest.mark.parametrize("bucket", ["balanced_full", "minimal"])
def test_on_device_glossary_profile_cannot_be_deleted(bucket):
    import glossary_document as gd

    owner = types.SimpleNamespace(config={})
    row = gd.GlossaryPromptProfiles(owner, bucket, "")
    row.select(ON_DEVICE)
    asked = []
    box = row.delete(ON_DEVICE, confirm=lambda name: asked.append(name) or True)
    assert box and box[0] == "Built-in Profile" and not asked
    assert owner.config["glossary_prompt_profiles"][bucket][ON_DEVICE] == _on_device_glossary(bucket)
    assert row.selected_name() == ON_DEVICE


# ---- choosing a local/ model --------------------------------------------------------------------------


@pytest.mark.parametrize("config", [{}, "existing"], ids=["fresh", "existing"])
def test_local_model_selects_on_device_and_cloud_model_restores(tmp_path, config):
    from glossarion_mobile.state.setting_writes import ON_DEVICE_BEFORE_KEY, write_setting

    config = _existing_config() if config == "existing" else config
    store = _store(tmp_path, config)
    before = store.snapshot()

    translation = ON_DEVICE_H2T if config else ON_DEVICE_BS  # existing: enhanced (Korean_html2text)
    changed = write_setting(store, "model", LOCAL)
    assert "model" in changed and store.get("model") == LOCAL
    _assert_on_device(store, translation)
    assert store.has(ON_DEVICE_BEFORE_KEY)
    # another local model keeps the remembered profiles
    remembered = store.get(ON_DEVICE_BEFORE_KEY)
    write_setting(store, "model", "local/other")
    assert store.get(ON_DEVICE_BEFORE_KEY) == remembered
    _assert_on_device(store, translation)

    write_setting(store, "model", CLOUD)
    after = store.snapshot()
    assert not store.has(ON_DEVICE_BEFORE_KEY)
    assert after.get("active_profile") == before.get("active_profile")
    if before:
        assert after["text_extraction_method"] == "enhanced"
        assert after["active_glossary_prompt_profiles"] == {"minimal": "Mini"}
        for key in ("manual_glossary_prompt3", "manual_glossary_prompt", "unified_auto_glosary_prompt3"):
            assert after[key] == before[key], key
    else:
        assert after.get("active_glossary_prompt_profiles") == {}
        assert not after.get("manual_glossary_prompt3") and not after.get("unified_auto_glosary_prompt3")
    store._saver.close()


def test_switch_back_keeps_a_profile_the_user_picked(tmp_path):
    import glossary_document as gd
    from glossarion_mobile.state.setting_writes import PROFILE_KEY, write_setting

    store = _store(tmp_path, _existing_config())
    write_setting(store, "model", LOCAL)
    _assert_on_device(store, ON_DEVICE_H2T)
    # meanwhile: the translation profile and the Minimal glossary profile picked by hand
    write_setting(store, PROFILE_KEY, "My style")
    owner = types.SimpleNamespace(config=store.snapshot())
    row = gd.GlossaryPromptProfiles(owner, "minimal", owner.config.get("unified_auto_glosary_prompt3", ""))
    row.select(gd.GLOSSARY_PROMPT_DEFAULT_PROFILE)
    store.set_many({k: owner.config[k] for k in ("active_glossary_prompt_profiles", "unified_auto_glosary_prompt3")})

    write_setting(store, "model", CLOUD)
    assert store.get("active_profile") == "My style"
    # Minimal stays on Default (picked by hand); Balanced/Full was still On-device: restored to Default
    assert store.get("active_glossary_prompt_profiles") == {}
    assert store.get("manual_glossary_prompt3") == "my balanced default"
    assert store.get("unified_auto_glosary_prompt3") == row.text.strip()
    store._saver.close()


def test_restored_profile_that_was_deleted_meanwhile_stays_on_device(tmp_path):
    from glossarion_mobile.state.setting_writes import ON_DEVICE_BEFORE_KEY, write_setting

    store = _store(tmp_path, _existing_config())
    write_setting(store, "active_profile", "My style")
    write_setting(store, "model", LOCAL)
    profiles = store.get("prompt_profiles")
    profiles.pop("My style")
    store.set_many({"prompt_profiles": profiles})
    write_setting(store, "model", CLOUD)
    # "My style" is gone: the switch-back refuses an unknown name instead of storing it
    assert store.get("active_profile") == ON_DEVICE_H2T and not store.has(ON_DEVICE_BEFORE_KEY)
    store._saver.close()


@pytest.mark.parametrize("config", [{}, {"custom_glossary_fields": ["description", "age"]}],
                         ids=["fresh", "existing"])
def test_local_model_drops_the_glossary_description_column(tmp_path, config):
    """On-device: no extra glossary columns (a 0.8B model writes name, translated name, gender only);
    the "removed" flag keeps owner_state from adding "description" back. A cloud model restores them."""
    from glossarion_mobile.state.setting_writes import (CUSTOM_FIELDS_KEY, DESCRIPTION_REMOVED_KEY,
                                                        write_setting)

    store = _store(tmp_path, config)
    changed = write_setting(store, "model", LOCAL)
    assert CUSTOM_FIELDS_KEY in changed
    assert store.get(CUSTOM_FIELDS_KEY) == [] and store.get(DESCRIPTION_REMOVED_KEY) is True
    write_setting(store, "model", "local/other")  # still On-device: the remembered columns stay
    write_setting(store, "model", CLOUD)
    if config:
        assert store.get(CUSTOM_FIELDS_KEY) == ["description", "age"]
        assert not store.get(DESCRIPTION_REMOVED_KEY)
    else:  # nothing was stored: the desktop default (owner_state adds "description") again
        assert not store.has(CUSTOM_FIELDS_KEY) and not store.has(DESCRIPTION_REMOVED_KEY)
    store._saver.close()


def test_columns_added_while_on_device_stay(tmp_path):
    from glossarion_mobile.state.setting_writes import CUSTOM_FIELDS_KEY, write_setting

    store = _store(tmp_path, {})
    write_setting(store, "model", LOCAL)
    store.set_many({CUSTOM_FIELDS_KEY: ["age"]})
    write_setting(store, "model", CLOUD)
    assert store.get(CUSTOM_FIELDS_KEY) == ["age"]
    store._saver.close()


@pytest.mark.parametrize("config", [{}, {"translation_temperature": 0.3, "batch_size": 8, "max_output_tokens": 32768,
                                         "token_limit": 200000}],
                         ids=["fresh", "saved"])
def test_local_model_sets_request_size_and_cloud_restores_it(tmp_path, monkeypatch, config):
    """On-device: temperature 0.4, batch size 1, output limit half the context; a cloud model gets the
    saved values back (or the defaults when nothing was saved) unless the user changed them meanwhile."""
    from glossarion_mobile.state import setting_writes as sw

    monkeypatch.setattr(sw, "local_output_limit", lambda model: 4096 if str(model).startswith("local/") else None)
    monkeypatch.setattr(sw, "local_context_limit", lambda model: 8192 if str(model).startswith("local/") else None)
    store = _store(tmp_path, config)
    sw.write_setting(store, "model", LOCAL)
    assert store.get("translation_temperature") == 0.4 and store.get("batch_size") == 1
    assert store.get("max_output_tokens") == 4096 and store.get("token_limit") == 8192
    assert "review_system_prompt" not in sw.ON_DEVICE_SETTINGS  # the On-device review profile instead
    sw.write_setting(store, "model", CLOUD)
    for key in ("translation_temperature", "batch_size", "max_output_tokens", "token_limit"):
        if key in config:
            assert store.get(key) == config[key], key
        else:  # nothing was saved: the default again
            assert not store.has(key), key
    # a value changed while On-device stays
    sw.write_setting(store, "model", LOCAL)
    store.set_many({"batch_size": 3})
    sw.write_setting(store, "model", CLOUD)
    assert store.get("batch_size") == 3
    assert store.get("translation_temperature") == config.get("translation_temperature", None) or (
        not config and not store.has("translation_temperature"))
    store._saver.close()



@pytest.mark.parametrize("config", [{}, {"manga_ocr_provider": "rapidocr"}], ids=["fresh", "saved"])
def test_local_model_reads_manga_with_hayai_and_cloud_restores_the_ocr_provider(tmp_path, config):
    """owner 2026-10-21: an on-device model defaults the Manga translator to Hayai OCR; a cloud model gets
    the saved provider back (or the default when nothing was saved)."""
    from glossarion_mobile.state import setting_writes as sw

    store = _store(tmp_path, config)
    sw.write_setting(store, "model", LOCAL)
    assert store.get("manga_ocr_provider") == sw.ON_DEVICE_OCR_PROVIDER == "hayai-ocr"
    sw.write_setting(store, "model", CLOUD)
    if config:
        assert store.get("manga_ocr_provider") == "rapidocr"
    else:
        assert not store.has("manga_ocr_provider")
    store._saver.close()

# ---- the review generator's On-device prompt profile --------------------------------------------------------


def _review_config(kind: str) -> dict:
    if kind == "custom":  # a pre-profile custom prompt pair (both prompts)
        return {"review_system_prompt": "My review in {target_lang}", "review_final_prompt": "My final synthesis"}
    if kind == "profile":  # profiles already, a user profile active
        return {"review_prompt_profiles": {"Mine": {"system": "Mine {target_lang}", "final": "Mine final"}},
                "active_review_prompt_profile": "Mine"}
    return {}


@pytest.mark.parametrize("kind,restored", [("fresh", "Default"), ("custom", "My review prompt"), ("profile", "Mine")])
def test_local_model_selects_the_on_device_review_profile_and_cloud_restores_it(tmp_path, kind, restored):
    """Both review prompts follow the model: On-device's short review and final prompts for a local/ model, the
    profile from before (a migrated custom pair included) for a cloud model again."""
    import review_generator as rg
    from glossarion_mobile.state.setting_writes import ON_DEVICE_BEFORE_KEY, write_setting

    store = _store(tmp_path, _review_config(kind))
    changed = write_setting(store, "model", LOCAL)
    assert "active_review_prompt_profile" in changed
    assert store.get("active_review_prompt_profile") == ON_DEVICE
    assert store.get("review_system_prompt") == rg.ON_DEVICE_REVIEW_PROMPT
    assert store.get("review_final_prompt") == rg.ON_DEVICE_FINAL_REVIEW_PROMPT
    assert store.get(ON_DEVICE_BEFORE_KEY)["review"] == restored
    profiles = store.get("review_prompt_profiles")
    assert list(profiles)[:2] == ["Default", ON_DEVICE] and restored in profiles
    write_setting(store, "model", "local/other")  # still On-device: the remembered profile stays
    assert store.get(ON_DEVICE_BEFORE_KEY)["review"] == restored
    write_setting(store, "model", CLOUD)
    assert store.get("active_review_prompt_profile") == restored
    texts = store.get("review_prompt_profiles")[restored]
    assert store.get("review_system_prompt") == texts["system"].strip()
    assert store.get("review_final_prompt") == texts["final"].strip()
    if kind == "custom":  # the custom pair was migrated, final prompt included
        assert texts == {"system": "My review in {target_lang}", "final": "My final synthesis"}
    elif kind == "fresh":
        assert texts == {"system": rg.DEFAULT_REVIEW_PROMPT, "final": rg.DEFAULT_FINAL_REVIEW_PROMPT}
    store._saver.close()


def test_review_profile_picked_while_on_device_stays(tmp_path):
    import review_generator as rg
    from glossarion_mobile.state.setting_writes import write_setting

    store = _store(tmp_path, _review_config("profile"))
    write_setting(store, "model", LOCAL)
    config = store.snapshot()
    assert rg.set_active_review_prompt_profile(config, "Default")  # the review tool's picker
    store.set_many({key: config[key] for key in ("active_review_prompt_profile", "review_system_prompt",
                                                 "review_final_prompt")})
    write_setting(store, "model", CLOUD)
    assert store.get("active_review_prompt_profile") == "Default"
    assert store.get("review_final_prompt") == rg.DEFAULT_FINAL_REVIEW_PROMPT.strip()
    store._saver.close()


@pytest.mark.parametrize("active", ["Default", ON_DEVICE, "Mine"])
def test_review_job_sends_the_active_profiles_prompts(active):
    """job_kinds.review: the active profile's review prompt and final prompt (not the mirrors)."""
    import review_generator as rg
    from glossarion_mobile.job_kinds.review import review_prompts

    config = {"review_prompt_profiles": {"Mine": {"system": "Mine {target_lang}", "final": "Mine final"}},
              "active_review_prompt_profile": active,
              "review_system_prompt": "stale mirror", "review_final_prompt": "stale final"}
    expected = {"Default": (rg.DEFAULT_REVIEW_PROMPT, rg.DEFAULT_FINAL_REVIEW_PROMPT),
                ON_DEVICE: (rg.ON_DEVICE_REVIEW_PROMPT, rg.ON_DEVICE_FINAL_REVIEW_PROMPT),
                "Mine": ("Mine {target_lang}", "Mine final")}[active]
    before = json.loads(json.dumps(config))
    assert review_prompts(rg, config) == expected
    assert config == before  # the job's config snapshot is not changed


# ---- the text extraction method and the On-device translation profiles ----------------------------------


@pytest.mark.parametrize("method,expected", [(None, ON_DEVICE_BS), ("standard", ON_DEVICE_BS),
                                             ("enhanced", ON_DEVICE_H2T)])
def test_local_model_selects_the_on_device_profile_of_the_extraction_method(tmp_path, method, expected):
    from glossarion_mobile.state.setting_writes import write_setting

    store = _store(tmp_path, {"text_extraction_method": method} if method else {})
    write_setting(store, "model", LOCAL)
    assert store.get("active_profile") == expected
    assert store.get("text_extraction_method", "standard") == (method or "standard")
    write_setting(store, "model", CLOUD)
    assert not store.has("active_profile")  # nothing was stored before: the first profile again
    assert store.get("text_extraction_method", "standard") == (method or "standard")
    store._saver.close()


def test_switching_the_extraction_method_switches_the_active_on_device_profile(tmp_path):
    from glossarion_mobile.state.setting_writes import EXTRACTION_KEY, ON_DEVICE_BEFORE_KEY, write_setting

    store = _store(tmp_path, {"active_profile": "My style", "prompt_profiles": _existing_config()["prompt_profiles"]})
    write_setting(store, "model", LOCAL)
    assert store.get("active_profile") == ON_DEVICE_BS
    write_setting(store, EXTRACTION_KEY, "enhanced")
    assert store.get("active_profile") == ON_DEVICE_H2T and store.get(EXTRACTION_KEY) == "enhanced"
    write_setting(store, EXTRACTION_KEY, "standard")
    assert store.get("active_profile") == ON_DEVICE_BS and store.get(EXTRACTION_KEY) == "standard"
    write_setting(store, EXTRACTION_KEY, "enhanced")
    # the remembered profile still comes back for a cloud model (On-device_html2text is On-device too)
    assert store.get(ON_DEVICE_BEFORE_KEY)["active_profile"] == "My style"
    write_setting(store, "model", CLOUD)
    assert store.get("active_profile") == "My style" and store.get(EXTRACTION_KEY) == "enhanced"
    store._saver.close()


def test_extraction_method_leaves_other_profiles_alone_and_a_suffixless_profile_keeps_it(tmp_path):
    """Universal (no _BeautifulSoup / _html2text suffix): the method is the user's choice alone - selecting it
    never resets the stored method, and changing the method never changes the profile."""
    from glossarion_mobile.state.setting_writes import EXTRACTION_KEY, PROFILE_KEY, write_setting

    store = _store(tmp_path, {"model": CLOUD})
    write_setting(store, EXTRACTION_KEY, "enhanced")
    assert store.get(EXTRACTION_KEY) == "enhanced" and not store.has(PROFILE_KEY)
    write_setting(store, PROFILE_KEY, "Korean_BeautifulSoup")
    assert store.get(EXTRACTION_KEY) == "standard"  # a suffixed profile still switches it (desktop)
    write_setting(store, EXTRACTION_KEY, "enhanced")
    assert store.get(PROFILE_KEY) == "Korean_BeautifulSoup"  # not an On-device profile: stays
    write_setting(store, PROFILE_KEY, "Universal")
    assert store.get(PROFILE_KEY) == "Universal" and store.get(EXTRACTION_KEY) == "enhanced"
    store._saver.close()


def test_standard_replaces_a_stale_legacy_enhanced_extraction_mode(tmp_path):
    """``extraction_mode`` 'enhanced' wins over ``text_extraction_method`` at run time: choosing
    BeautifulSoup writes the filtering level there, as the desktop save does."""
    from glossarion_mobile.state.setting_writes import EXTRACTION_KEY, write_setting

    store = _store(tmp_path, {"extraction_mode": "enhanced", "text_extraction_method": "enhanced",
                              "file_filtering_level": "comprehensive"})
    write_setting(store, EXTRACTION_KEY, "standard")
    assert store.get("extraction_mode") == "comprehensive" and store.get(EXTRACTION_KEY) == "standard"
    store._saver.close()


def _env_value(config: dict, tmp_path, name: str) -> str:
    from glossarion_mobile.ui.screens import env_preview as ep

    result = ep.build_env_preview(config, input_path=ep.preview_input_path("epub", str(tmp_path)),
                                  lock=__import__("threading").Lock())
    assert result.ok, result.error
    return {row.key: row.value for row in result.rows}[name]


@pytest.mark.parametrize("method,html2text", [("enhanced", "1"), ("standard", "0")])
def test_cloud_model_universal_runs_with_the_chosen_extraction(tmp_path, method, html2text):
    """Welcome › Text extraction with a cloud model and the default Universal profile: the job's config
    (config.json + the chat's overrides, as ``JobService`` builds it) runs with that method."""
    from glossarion_mobile.state.setting_writes import EXTRACTION_KEY, write_setting
    from glossarion_mobile.ui.chat.run_request import config_overrides

    store = _store(tmp_path, {"model": CLOUD})
    write_setting(store, EXTRACTION_KEY, method)
    config = store.snapshot()
    config.update(config_overrides({"profile": "Universal", "model": CLOUD}))
    assert "text_extraction_method" not in config_overrides({"profile": "Universal"})
    assert config["text_extraction_method"] == method
    assert _env_value(config, tmp_path, "TEXT_EXTRACTION_METHOD") == method
    assert _env_value(config, tmp_path, "USE_HTML2TEXT") == html2text
    store._saver.close()


@pytest.mark.parametrize("model,method,profile", [(LOCAL, "enhanced", ON_DEVICE_H2T), (LOCAL, "standard", ON_DEVICE_BS),
                                                   (CLOUD, "enhanced", None)])
def test_welcome_text_extraction_is_saved_and_selects_the_on_device_profile(tmp_path, model, method, profile):
    """Welcome › Text extraction: Finish saves ``text_extraction_method`` for every model (written like the
    app's Finish: ``write_setting``), and an on-device model's translation profile follows it."""
    pytest.importorskip("flet")
    from glossarion_mobile.state.setting_writes import EXTRACTION_KEY, write_setting
    from glossarion_mobile.ui.router import parse_route
    from glossarion_mobile.ui.screens.welcome import WelcomeScreen

    store = _store(tmp_path, {})
    write_setting(store, "model", model)  # step 1 picks the model
    screen = WelcomeScreen(parse_route("/welcome"), current_model=lambda: store.get("model", ""),
                           config_get=lambda key, default=None: store.get(key, default))
    screen.get_body()
    screen.go("language_glossary")
    assert screen.shown_extraction_method() == "standard"  # BeautifulSoup (recommended) by default
    screen.flow.select_extraction(method)
    updates = screen.finish()
    assert updates[EXTRACTION_KEY] == method
    write_setting(store, EXTRACTION_KEY, updates[EXTRACTION_KEY])
    assert store.get(EXTRACTION_KEY) == method
    if profile:
        assert store.get("active_profile") == profile
    else:
        assert not store.has("active_profile")
    store._saver.close()


@pytest.mark.parametrize("config", [{}, {"auto_glossary_mode": "full", "enable_auto_glossary": True}],
                         ids=["fresh", "saved"])
def test_small_on_device_model_defaults_to_no_glossary_and_cloud_restores(tmp_path, config):
    """Owner 2026-10-19: Auto glossary is NO GLOSSARY on an on-device model, except the 9B one."""
    from glossarion_mobile.state import setting_writes as sw

    assert not sw.is_large_on_device_model(LOCAL) and sw.is_large_on_device_model("local/Qwen3.5-9B-Q4_K_M")
    assert not sw.is_large_on_device_model("authgpt/gpt-9b") and not sw.is_large_on_device_model(CLOUD)
    store = _store(tmp_path, config)
    sw.write_setting(store, "model", LOCAL)
    assert store.get("auto_glossary_mode") == "no_glossary" and store.get("enable_auto_glossary") is False
    sw.write_setting(store, "model", CLOUD)
    for key in ("auto_glossary_mode", "enable_auto_glossary"):
        if key in config:
            assert store.get(key) == config[key], key
    if not config:  # nothing was saved: Balanced (mobile hotfix 3; an unset mode ran as "Off")
        assert store.get("auto_glossary_mode") == sw.GLOSSARY_MODE_AFTER_ON_DEVICE == "balanced"
        assert store.get("enable_auto_glossary") is True
    # a mode picked while on-device stays
    sw.write_setting(store, "model", LOCAL)
    store.set_many({"auto_glossary_mode": "minimal", "enable_auto_glossary": True})
    sw.write_setting(store, "model", CLOUD)
    assert store.get("auto_glossary_mode") == "minimal"
    store._saver.close()


def test_the_9b_on_device_model_keeps_the_glossary_mode(tmp_path):
    from glossarion_mobile.state import setting_writes as sw

    store = _store(tmp_path, {"auto_glossary_mode": "balanced", "enable_auto_glossary": True})
    sw.write_setting(store, "model", "local/Qwen3.5-9B-Q4_K_M")
    assert store.get("auto_glossary_mode") == "balanced" and store.get("enable_auto_glossary") is True
    store._saver.close()
