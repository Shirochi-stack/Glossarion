"""Built-in "On-device" prompt profiles for small on-device models (mobile llama.cpp, ``local/<stem>``).

* the translation profile "On-device" is a desktop built-in (always included, protected), so a fresh
  and an existing config both list it;
* the Balanced/Full and Minimal glossary buckets get an "On-device" profile (``glossary_document``),
  seeded without overwriting an edited text and protected from deletion like Default;
* choosing a ``local/`` model (``setting_writes.write_setting(store, "model", ...)``, the path every
  global model writer takes) selects the three On-device profiles and the current glossary prompt keys
  runs read; choosing a model that is not ``local/`` again restores the remembered profiles where
  On-device is still the active one.

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
TRANSLATION_TEXT = "You must only output {target_lang} translations."


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
    profiles = {name: text for name, text in defaults.items() if name != ON_DEVICE}
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


def _assert_on_device(store) -> None:
    assert store.get("active_profile") == ON_DEVICE
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
    assert state.prompt_profiles[ON_DEVICE] == TRANSLATION_TEXT
    assert ON_DEVICE in prompt_profiles.protected_profiles(state)
    store = _store(tmp_path, config)
    assert ON_DEVICE in ProfileService(store).listing().names
    store._saver.close()


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
    """Very short prompts a 0.8B model follows: raw_name,translated_name,gender lines, {language} the only
    placeholder; the example row is what both extractors read as a character."""
    import glossary_document as gd

    for text in gd.ON_DEVICE_GLOSSARY_PROMPTS.values():
        assert len(text) < 300 and "raw_name,translated_name,gender" in text and "{language}" in text
        assert "{fields}" not in text and "{entries}" not in text and "description" not in text
        example = text.strip().splitlines()[-1]
        assert example == "田中太郎,Tanaka Taro,male"


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

    changed = write_setting(store, "model", LOCAL)
    assert "model" in changed and store.get("model") == LOCAL
    _assert_on_device(store)
    assert store.has(ON_DEVICE_BEFORE_KEY)
    # another local model keeps the remembered profiles
    remembered = store.get(ON_DEVICE_BEFORE_KEY)
    write_setting(store, "model", "local/other")
    assert store.get(ON_DEVICE_BEFORE_KEY) == remembered
    _assert_on_device(store)

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
    _assert_on_device(store)
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
    assert store.get("active_profile") == ON_DEVICE and not store.has(ON_DEVICE_BEFORE_KEY)
    store._saver.close()
