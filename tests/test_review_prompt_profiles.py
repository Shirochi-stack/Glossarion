"""Review generator prompt profiles (review_generator.ReviewPromptProfiles, desktop Review dialog).

Each profile holds both prompts of a review: "system" (the review prompt, also every chunk's prompt) and
"final" (the Full Review Mode synthesis prompt). Built-ins "Default" and "On-device" are seeded, protected
and resettable; a pre-profile custom review_system_prompt / review_final_prompt becomes "My review prompt";
review_system_prompt / review_final_prompt stay as mirrors of the active profile, and the runs use the
active profile's texts.
"""
import os
import sys
import types

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import review_generator as rg  # noqa: E402

DEFAULT = {"system": rg.DEFAULT_REVIEW_PROMPT, "final": rg.DEFAULT_FINAL_REVIEW_PROMPT}
ON_DEVICE = {"system": rg.ON_DEVICE_REVIEW_PROMPT, "final": rg.ON_DEVICE_FINAL_REVIEW_PROMPT}


def _owner(config=None):
    return types.SimpleNamespace(config=dict(config or {}))


def _profiles(config=None, persist=None):
    owner = _owner(config)
    return owner, rg.ReviewPromptProfiles(owner, persist=persist)


# ---- built-ins ---------------------------------------------------------------------------------------------


def test_builtins_are_seeded_first_with_both_prompts():
    config = {}
    profiles = rg.ensure_review_prompt_profiles(config)
    assert list(profiles) == ["Default", "On-device"]
    assert profiles["Default"] == DEFAULT and profiles["On-device"] == ON_DEVICE
    assert config["active_review_prompt_profile"] == "Default"
    # a fresh config: the mirrors hold the Default texts
    assert config["review_system_prompt"] == rg.DEFAULT_REVIEW_PROMPT.strip()
    assert config["review_final_prompt"] == rg.DEFAULT_FINAL_REVIEW_PROMPT.strip()
    assert "{target_lang}" in rg.ON_DEVICE_REVIEW_PROMPT and "{target_lang}" in rg.ON_DEVICE_FINAL_REVIEW_PROMPT
    assert rg.is_builtin_review_prompt_profile("default") and rg.is_builtin_review_prompt_profile("On-device")
    assert not rg.is_builtin_review_prompt_profile("Mine")


def test_edited_builtin_survives():
    config = {"review_prompt_profiles": {"On-device": {"system": "edited", "final": "edited final"}},
              "active_review_prompt_profile": "Default"}
    profiles = rg.ensure_review_prompt_profiles(config)
    assert profiles["On-device"] == {"system": "edited", "final": "edited final"}
    assert profiles["Default"] == DEFAULT


def test_builtins_cannot_be_deleted_and_reset_restores_both_prompts():
    owner, profiles = _profiles()
    assert profiles.delete("Default") == ("Default Profile", "The Default review prompt profile cannot be deleted.")
    assert profiles.delete("On-device")[0] == "Built-in Profile"
    profiles.select("On-device")
    profiles.stage("On-device", {"system": "my short prompt", "final": "my short final"})
    assert owner.config["review_prompt_profiles"]["On-device"] == {"system": "my short prompt",
                                                                    "final": "my short final"}
    assert owner.config["review_final_prompt"] == "my short final"
    assert profiles.reset("On-device") is None
    assert owner.config["review_prompt_profiles"]["On-device"] == ON_DEVICE
    assert profiles.texts == ON_DEVICE
    assert owner.config["review_system_prompt"] == rg.ON_DEVICE_REVIEW_PROMPT
    assert owner.config["review_final_prompt"] == rg.ON_DEVICE_FINAL_REVIEW_PROMPT
    # Default resets too, while another profile is selected
    profiles.select("Default")
    profiles.stage("Default", {"system": "changed", "final": "changed final"})
    profiles.select("On-device")
    assert profiles.reset("Default") is None and profiles.profile_texts("Default") == DEFAULT
    assert profiles.selected_name() == "On-device"
    profiles.new()
    assert profiles.reset()[0] == "Not a Built-in Profile"


# ---- migration -----------------------------------------------------------------------------------------------


@pytest.mark.parametrize("system,final,expected", [
    ("Custom review {target_lang}", "", {"system": "Custom review {target_lang}", "final": rg.DEFAULT_FINAL_REVIEW_PROMPT}),
    ("", "Custom final", {"system": rg.DEFAULT_REVIEW_PROMPT, "final": "Custom final"}),
    ("Custom review", "Custom final", {"system": "Custom review", "final": "Custom final"}),
], ids=["system-only", "final-only", "both"])
def test_custom_pre_profile_prompts_become_my_review_prompt(system, final, expected):
    config = {"review_system_prompt": system, "review_final_prompt": final}
    owner = types.SimpleNamespace(config=config)
    profiles = rg.ensure_review_prompt_profiles(config, owner)
    assert list(profiles) == ["Default", "On-device", "My review prompt"]
    assert profiles["My review prompt"] == expected
    assert config["active_review_prompt_profile"] == "My review prompt"
    assert config["review_system_prompt"] == expected["system"].strip()
    assert config["review_final_prompt"] == expected["final"].strip()
    assert owner.review_final_prompt_var == expected["final"].strip()
    # migrated once: later calls change nothing
    before = {k: (dict(v) if isinstance(v, dict) else v) for k, v in config.items()}
    rg.ensure_review_prompt_profiles(config)
    assert config == before
    assert rg.active_review_prompts(config) == (expected["system"].strip(), expected["final"].strip())


@pytest.mark.parametrize("system,final,expected", [
    ("", "", "Default"),
    (rg.DEFAULT_REVIEW_PROMPT, rg.DEFAULT_FINAL_REVIEW_PROMPT + "\n", "Default"),
    (rg.ON_DEVICE_REVIEW_PROMPT, rg.ON_DEVICE_FINAL_REVIEW_PROMPT, "On-device"),
])
def test_prompts_equal_to_a_builtin_select_it_without_a_new_profile(system, final, expected):
    config = {"review_system_prompt": system, "review_final_prompt": final}
    profiles = rg.ensure_review_prompt_profiles(config)
    assert list(profiles) == ["Default", "On-device"]
    assert rg.active_review_prompt_profile(config) == expected


def test_migrated_name_is_numbered_when_taken():
    config = {"review_prompt_profiles": {"My review prompt": {"system": "a", "final": "b"}}}
    name = rg.review_profile_for_prompts(config, "other", "texts")
    assert name == "My review prompt 2"
    assert rg.review_profile_for_prompts(config, "a", "b") == "My review prompt"  # an existing one is reused


# ---- the profiles object (desktop row / mobile bar) ----------------------------------------------------------


def test_new_save_rename_delete_and_copy_of_a_builtin():
    saves = []
    owner, profiles = _profiles(persist=lambda: saves.append(1))
    assert profiles.names == ["Default", "On-device"] and profiles.selected_name() == "Default"
    assert profiles.new() is None and profiles.selected_name() == "New Profile #1"
    assert owner.config["review_prompt_profiles"]["New Profile #1"] == {"system": "", "final": ""}
    # an empty profile runs with the shipped defaults (as an empty prompt always did)
    assert rg.active_review_prompts(owner.config) == (rg.DEFAULT_REVIEW_PROMPT, rg.DEFAULT_FINAL_REVIEW_PROMPT)
    assert profiles.save("Mine", {"system": "My review", "final": "My final"}) is None  # rename
    assert list(owner.config["review_prompt_profiles"]) == ["Default", "On-device", "Mine"]
    assert owner.config["active_review_prompt_profile"] == "Mine"
    assert (owner.review_system_prompt_var, owner.review_final_prompt_var) == ("My review", "My final")
    assert profiles.save("On-device", {"system": "x", "final": "y"})[0] == "Profile Name Exists"
    # a built-in saved under another name is copied; the built-in keeps its name
    profiles.select("On-device")
    assert profiles.save("Small", dict(ON_DEVICE)) is None
    assert owner.config["review_prompt_profiles"]["On-device"] == ON_DEVICE
    assert owner.config["review_prompt_profiles"]["Small"] == ON_DEVICE
    assert profiles.selected_name() == "Small"
    assert profiles.delete("Mine", confirm=lambda name: False) is None
    assert "Mine" in owner.config["review_prompt_profiles"]
    assert profiles.delete("Mine", confirm=lambda name: True) is None
    assert "Mine" not in owner.config["review_prompt_profiles"]
    assert len(saves) >= 4


def test_failed_save_rolls_back():
    owner, profiles = _profiles(persist=lambda: False)
    before = dict(owner.config)
    assert profiles.new()[0] == "Save Failed"
    assert owner.config == before and profiles.selected_name() == "Default"


# ---- the runs ---------------------------------------------------------------------------------------------------


def test_run_params_use_the_active_profiles_review_and_final_prompts():
    config = {"review_prompt_profiles": {"Mine": {"system": "Review in {target_lang}", "final": "Final in {target_lang}"}},
              "active_review_prompt_profile": "Mine",
              "review_system_prompt": "stale", "review_final_prompt": "stale"}
    prompt, final = rg.active_review_prompts(config)
    assert (prompt, final) == ("Review in {target_lang}", "Final in {target_lang}")
    assert config["review_system_prompt"] == "stale"  # the run does not write the config
    gui = types.SimpleNamespace(config=config, model_var="m", lang_var="Korean")
    params = rg.review_run_params(gui, prompt, False, final)
    assert params.system_prompt == "Review in Korean" and params.final_review_prompt == "Final in Korean"
    config["active_review_prompt_profile"] = "On-device"
    assert rg.active_review_prompts(config) == (rg.ON_DEVICE_REVIEW_PROMPT, rg.ON_DEVICE_FINAL_REVIEW_PROMPT)


# ---- the desktop dialog -------------------------------------------------------------------------------------------


@pytest.fixture
def dialog_factory(tmp_path, monkeypatch):
    pytest.importorskip("PySide6")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.setenv("OUTPUT_DIRECTORY", str(tmp_path / "out"))
    from PySide6.QtWidgets import QApplication
    import review_dialog

    monkeypatch.setattr(review_dialog, "count_epub_tokens", lambda _path: 3)
    monkeypatch.setattr(review_dialog, "count_review_tokens", lambda paths: 3)
    book = tmp_path / "Novel.txt"
    book.write_text("text", encoding="utf-8")
    app = QApplication.instance() or QApplication([])
    made = []

    class FakeGui:
        def __init__(self, config):
            self.selected_files = [str(book)]
            self.config = config
            self.base_dir = str(tmp_path)
            self.saves = 0

        def save_config(self, show_message=False):
            self.saves += 1

    def make(config):
        gui = FakeGui(config)
        dialog = review_dialog.ReviewDialog(None, gui, str(book))
        made.append(dialog)
        return gui, dialog

    yield make
    for dialog in made:
        dialog.deleteLater()
    app.processEvents()


def _combo_items(combo):
    return [combo.itemText(i) for i in range(combo.count())]


def test_dialog_opens_on_the_migrated_profile_with_both_prompts(dialog_factory):
    gui, dialog = dialog_factory({"review_system_prompt": "Mine {target_lang}", "review_final_prompt": "Mine final"})
    combo = dialog._review_profile_combo
    assert _combo_items(combo) == ["Default", "On-device", "My review prompt"]
    assert combo.currentText() == "My review prompt"
    assert dialog.prompt_edit.toPlainText() == "Mine {target_lang}"
    assert dialog._final_review_prompt == "Mine final"
    assert dialog._review_profile_buttons["delete"].text() == "🗑 Delete Profile"
    # the dropdown uses the Halgakos arrow like the dialog's other combo
    assert "Halgakos.ico" in combo.styleSheet() and "down-arrow" in combo.styleSheet()
    assert dialog._review_run_prompts() == ("Mine {target_lang}", "Mine final")


def test_dialog_select_edit_new_rename_delete_and_reset(dialog_factory, monkeypatch):
    gui, dialog = dialog_factory({})
    boxes = []
    monkeypatch.setattr(dialog, "_review_profile_box", lambda box: boxes.append(box) or box)
    monkeypatch.setattr(dialog, "_confirm_review_profile", lambda title, text: True)
    combo = dialog._review_profile_combo
    assert combo.currentText() == "Default"
    assert dialog.prompt_edit.toPlainText() == rg.DEFAULT_REVIEW_PROMPT
    assert dialog._review_profile_buttons["delete"].text() == "↺ Reset Profile"

    # picking On-device shows both of its prompts and saves the choice
    saves = gui.saves
    combo.setCurrentIndex(combo.findText("On-device"))
    assert dialog.prompt_edit.toPlainText() == rg.ON_DEVICE_REVIEW_PROMPT
    assert dialog._final_review_prompt == rg.ON_DEVICE_FINAL_REVIEW_PROMPT
    assert gui.config["active_review_prompt_profile"] == "On-device" and gui.saves > saves
    assert gui.review_system_prompt_var == rg.ON_DEVICE_REVIEW_PROMPT

    # the final prompt dialog edits the selected profile's final prompt
    dialog._open_final_prompt_dialog()
    assert "On-device" in dialog._final_prompt_dialog.windowTitle()
    assert dialog._final_prompt_editor.toPlainText() == rg.ON_DEVICE_FINAL_REVIEW_PROMPT
    dialog._final_review_prompt = "Edited final"
    dialog._stage_review_prompts()
    dialog.prompt_edit.setPlainText("Edited review")
    assert gui.config["review_prompt_profiles"]["On-device"] == {"system": "Edited review", "final": "Edited final"}
    assert dialog._review_run_prompts() == ("Edited review", "Edited final")
    assert dialog._review_profile_shipped_final() == rg.ON_DEVICE_FINAL_REVIEW_PROMPT
    dialog._final_prompt_dialog.close()

    # Reset: both shipped prompts back
    dialog._delete_or_reset_review_profile()
    assert gui.config["review_prompt_profiles"]["On-device"] == ON_DEVICE
    assert dialog.prompt_edit.toPlainText() == rg.ON_DEVICE_REVIEW_PROMPT
    assert dialog._final_review_prompt == rg.ON_DEVICE_FINAL_REVIEW_PROMPT

    # + New Profile, edit, rename by editing the name and saving
    dialog._new_review_profile()
    assert combo.currentText() == "New Profile #1" and dialog.prompt_edit.toPlainText() == ""
    assert dialog._review_profile_buttons["delete"].text() == "🗑 Delete Profile"
    dialog.prompt_edit.setPlainText("Fresh review")
    combo.setEditText("Fresh")
    dialog._save_review_profile()
    assert _combo_items(combo) == ["Default", "On-device", "Fresh"] and combo.currentText() == "Fresh"
    assert gui.config["review_prompt_profiles"]["Fresh"]["system"] == "Fresh review"
    assert gui.config["active_review_prompt_profile"] == "Fresh"
    combo.setEditText("Default")
    dialog._save_review_profile()
    assert boxes[-1][0] == "Default Profile"

    # Delete the user profile: the first remaining profile is selected
    combo.setEditText("Fresh")
    dialog._delete_or_reset_review_profile()
    assert "Fresh" not in gui.config["review_prompt_profiles"]
    assert _combo_items(combo) == ["Default", "On-device"]
    assert combo.currentText() == "On-device"


def test_dialog_start_sends_the_active_profiles_prompts(dialog_factory, monkeypatch):
    import review_dialog

    gui, dialog = dialog_factory({"review_prompt_profiles": {"Mine": {"system": "S {target_lang}", "final": "F"}},
                                  "active_review_prompt_profile": "Mine"})
    seen = []

    def fake_params(gui_, prompt, spoiler, final):
        seen.append((prompt, final))
        raise RuntimeError("stop here")

    monkeypatch.setattr(review_dialog, "review_run_params", fake_params)
    dialog.log_field.clear()
    with pytest.raises(RuntimeError, match="stop here"):
        dialog._on_start_review()
    assert seen == [("S {target_lang}", "F")]
