"""Custom image edit prompt profiles and blank user-drawn boxes (owner request, 2026-11-07).

* P: ``manga_env.ImageEditPromptProfiles`` (the review / glossary refinement profile engine,
  ``prompt_profiles.PromptSetProfiles``): Default is the prompt used before profiles (a saved custom
  prompt stays in it), the built-ins Translate SFX and Translate SFX + background text (protected,
  ↺ Reset), the user's own profiles, persistence and rollback.
* D: the manga tab's Custom Image Edit Prompt dialog shows the review dialog's shared profile row
  (offscreen Qt) and the selected profile's prompts become the ones the runs read.
* R: the request that reaches the image edit model carries the selected profile's prompt with
  ``{target_lang}`` (the translation prompt profiles' placeholder) replaced by the configured output
  language; a translate prompt sends and keeps the whole crop and skips the leftover-text check.
* B: a box the user drew with no OCR text is still in the inpaint / image edit mask (Translate),
  detector boxes without OCR text still are not.

Synthetic pages, isolated configs (the owner's src/config.json is never read or written), sockets
blocked, fake keys only.
"""
import base64
import os
import socket
import sys
import threading
import time
import types
from pathlib import Path
from queue import Queue

import pytest

TESTS_DIR = Path(__file__).resolve().parent
SRC_DIR = TESTS_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
os.environ.setdefault("GLOSSARION_HTTP_LOG", "0")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import manga_env  # noqa: E402

SFX = "Translate SFX"
SFX_BG = "Translate SFX + background text"
DEFAULT_PROMPT = manga_env.default_custom_image_edit_system_prompt()


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    def refuse(*_args, **_kwargs):
        raise OSError("network access is blocked in this test")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)
    for name in ("OUTPUT_LANGUAGE", "GLOSSARY_TARGET_LANGUAGE", "TARGET_LANGUAGE", "TARGET_LANG",
                 "QWEN_IMAGE_EDIT_INPAINT_PROMPT", "USE_INPAINTER_KEYS", "GRACEFUL_STOP", "TRANSLATION_CANCELLED"):
        monkeypatch.delenv(name, raising=False)


def _owner(config=None):
    return types.SimpleNamespace(config=dict(config or {}))


# ---------------------------------------------------------------------------
# P: the profiles
# ---------------------------------------------------------------------------

def test_fresh_config_has_default_and_the_two_builtins():
    owner = _owner()
    profiles = manga_env.ImageEditPromptProfiles(owner)
    assert profiles.names == ["Default", SFX, SFX_BG]
    assert profiles.selected_name() == "Default"
    assert profiles.texts == {"system": DEFAULT_PROMPT, "user": ""}
    assert owner.config["custom_image_edit_system_prompt"] == DEFAULT_PROMPT
    assert all(profiles.is_builtin(name) for name in profiles.names)
    for name in (SFX, SFX_BG):
        assert "{target_lang}" in profiles.profile_texts(name)["system"]
    # the second profile covers the text the detector misses, long vertical titles included
    background = profiles.profile_texts(SFX_BG)["system"].lower()
    for word in ("sound effect", "sign", "letters", "guild card", "papers", "poster", "label", "screen",
                 "long vertical titles", "vertical line"):
        assert word in background, word
    # neither new prompt looks like the old translate default the tab replaces on load
    for name in (SFX, SFX_BG):
        text = " ".join(profiles.profile_texts(name)["system"].split()).lower()
        assert "replacing all foreign-language text" not in text
        assert "if the image has no translatable text" not in text


def test_existing_custom_prompt_stays_the_active_default():
    owner = _owner({"custom_image_edit_system_prompt": "My own edit prompt", "custom_image_edit_user_prompt": "Be gentle"})
    profiles = manga_env.ImageEditPromptProfiles(owner)
    assert profiles.selected_name() == "Default"
    assert profiles.texts == {"system": "My own edit prompt", "user": "Be gentle"}
    # nothing the runs read changed
    assert owner.config["custom_image_edit_system_prompt"] == "My own edit prompt"
    assert owner.config["custom_image_edit_user_prompt"] == "Be gentle"
    assert owner.config["custom_image_edit_prompt_profile_default"] == {"system": "My own edit prompt",
                                                                       "user": "Be gentle"}
    # the legacy key alone migrates too
    legacy = _owner({"custom_image_edit_prompt": "Legacy prompt"})
    assert manga_env.ImageEditPromptProfiles(legacy).texts["system"] == "Legacy prompt"
    # Reset gives the shipped Default back
    assert profiles.reset("Default") is None
    assert profiles.texts == {"system": DEFAULT_PROMPT, "user": ""}
    assert owner.config["custom_image_edit_system_prompt"] == DEFAULT_PROMPT


def test_selection_persists_and_reopens():
    saves = []
    owner = _owner({"custom_image_edit_system_prompt": "My own edit prompt"})
    profiles = manga_env.ImageEditPromptProfiles(owner, persist=lambda: saves.append(1))
    assert profiles.select(SFX)["system"] == manga_env.IMAGE_EDIT_TRANSLATE_SFX_PROMPT
    assert owner.config["active_custom_image_edit_prompt_profile"] == SFX
    assert owner.config["custom_image_edit_system_prompt"] == manga_env.IMAGE_EDIT_TRANSLATE_SFX_PROMPT
    assert owner.custom_image_edit_system_prompt_var == manga_env.IMAGE_EDIT_TRANSLATE_SFX_PROMPT
    reopened = manga_env.ImageEditPromptProfiles(_owner(owner.config))
    assert reopened.selected_name() == SFX
    assert reopened.profile_texts("Default")["system"] == "My own edit prompt"  # Default kept the custom prompt


def test_user_profiles_new_save_rename_delete_and_protected_builtins():
    saves = []
    owner = _owner()
    profiles = manga_env.ImageEditPromptProfiles(owner, persist=lambda: saves.append(1))
    assert profiles.new() is None and profiles.selected_name() == "New Profile #1"
    assert profiles.save("Signs only", {"system": "Translate the signs to {target_lang}.", "user": ""}) is None
    assert profiles.names == ["Default", SFX, SFX_BG, "Signs only"]
    assert owner.config["custom_image_edit_system_prompt"] == "Translate the signs to {target_lang}."
    assert profiles.delete(SFX)[0] == "Built-in Profile"
    assert profiles.delete("Default")[0] == "Default Profile"
    # saving a built-in under another name saves a copy
    profiles.select(SFX_BG)
    assert profiles.save("My background", {"system": "Changed {target_lang}", "user": ""}) is None
    assert SFX_BG in profiles.names and profiles.selected_name() == "My background"
    assert profiles.delete("Signs only", confirm=lambda name: True) is None
    assert "Signs only" not in profiles.names
    assert saves  # every action saved the config


def test_failed_save_rolls_back():
    owner = _owner({"custom_image_edit_system_prompt": "Mine"})
    profiles = manga_env.ImageEditPromptProfiles(owner, persist=lambda: False)
    assert profiles.save("Other", {"system": "x", "user": ""})[0] == "Save Failed"
    assert profiles.names == ["Default", SFX, SFX_BG]
    assert owner.config["custom_image_edit_system_prompt"] == "Mine"


# ---------------------------------------------------------------------------
# D: the desktop dialog (offscreen)
# ---------------------------------------------------------------------------

@pytest.fixture
def qt_dialog(monkeypatch):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    import manga_integration
    import review_dialog

    app = QApplication.instance() or QApplication([])
    boxes = []
    monkeypatch.setattr(manga_integration, "QMessageBox", types.SimpleNamespace(
        warning=lambda *a, **k: boxes.append(a[1:]), question=lambda *a, **k: 1, Yes=1, No=0))
    monkeypatch.setattr(review_dialog, "confirm_prompt_profile_action", lambda *a, **k: True)

    def open_dialog(config, driver):
        saves = []
        main_gui = types.SimpleNamespace(config=config, save_config=lambda show_message=False: saves.append(1))
        tab = types.SimpleNamespace(dialog=None, main_gui=main_gui, logs=[], rendering_saves=[])
        tab._log = lambda message, level="info": tab.logs.append((message, level))
        tab._save_rendering_settings = lambda: tab.rendering_saves.append(1)
        cls = manga_integration.MangaTranslationTab
        for name in ("_sync_custom_image_edit_prompt", "_default_custom_image_edit_system_prompt",
                     "_is_old_custom_image_edit_default_prompt"):
            setattr(tab, name, types.MethodType(getattr(cls, name), tab))
        results = {}

        def fake_exec(dialog):
            results["value"] = driver(dialog)
            return 0

        monkeypatch.setattr(manga_integration.QDialog, "exec", fake_exec)
        cls._edit_custom_image_edit_prompt(tab)
        app.processEvents()
        return tab, saves, results.get("value")

    yield open_dialog, boxes
    app.processEvents()


def _items(combo):
    return [combo.itemText(i) for i in range(combo.count())]


def test_dialog_profile_row_selects_saves_and_resets(qt_dialog):
    from PySide6.QtWidgets import QPushButton

    open_dialog, boxes = qt_dialog
    config = {"custom_image_edit_system_prompt": "My own edit prompt"}

    def drive(dialog):
        combo = dialog._image_edit_profile_combo
        buttons = dialog._image_edit_profile_buttons
        system_editor, user_editor = dialog._image_edit_prompt_editors
        seen = {"items": _items(combo), "current": combo.currentText(), "system": system_editor.toPlainText(),
                "delete_label": buttons["delete"].text(),
                "names": (combo.objectName(), sorted(b.objectName() for b in buttons.values())),
                "styles": (combo.styleSheet(), buttons["new"].styleSheet(), buttons["save"].styleSheet())}
        combo.setCurrentIndex(combo.findText(SFX))
        seen["after_select"] = (system_editor.toPlainText(), config.get("active_custom_image_edit_prompt_profile"),
                                config.get("custom_image_edit_system_prompt"))
        system_editor.setPlainText("Edited SFX prompt for {target_lang}")
        buttons["delete"].click()  # ↺ Reset the built-in: the shipped text comes back
        seen["after_reset"] = system_editor.toPlainText()
        system_editor.setPlainText("Edited SFX prompt for {target_lang}")
        [b for b in dialog.findChildren(QPushButton) if b.text() == "Save"][0].click()
        seen["after_save"] = config.get("custom_image_edit_system_prompt")
        return seen

    tab, saves, seen = open_dialog(config, drive)
    assert seen["items"] == ["Default", SFX, SFX_BG]
    assert seen["current"] == "Default" and seen["system"] == "My own edit prompt"
    assert seen["delete_label"] == "↺ Reset Profile"  # Default is built-in (its shipped prompt)
    assert seen["names"] == ("image_edit_prompt_profile_combo",
                             ["image_edit_profile_delete_button", "image_edit_profile_new_button",
                              "image_edit_profile_save_button"])
    # the same styling as the review / glossary prompt profile rows
    combo_style, new_style, save_style = seen["styles"]
    assert "Halgakos.ico" in combo_style and "#2d5a2d" in new_style and "#2f5f8f" in save_style
    assert seen["after_select"] == (manga_env.IMAGE_EDIT_TRANSLATE_SFX_PROMPT, SFX,
                                    manga_env.IMAGE_EDIT_TRANSLATE_SFX_PROMPT)
    assert seen["after_reset"] == manga_env.IMAGE_EDIT_TRANSLATE_SFX_PROMPT
    assert seen["after_save"] == "Edited SFX prompt for {target_lang}"
    # the runs read the tab value / main GUI var / config of the saved profile
    assert tab.custom_image_edit_system_prompt_value == "Edited SFX prompt for {target_lang}"
    assert tab.main_gui.custom_image_edit_system_prompt_var == "Edited SFX prompt for {target_lang}"
    assert config["custom_image_edit_prompt_profiles"][SFX]["system"] == "Edited SFX prompt for {target_lang}"
    assert config["custom_image_edit_prompt_profile_default"]["system"] == "My own edit prompt"
    assert saves and tab.rendering_saves and not boxes


# ---------------------------------------------------------------------------
# R: the request that reaches the image edit model
# ---------------------------------------------------------------------------

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")


def _inpainter(**config):
    from local_inpainter import LocalInpainter

    inpainter = object.__new__(LocalInpainter)
    inpainter.config = dict(config)
    inpainter.current_method = "custom-image-edit"
    inpainter.model_loaded = True
    inpainter._mp_enabled = False
    inpainter._check_stop = lambda: False
    inpainter.logs = []
    inpainter._log = lambda message, level="info": inpainter.logs.append((level, message))
    inpainter._sync_inpainter_key_pool_from_config = lambda: None
    inpainter._custom_image_edit_use_current_provider = False
    inpainter._custom_image_edit_endpoint = "http://edit.test/v1"
    return inpainter


def _page():
    """A bubble with dialogue at the top, a sound effect drawn in the art below it (not in the mask)."""
    img = np.full((400, 300, 3), 230, np.uint8)
    cv2.ellipse(img, (150, 80), (110, 45), 0, 0, 360, (255, 255, 255), -1)
    cv2.ellipse(img, (150, 80), (110, 45), 0, 0, 360, (0, 0, 0), 3)
    cv2.putText(img, "HI", (120, 95), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
    cv2.putText(img, "KWANG", (40, 300), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (20, 20, 160), 6)
    mask = np.zeros(img.shape[:2], np.uint8)
    cv2.rectangle(mask, (60, 50), (240, 110), 255, -1)
    return img, mask


def _fake_endpoint(monkeypatch, output):
    import requests

    ok, png = cv2.imencode(".png", output)
    assert ok
    posted = []

    class Response:
        status_code = 200
        text = ""

        @staticmethod
        def json():
            return {"data": [{"b64_json": base64.b64encode(png.tobytes()).decode("ascii")}]}

    def fake_post(url, **kwargs):
        posted.append((url, kwargs))
        return Response()

    monkeypatch.setattr(requests, "post", fake_post)
    monkeypatch.setenv("CUSTOM_IMAGE_EDIT_MODEL", "fake-edit-model")
    monkeypatch.setenv("CUSTOM_IMAGE_EDIT_API_KEY", "sk-fake")
    monkeypatch.setenv("CUSTOM_IMAGE_EDIT_FULL_PAGE_OUTPUT", "10")
    return posted


def _sent_mask(kwargs):
    raw = np.frombuffer(kwargs["files"]["mask"][1], np.uint8)
    rgba = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
    return rgba[:, :, 3] == 0  # transparent = editable


def _translated_page():
    page, _ = _page()
    out = page.copy()
    cv2.ellipse(out, (150, 80), (100, 38), 0, 0, 360, (255, 255, 255), -1)  # dialogue removed
    cv2.rectangle(out, (30, 240), (290, 320), (230, 230, 230), -1)
    cv2.putText(out, "BOOM", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (20, 20, 160), 6)  # SFX redrawn
    return out


def test_selected_profile_prompt_reaches_the_request_with_the_target_language(monkeypatch):
    """The SFX profile's prompt is what the image edit model receives, {target_lang} replaced by the
    output language the text translation uses (OUTPUT_LANGUAGE), not the glossary's language."""
    page, mask = _page()
    owner = _owner({"output_language": "Spanish"})
    profiles = manga_env.ImageEditPromptProfiles(owner)
    profiles.select(SFX)
    posted = _fake_endpoint(monkeypatch, _translated_page())
    monkeypatch.setenv("OUTPUT_LANGUAGE", "Spanish")
    monkeypatch.setenv("GLOSSARY_TARGET_LANGUAGE", "English")  # set by the manga run env as well
    inpainter = _inpainter(**owner.config)

    result = inpainter._custom_image_edit_inpaint_chunked(page, mask)

    assert len(posted) == 1
    url, kwargs = posted[0]
    prompt = kwargs["data"]["prompt"]
    assert prompt == manga_env.IMAGE_EDIT_TRANSLATE_SFX_PROMPT.replace("{target_lang}", "Spanish")
    assert "{target_lang}" not in prompt and "into Spanish" in prompt and "English" not in prompt
    # a translate prompt edits (and keeps) the whole crop: the SFX lies outside the detected box
    assert _sent_mask(kwargs).all()
    assert result is not None and result.shape == page.shape
    sfx_area = (slice(240, 320), slice(30, 290))
    assert np.abs(result[sfx_area].astype(int) - _translated_page()[sfx_area].astype(int)).mean() < 3
    # the leftover-text check does not undo the redrawn lettering
    assert inpainter.last_image_edit_unremoved_mask is None
    assert not any("left the text in place" in message for _level, message in inpainter.logs)


def test_placeholder_is_the_translation_output_language_not_the_glossary_language(monkeypatch):
    """{target_lang} in a saved prompt (no profile involved) is filled like the translation prompt profiles
    fill it: OUTPUT_LANGUAGE wins over the GLOSSARY_TARGET_LANGUAGE the manga run env also sets."""
    page, mask = _page()
    posted = _fake_endpoint(monkeypatch, _translated_page())
    monkeypatch.setenv("OUTPUT_LANGUAGE", "French")
    monkeypatch.setenv("GLOSSARY_TARGET_LANGUAGE", "English")
    inpainter = _inpainter(custom_image_edit_system_prompt="Redraw the sound effects in {target_lang}.",
                           custom_image_edit_user_prompt="Use {target_lang} lettering.")
    inpainter._custom_image_edit_inpaint_chunked(page, mask)
    prompt = posted[0][1]["data"]["prompt"]
    assert prompt == "Redraw the sound effects in French.\n\nUser prompt:\nUse French lettering."


def test_config_language_fills_the_placeholder_without_the_env(monkeypatch):
    page, mask = _page()
    owner = _owner({"output_language": "Korean", "glossary_target_language": "English"})
    profiles = manga_env.ImageEditPromptProfiles(owner)
    profiles.select(SFX_BG)
    posted = _fake_endpoint(monkeypatch, _translated_page())
    inpainter = _inpainter(**owner.config)
    inpainter._custom_image_edit_inpaint_chunked(page, mask)
    prompt = posted[0][1]["data"]["prompt"]
    assert prompt == manga_env.IMAGE_EDIT_TRANSLATE_SFX_BACKGROUND_PROMPT.replace("{target_lang}", "Korean")


def test_default_profile_keeps_the_mask_only_edit(monkeypatch):
    """The Default (removal) prompt is unchanged: only the detected box is editable and kept."""
    page, mask = _page()
    owner = _owner()
    manga_env.ImageEditPromptProfiles(owner)
    posted = _fake_endpoint(monkeypatch, _translated_page())
    inpainter = _inpainter(**owner.config)
    result = inpainter._custom_image_edit_inpaint_chunked(page, mask)
    url, kwargs = posted[0]
    assert kwargs["data"]["prompt"] == DEFAULT_PROMPT
    sent = _sent_mask(kwargs)
    assert sent[80, 150] and not sent[300, 150]
    sfx_area = (slice(260, 310), slice(40, 280))
    assert np.array_equal(result[sfx_area], page[sfx_area])  # the art outside the box is the source's


def test_translate_prompt_sends_crops_without_detected_text(monkeypatch):
    """A tall page: the crop holding only the sound effect is sent too with a translate prompt."""
    top, top_mask = _page()
    bottom = np.full((400, 300, 3), 230, np.uint8)
    cv2.putText(bottom, "KWANG", (40, 200), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (20, 20, 160), 6)
    page = np.vstack([top, bottom])
    mask = np.vstack([top_mask, np.zeros(bottom.shape[:2], np.uint8)])
    settings = {"manga_settings": {"preprocessing": {"chunk_height": 400, "chunk_overlap": 0}},
                "manga_batch_image_requests_enabled": False}
    for name, expected in (("Default", 1), (SFX, 2)):
        owner = _owner(settings)
        manga_env.ImageEditPromptProfiles(owner).select(name)
        import requests

        posted = []

        def fake_post(url, **kwargs):
            posted.append(kwargs)
            raw = kwargs["files"]["image"][1]
            crop = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
            ok, png = cv2.imencode(".png", crop)
            return types.SimpleNamespace(status_code=200, text="", json=lambda: {
                "data": [{"b64_json": base64.b64encode(png.tobytes()).decode("ascii")}]})

        monkeypatch.setattr(requests, "post", fake_post)
        monkeypatch.setenv("CUSTOM_IMAGE_EDIT_MODEL", "fake-edit-model")
        monkeypatch.setenv("CUSTOM_IMAGE_EDIT_API_KEY", "sk-fake")
        inpainter = _inpainter(**owner.config)
        assert inpainter._custom_image_edit_inpaint_chunked(page, mask) is not None
        assert len(posted) == expected, name


def test_translate_prompt_failed_textless_crop_keeps_source_and_overlap_is_a_hard_cut(monkeypatch):
    """Live test (S21: a Vertex "content silently blocked" crop failed the whole page; overlapping crops
    were cross-faded, so the redrawn art showed twice). With a translate prompt a failed crop that has no
    detected text keeps its source art, a failed crop with text still fails the page, and the overlap
    of two kept crops is a hard cut (no mix of both redraws)."""
    import requests

    top, top_mask = _page()
    bottom = np.full((400, 300, 3), 230, np.uint8)
    page = np.vstack([top, bottom])
    mask = np.vstack([top_mask, np.zeros(bottom.shape[:2], np.uint8)])
    settings = {"manga_settings": {"preprocessing": {"chunk_height": 420, "chunk_overlap": 40}},
                "manga_batch_image_requests_enabled": False}
    owner = _owner(settings)
    manga_env.ImageEditPromptProfiles(owner).select(SFX)
    monkeypatch.setenv("CUSTOM_IMAGE_EDIT_MODEL", "fake-edit-model")
    monkeypatch.setenv("CUSTOM_IMAGE_EDIT_API_KEY", "sk-fake")
    monkeypatch.setenv("CUSTOM_IMAGE_EDIT_FULL_PAGE_OUTPUT", "10")

    def endpoint(fill_for_call):
        calls = []

        def fake_post(url, **kwargs):
            calls.append(kwargs)
            fill = fill_for_call(len(calls))
            if fill is None:
                return types.SimpleNamespace(status_code=500, text="blocked", json=lambda: {})
            crop = cv2.imdecode(np.frombuffer(kwargs["files"]["image"][1], np.uint8), cv2.IMREAD_COLOR)
            ok, png = cv2.imencode(".png", np.full_like(crop, fill))
            return types.SimpleNamespace(status_code=200, text="", json=lambda: {
                "data": [{"b64_json": base64.b64encode(png.tobytes()).decode("ascii")}]})

        monkeypatch.setattr(requests, "post", fake_post)
        return calls

    inpainter = _inpainter(**owner.config)
    ranges = inpainter._custom_image_edit_chunk_ranges(mask, 420, 40)
    assert len(ranges) == 2 and ranges[1][0] < ranges[0][1]  # the two crops overlap

    # both crops come back: the overlap is the first crop's top half and the second crop's bottom half
    calls = endpoint(lambda n: 50 if n == 1 else 200)
    result = inpainter._custom_image_edit_inpaint_chunked(page, mask)
    assert len(calls) == 2 and result is not None
    assert set(np.unique(result)) <= {50, 200}  # no cross-faded rows
    cut = ranges[1][0] + (ranges[0][1] - ranges[1][0]) // 2
    assert (result[:cut] == 50).all() and (result[cut:] == 200).all()

    # the text-free bottom crop fails: the page is kept, with the bottom crop's source art
    calls = endpoint(lambda n: 50 if n == 1 else None)
    inpainter = _inpainter(**owner.config)
    result = inpainter._custom_image_edit_inpaint_chunked(page, mask)
    assert len(calls) == 2 and result is not None
    assert np.array_equal(result[ranges[0][1]:], page[ranges[0][1]:])

    # the crop with the detected dialogue fails: the page fails as before
    calls = endpoint(lambda n: None if n == 1 else 200)
    inpainter = _inpainter(**owner.config)
    assert inpainter._custom_image_edit_inpaint_chunked(page, mask) is None


# ---------------------------------------------------------------------------
# B: blank user-drawn boxes stay in the mask
# ---------------------------------------------------------------------------

import manga_editor_core  # noqa: E402


class _FastClockThread(threading.Thread):
    def join(self, timeout=None):
        return super().join(None if timeout is None else timeout / 100.0)


def _editor(config=None):
    editor = types.SimpleNamespace(main_gui=types.SimpleNamespace(config=dict(config or {})), update_queue=Queue(),
                                   inpaint_method_value="local", skip_inpainting_value=False,
                                   full_page_context_value=False, logs=[])
    editor._log = lambda message, level="info": editor.logs.append((str(message), level))
    return editor


DIALOGUE_BOX = [10, 10, 40, 20]
SFX_BOX = [60, 60, 50, 30]
DETECTED_BLANK_BOX = [10, 70, 20, 20]


def test_translate_masks_a_blank_user_drawn_box(monkeypatch, tmp_path):
    """Translate cleans the user's box around a sound effect although OCR read no text in it; a detector
    box without OCR text is still left alone."""
    page = tmp_path / "page.png"
    page.write_bytes(b"")
    editor = _editor({"manga_inpaint_method": "local", "manga_local_inpaint_model": "custom-image-edit"})
    editor._current_regions = [
        {"bbox": DIALOGUE_BOX, "rect_index": 0, "bubble_type": "text_bubble"},
        {"bbox": SFX_BOX, "rect_index": 1, "user_drawn": True},
        {"bbox": DETECTED_BLANK_BOX, "rect_index": 2, "bubble_type": "free_text"},
    ]
    masked = []

    def record_inpaint(self, image_path, regions):
        masked.extend(regions)
        return None

    monkeypatch.setattr(threading, "Thread", _FastClockThread)
    monkeypatch.setattr(manga_editor_core, "_run_inpainting_sync", record_inpaint)
    monkeypatch.setattr(manga_editor_core, "_translate_individually",
                        lambda self, texts, image_path: [dict(texts[0], translation="Hello")])
    texts = [{"text": "안녕", "bbox": DIALOGUE_BOX, "region_index": 0}]
    manga_editor_core._run_translate_background(editor, texts, str(page))

    boxes = [list(region["bbox"]) for region in masked]
    assert DIALOGUE_BOX in boxes and SFX_BOX in boxes
    assert DETECTED_BLANK_BOX not in boxes


def test_ocr_filter_keeps_user_drawn_boxes():
    regions = [{"bbox": DIALOGUE_BOX}, {"bbox": SFX_BOX, "user_drawn": True}, {"bbox": DETECTED_BLANK_BOX}]
    kept = manga_editor_core._regions_with_ocr_text(regions, [{"text": "안녕", "bbox": DIALOGUE_BOX, "region_index": 0}])
    assert [r["bbox"] for r in kept] == [DIALOGUE_BOX, SFX_BOX]


class _RecordingInpainter:
    masks = []
    unchecked = []

    def __init__(self, enable_worker_process=True, **kwargs):
        self.config = {}
        self.log_callback = None
        self.model_loaded = False

    def load_model(self, method, model_path, force_reload=False):
        self.model_loaded = True
        return True

    def set_log_callback(self, callback):
        self.log_callback = callback

    def inpaint(self, image, mask, **kwargs):
        _RecordingInpainter.masks.append(mask.copy())
        _RecordingInpainter.unchecked.append(getattr(self, "_image_edit_unchecked_mask", None))
        return image


def test_inpainting_mask_covers_a_blank_user_box(monkeypatch, tmp_path):
    import local_inpainter
    from manga_translator import MangaTranslator

    monkeypatch.setattr(MangaTranslator, "_inpaint_pool", {})
    MangaTranslator.set_global_cancellation(False)
    page = tmp_path / "page.png"
    cv2.imwrite(str(page), np.full((120, 120, 3), 255, dtype=np.uint8))
    model_file = tmp_path / "lama_fp32.onnx"
    model_file.write_bytes(b"onnx")
    editor = _editor({"manga_inpaint_method": "local", "manga_local_inpaint_model": "lama_onnx",
                      "manga_lama_onnx_model_path": str(model_file)})
    monkeypatch.setattr(local_inpainter, "LocalInpainter", _RecordingInpainter)
    monkeypatch.setattr(_RecordingInpainter, "masks", [])
    monkeypatch.setattr(_RecordingInpainter, "unchecked", [])
    regions = [{"bbox": DIALOGUE_BOX, "text": "안녕"},
               {"bbox": SFX_BOX, "text": "", "user_drawn": True},  # the user's box, OCR read nothing
               {"bbox": DETECTED_BLANK_BOX, "text": ""}]  # a detector box OCR read nothing in
    manga_editor_core._run_inpainting_sync(editor, str(page), regions)
    assert len(_RecordingInpainter.masks) == 1
    mask = _RecordingInpainter.masks[0]

    def covered(box):
        x, y, w, h = box
        return bool(mask[y + h // 2, x + w // 2])

    assert covered(DIALOGUE_BOX) and covered(SFX_BOX) and not covered(DETECTED_BLANK_BOX)
    # only the user's blank box is left out of the image edit's leftover-text check
    unchecked = _RecordingInpainter.unchecked[0]
    assert unchecked is not None
    x, y, w, h = SFX_BOX
    assert unchecked[y + h // 2, x + w // 2] and not unchecked[DIALOGUE_BOX[1] + 5, DIALOGUE_BOX[0] + 5]


def test_leftover_check_keeps_the_edit_in_a_blank_user_box_with_art():
    """Live test 2026-11-07 (S21 wanted poster): the user's blank box held the poster's picture; the model
    removed the letters, but the leftover-text check counted the picture as text left and put the source art
    back. Components touching the user's blank box are not judged; other text still is."""
    from local_inpainter import LocalInpainter

    inpainter = object.__new__(LocalInpainter)
    inpainter.config = {}
    inpainter._log = lambda *a, **k: None
    inpainter._image_edit_local_fallback = None
    source = np.full((200, 200, 3), 235, dtype=np.uint8)
    cv2.circle(source, (60, 70), 22, (30, 30, 30), 4)          # drawn art inside the user's box
    cv2.putText(source, "AB", (36, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2)  # its letters
    cv2.putText(source, "CD", (130, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2)  # detected text
    edited = source.copy()
    edited[100:130, 30:90] = 235                               # the model removed "AB", kept the art
    mask = np.zeros((200, 200), np.uint8)
    mask[30:135, 25:100] = 255                                  # the user's blank box
    mask[35:70, 120:180] = 255                                  # a detected box the model missed
    user_box = np.zeros_like(mask)
    user_box[30:135, 25:100] = 255

    inpainter._image_edit_unchecked_mask = None
    reverted = inpainter._custom_image_edit_check_text_removed(source, edited.copy(), mask)
    assert np.array_equal(reverted[100:130, 30:90], source[100:130, 30:90])  # the old behaviour

    inpainter._image_edit_unchecked_mask = user_box
    kept = inpainter._custom_image_edit_check_text_removed(source, edited.copy(), mask)
    assert np.array_equal(kept[100:130, 30:90], edited[100:130, 30:90])
    left = inpainter.last_image_edit_unremoved_mask
    assert left is not None and left[50, 150] and not left[115, 60]  # "CD" is still judged and flagged


def test_drawn_box_is_marked_user_drawn_and_reaches_the_regions():
    pytest.importorskip("PySide6")
    from PySide6.QtCore import QRectF
    from PySide6.QtWidgets import QApplication
    import manga_image_preview

    app = QApplication.instance() or QApplication([])
    widget = manga_image_preview.MangaImagePreviewWidget(main_gui=types.SimpleNamespace(config={}))
    try:
        item = manga_image_preview.MoveableRectItem(QRectF(60, 60, 50, 30))
        widget.viewer._scene.addItem(item)
        widget.viewer.rectangles.append(item)
        widget.viewer.rectangle_created.emit(QRectF(60, 60, 50, 30))
        app.processEvents()
        assert getattr(item, "user_drawn", False) is True
        host = types.SimpleNamespace(image_preview_widget=widget, _log=lambda *a, **k: None)
        regions = manga_editor_core._extract_regions_from_preview(host)
        assert regions[-1]["user_drawn"] is True  # bbox: the scene rect (pen width included)
        # the mark is saved with the box and comes back on a redrawn box
        saved = {}
        widget.manga_integration = types.SimpleNamespace(image_state_manager=types.SimpleNamespace(
            get_state=lambda path: {}, set_state=lambda path, state, save=True: saved.update(state)))
        widget.current_image_path = "page.png"
        widget._persist_rectangles_state()
        assert saved["viewer_rectangles"][-1]["user_drawn"] is True
        restored = types.SimpleNamespace()
        manga_editor_core._copy_region_metadata_to_item(
            restored, manga_editor_core._saved_rect_metadata({}, 0, saved["viewer_rectangles"][-1]))
        assert restored.user_drawn is True
    finally:
        widget.deleteLater()
        app.processEvents()
