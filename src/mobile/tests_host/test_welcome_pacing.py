"""Welcome step 3: API call delay, temperature / Disable temperature and the batch size (owner 2026-10-12),
and Glossarion Mobile's own defaults (model authchatgpt/auto; batch size 1 for the web-chat routes)."""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

MOBILE_DIR = Path(__file__).resolve().parents[1]
for entry in (str(MOBILE_DIR / "app"), str(MOBILE_DIR.parent)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

pytest.importorskip("flet")

from glossarion_mobile.state.setting_writes import (  # noqa: E402
    MOBILE_DEFAULT_MODEL,
    default_batch_size,
    with_mobile_streaming_defaults,
)
from glossarion_mobile.ui.chat.send_state import DEFAULT_MODEL  # noqa: E402
from glossarion_mobile.ui.router import parse_route  # noqa: E402
from glossarion_mobile.ui.screens.welcome import BATCH_SLIDER, WelcomeScreen  # noqa: E402


def controls(root):
    """Every control under ``root`` (depth first)."""
    stack = [root]
    while stack:
        node = stack.pop()
        yield node
        for name in ("controls", "content"):
            child = getattr(node, name, None)
            if isinstance(child, list):
                stack.extend(reversed(child))
            elif child is not None and hasattr(child, "__dict__") and not isinstance(child, str):
                stack.append(child)


def by_key(screen):
    return {getattr(c, "key", None): c for c in controls(screen.page_area.content)}


def event(value):
    return types.SimpleNamespace(control=types.SimpleNamespace(value=value))


def welcome(model="", saved=None):
    saved = dict(saved or {})
    screen = WelcomeScreen(parse_route("/welcome"), current_model=lambda: model,
                           config_get=lambda key, default=None: saved.get(key, default))
    screen.get_body()
    screen.go("language_glossary")
    return screen


def test_mobile_defaults_are_chatgpt_web_chat_and_one_request_at_a_time():
    assert DEFAULT_MODEL == MOBILE_DEFAULT_MODEL == "authchatgpt/auto"
    config = with_mobile_streaming_defaults({})
    assert config["model"] == "authchatgpt/auto" and config["batch_size"] == 1
    assert with_mobile_streaming_defaults({"model": "authds/flash"})["batch_size"] == 1
    assert with_mobile_streaming_defaults({"model": "authgpt/gpt-6-luna"})["batch_size"] == 5
    saved = with_mobile_streaming_defaults({"model": "authchatgpt/auto", "batch_size": 3})
    assert saved["batch_size"] == 3  # a saved value always wins
    assert [default_batch_size(m) for m in ("authds/pro", "AuthChatGPT/gpt-6-pro", "gemini-3.5-flash")] == [1, 1, 5]


def test_pacing_controls_sit_between_the_output_limit_and_the_batch_size():
    screen = welcome("authchatgpt/auto")
    keys = [k for k in by_key(screen) if k]
    order = [keys.index(k) for k in ("welcome-max-tokens-slider", "welcome-delay", "welcome-temperature",
                                     "welcome-disable-temperature", "welcome-batch-size-slider",
                                     "welcome-rate-limit-warning")]
    assert order == sorted(order)
    warning = by_key(screen)["welcome-rate-limit-warning"].value
    assert "authds/" in warning and "authchatgpt/" in warning and "rate limit" in warning


@pytest.mark.parametrize("model,expected", [("authchatgpt/auto", 1), ("authds/flash", 1), ("", 1),
                                            ("authgpt/gpt-6-luna", 5), ("gemini-3.5-flash", 5)])
def test_batch_size_default_follows_the_model(model, expected):
    rows = by_key(welcome(model))
    assert rows["welcome-batch-size"].value == str(expected)
    assert rows["welcome-batch-size-slider"].value == expected


def test_saved_values_are_shown_and_left_alone_unless_changed():
    screen = welcome("authchatgpt/auto", {"batch_size": 7, "delay": 2.5, "translation_temperature": 0.7,
                                          "disable_temperature": True})
    rows = by_key(screen)
    assert rows["welcome-batch-size"].value == "7" and rows["welcome-delay"].value == "2.5"
    assert rows["welcome-temperature"].value == "0.7" and rows["welcome-temperature"].disabled
    assert rows["welcome-disable-temperature"].value is True
    updates = screen.flow.finish_updates()
    assert not {"batch_size", "delay", "translation_temperature", "disable_temperature"} & set(updates)


def test_changes_are_written_on_finish_and_the_field_goes_beyond_the_slider():
    screen = welcome("authgpt/gpt-6-luna")
    rows = by_key(screen)
    rows["welcome-batch-size"].on_change(event("120"))
    assert screen.flow.batch_size == 120 and rows["welcome-batch-size-slider"].value == BATCH_SLIDER[1]
    rows["welcome-batch-size-slider"].on_change(event(12.0))
    assert screen.flow.batch_size == 12 and rows["welcome-batch-size"].value == "12"
    rows["welcome-delay"].on_change(event("0.5"))
    rows["welcome-temperature"].on_change(event("1.1"))
    rows["welcome-temperature"].on_change(event("9"))  # out of range: ignored
    rows["welcome-disable-temperature"].on_change(event(True))
    assert rows["welcome-temperature"].disabled
    updates = screen.flow.finish_updates()
    assert (updates["batch_size"], updates["delay"], updates["translation_temperature"],
            updates["disable_temperature"]) == (12, 0.5, 1.1, True)


def test_target_language_is_editable_and_a_typed_language_is_saved():
    """Step 2 "Choose Your Language and Glossary Mode": pick a listed language or type any other one."""
    from glossarion_mobile.ui.screens.welcome_flow import STEP_TITLES

    assert STEP_TITLES["language_glossary"] == "Choose Your Language and Glossary Mode"
    screen = welcome("authchatgpt/auto")
    language = by_key(screen)["welcome-language"]
    assert language.editable and language.enable_filter and language.value == "English"
    language.on_text_change(types.SimpleNamespace(control=types.SimpleNamespace(text="  Tagalog ")))
    assert screen.flow.target_language == "Tagalog"
    language.on_text_change(types.SimpleNamespace(control=types.SimpleNamespace(text="")))  # cleared: kept
    assert screen.flow.target_language == "Tagalog"
    language.on_select(event("Korean"))
    assert screen.finish()["output_language"] == "Korean"


@pytest.mark.parametrize("model", ["authchatgpt/auto", "local/qwen3-0.6b-q4"])
def test_text_extraction_choice_sits_below_the_language_and_is_saved_for_every_model(model):
    """Welcome › Text extraction: BeautifulSoup (recommended, the default) or html2text, a line on each; Finish
    always saves ``text_extraction_method`` (the default too: the choice is what sets the method)."""
    from glossarion_mobile.ui.screens.welcome_flow import EXTRACTION_CHOICES

    screen = welcome(model)
    rows = by_key(screen)
    keys = [k for k in rows if k]
    assert keys.index("welcome-language") < keys.index("welcome-extraction") < keys.index("welcome-max-tokens-slider")
    group = rows["welcome-extraction"]
    assert group.value == "standard"
    assert [c[:2] for c in EXTRACTION_CHOICES] == [("standard", "BeautifulSoup (recommended)"),
                                                  ("enhanced", "html2text")]
    assert rows["welcome-extraction-standard"].label == "BeautifulSoup (recommended)"
    assert welcome(model).finish()["text_extraction_method"] == "standard"
    group.on_change(event("enhanced"))
    group.on_change(event("bogus"))  # not a choice: ignored
    assert screen.finish()["text_extraction_method"] == "enhanced"


@pytest.mark.parametrize("saved,shown", [({"text_extraction_method": "enhanced"}, "enhanced"),
                                         ({"text_extraction_method": "standard"}, "standard"),
                                         ({"extraction_mode": "enhanced"}, "enhanced"), ({}, "standard")])
def test_text_extraction_shows_the_saved_method_when_reopened(saved, shown):
    screen = welcome("authchatgpt/auto", saved)
    assert by_key(screen)["welcome-extraction"].value == shown
    assert screen.finish()["text_extraction_method"] == shown

