"""Owner request: "add the same buttons on top here that the initial response card had when we attached the
file". A stopped / failed / interrupted Result (Resume / Retry failed) shows the Plan card's model, profile,
language, glossary (+ Policy) and range chips above its Run options - not output mode / Save to - and Resume
runs with what they show: the chat's model / profile / language / glossary now and the card's range.

Run from src/mobile (mobile venv):
    python -m pytest -p no:cacheprovider -W ignore tests_host/test_result_card_chips.py -q
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path

MOBILE_DIR = Path(__file__).resolve().parents[1]
for _path in (MOBILE_DIR / "app", MOBILE_DIR / "tests"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # its dataclasses look their module up
    spec.loader.exec_module(module)
    return module


TC = _load("_glossarion_chat_helpers_result_chips", Path(__file__).with_name("test_chat.py"))
RO = _load("_glossarion_run_options_helpers_result_chips", Path(__file__).with_name("test_run_options_resume.py"))
desktop_store_cls = TC.desktop_store_cls
isolated_env = TC.isolated_env
needs_flet = TC.needs_flet

PLAN_KEYS = ["plan-model", "plan-profile", "plan-language", "plan-glossary", "plan-output", "plan-range",
             "plan-destination"]
RESULT_KEYS = ["result-model", "result-profile", "result-language", "result-glossary", "result-range"]


def _plan_chips(view) -> list:
    import flet as ft

    rows = [c for c in TC._plan_box(view) if isinstance(c, ft.Row)]
    return [c for row in rows for c in row.controls if isinstance(c, ft.Chip)]


def _result_card(view):
    from glossarion_mobile.ui.chat.cards import JobCard

    return next(c for c in TC._cards(view, JobCard) if c.offers_resume)


def _chips(card) -> dict:
    return {c.key: c for c in card.resume_chips.controls}


def _label(chip) -> str:
    return str(chip.label.value)


@needs_flet
def test_result_card_shows_the_plan_chips_and_resume_runs_with_them(desktop_store_cls, isolated_env):
    tmp_path = isolated_env

    async def scenario():
        chat = TC._chat_view(desktop_store_cls, tmp_path)
        view, adapter, jobs = chat.view, chat.adapter, chat.jobs
        cid = view.cid
        try:
            await RO._settle(chat.calls)
            book = tmp_path / "novel.epub"
            book.write_bytes(b"PK\x03\x04")
            assert view.attach_file(str(book))
            view.begin_send()
            # the Plan card keeps exactly its chips
            assert [c.key for c in _plan_chips(view)] == PLAN_KEYS
            view.start_plan()
            await RO._settle(chat.calls)
            sent = jobs.submitted[-1].params
            assert sent["options"]["force_no_glossary"] is False

            panel = await RO._stop(chat)
            card = _result_card(view)
            chips = _chips(card)
            # the Plan chips without output mode / Save to, above the Run options
            assert list(chips) == RESULT_KEYS and card.resume_chips.visible
            body = card.body.controls
            assert body.index(card.resume_chips) == body.index(card.options_box) - 1
            assert _label(chips["result-range"]) == panel.range_chip_label()
            opened: list = []
            view.open_model_sheet = lambda tab="model", **kw: opened.append(tab)
            # hotfix 5: profile and language open the compact preview (configured entries, custom at the end)
            view.open_mini_model_sheet = lambda tab="model": opened.append(f"mini-{tab}")
            for key in ("result-model", "result-profile", "result-language"):
                chips[key].on_click(None)
            assert opened == ["model", "mini-profile", "mini-language"]

            # a chip change (what an in-chat pick writes) refreshes the labels
            adapter.set_override(cid, "model", "gpt-new")
            adapter.set_override(cid, "profile", "Universal")
            adapter.set_override(cid, "target_language", "French")
            adapter.set_override(cid, "glossary_override_mode", "no_glossary")
            view.apply_settings_changed()
            card = _result_card(view)
            chips = _chips(card)
            assert _label(chips["result-model"]) == "gpt-new"
            assert _label(chips["result-profile"]) == "Universal"
            assert _label(chips["result-language"]) == "→ French"
            assert _label(chips["result-glossary"]) == "Glossary: Off"
            assert _label(chips["result-policy"]) == "Policy: Off"  # the chat's glossary policy chip
            assert "result-output" not in chips and "result-destination" not in chips

            # the range chip writes this card's resume Run options ("Only for this run": its own values)
            panel = RO._resume_panel(view)
            panel.switch.value = True
            panel._on_switch()
            chips["result-range"].on_click(None)
            sheet = view.chapters_sheet
            assert sheet.store is panel.overlay
            assert sheet.apply("3-5")
            if sheet._task is not None:
                sheet._task.cancel()
            assert _label(chips["result-range"]) == panel.range_chip_label() != "All chapters"
            assert panel.values["chapter_range"] == "3-5"

            config = await RO._resume(chat)
            resumed = jobs.submitted[-1].params
            assert config["chapter_range"] == "3-5"
            assert config["model"] == "gpt-new" and resumed["model"] == "gpt-new"
            assert config["active_profile"] == "Universal" and config["output_language"] == "French"
            assert resumed["options"]["force_no_glossary"] is True
            # everything else is what the job ran with (same run root and input)
            assert resumed["run"] == sent["run"] and resumed["input_path"] == sent["input_path"]
            assert resumed["options"]["output_mode"] == sent["options"]["output_mode"]

            # the chat back on the main glossary: the next Resume runs with it
            await RO._stop(chat)
            adapter.set_override(cid, "glossary_override_mode", None)
            adapter.set_override(cid, "target_language", None)
            view.apply_settings_changed()
            chips = _chips(_result_card(view))
            assert "result-policy" not in chips and _label(chips["result-glossary"]).endswith("(auto)")
            config = await RO._resume(chat)
            resumed = jobs.submitted[-1].params
            assert resumed["options"]["force_no_glossary"] is False
            assert "output_language" not in resumed["config_overrides"]
        finally:
            TC._close(chat)

    asyncio.run(scenario())


def test_resubmit_follows_the_chats_overrides_and_glossary_policy(desktop_store_cls, tmp_path):
    """``ChatRuns.resubmit(overrides=, settings=)``: the chat's keys follow the chat now, the run's Run
    options and its own manual glossary (while the policy is still Manual) stay."""
    from glossarion_mobile.ui.chat.direct_text_rules import DirectTextSettings
    from glossarion_mobile.ui.chat.run_request import DirectTextRun, with_chat_overrides, with_current_glossary

    overrides = {"model": "gpt-old", "active_profile": "Old", "output_language": "German", "batch_size": 4}
    assert with_chat_overrides(overrides, {"target_language": "French"}) == {"batch_size": 4,
                                                                             "output_language": "French"}
    run = DirectTextRun(temp_root="r", source_path="s", source_extension=".epub", is_attachment=True,
                        expected_output="o", manual_glossary_path="r/glossary.csv")
    options = {"manual_glossary_path": "r/glossary.csv", "force_no_glossary": False, "output_mode": "text"}
    manual = with_current_glossary(options, run, DirectTextSettings(glossary_override_mode="manual"))
    assert manual == options
    auto = with_current_glossary(options, run, DirectTextSettings(glossary_override_mode="none"))
    assert auto["manual_glossary_path"] == "" and auto["output_mode"] == "text"
    off = with_current_glossary(options, run, DirectTextSettings(glossary_override_mode="no_glossary"))
    assert off["manual_glossary_path"] == "" and off["force_no_glossary"] is True
