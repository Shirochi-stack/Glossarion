"""Owner issues 5 + 6: the chat job card's Run options after a stopped request, and their keyboard.

5. "The chat response card doesn't show me the settings menu after a stopped request. It only does when a
   request starts." The Run options (the Plan card's ``RunOptionsPanel``) are on every Result card that
   offers Resume / Retry failed (stopped, failed, interrupted, done with failed chapters), starting from the
   values the job ran with; Resume runs with what the panel holds (``ChatRuns.resubmit(run_config=…)``).

6. "I can't edit any fields in the response card settings that require the keyboard, since the keyboard
   pops out as soon as it pops in." Root cause: Flet 1.0.3's ``ControlWidget.build`` makes a *new*
   ``GlobalKey()`` for a ScrollKey'd control on every build. The transcript's ``CardSlot`` and every settings
   tile were ScrollKey'd, so each rebuild of the transcript ``ListView`` (its ``LayoutBuilder`` runs again when
   the keyboard shrinks the viewport) unmounted the card and the tile and mounted fresh ones: the focused
   TextField's state and FocusNode were gone and the keyboard closed. The ScrollKey now sits on a zero-size
   marker beside the content (``components.scroll_anchor``), so no editable field has a ScrollKey'd ancestor.

Run from src/mobile (mobile venv):
    python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic tests_host/test_owner_card_settings.py
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
import types
from pathlib import Path

import pytest

MOBILE_DIR = Path(__file__).resolve().parents[1]
for _path in (MOBILE_DIR / "app", MOBILE_DIR / "tests"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TC = _load("_glossarion_chat_helpers_owner_card", Path(__file__).with_name("test_chat.py"))
desktop_store_cls = TC.desktop_store_cls
isolated_env = TC.isolated_env
storage = TC.storage
app_env = TC._TB.app_env
needs_flet = TC.needs_flet


def _scroll_keyed_ancestors(root, target) -> list:
    """The controls on the path from ``root`` down to ``target`` (both included) that carry a ScrollKey."""
    import flet as ft
    from host_tester import _children

    def walk(control, path):
        path = path + [control]
        if control is target:
            return path
        for child in _children(control):
            found = walk(child, path)
            if found is not None:
                return found
        return None

    path = walk(root, []) or []
    return [c for c in path if isinstance(getattr(c, "key", None), ft.ScrollKey)]


def _text_fields(root) -> list:
    import flet as ft
    from host_tester import _children

    out, stack = [], [root]
    while stack:
        control = stack.pop()
        if isinstance(control, ft.TextField):
            out.append(control)
        stack.extend(_children(control))
    return out


# ---- owner #6: no ScrollKey above an editable field ------------------------------------------------------


@needs_flet
def test_card_slot_keeps_its_scroll_key_on_a_marker_not_on_the_card():
    import flet as ft

    from glossarion_mobile.ui.chat.transcript import CardSlot, Transcript

    transcript = Transcript()
    card = ft.Column([ft.TextField(value="10")])
    slot = transcript.slot("job-3", card)
    assert isinstance(slot, CardSlot) and slot.card is card and slot.slot_key == "job-3"
    assert slot.key is None, "a ScrollKey'd slot is re-keyed (remounted) by Flet on every build"
    marker = slot.content.controls[0]
    assert marker.key == ft.ScrollKey("job-3") and marker.height == 0
    assert transcript.slot_for(ft.ScrollKey("job-3")) is slot
    other = ft.Text("x")
    assert transcript.slot("job-3", other) is slot and slot.card is other and slot.content.controls[0] is marker
    assert _scroll_keyed_ancestors(slot, other) == []


@needs_flet
def test_settings_tile_fields_have_no_scroll_keyed_ancestor(desktop_store_cls, isolated_env):
    """The Run options' tiles (and every Settings tile): search still reaches the tile through its marker."""
    import flet as ft

    from glossarion_mobile.ui.chat.plan_card import RunOptionsPanel

    store, schema = TC._config_store(isolated_env)
    try:
        ctx = types.SimpleNamespace(store=store, schema=schema, page=None, push=lambda *a: None,
                                    say=lambda *a, **k: None)
        panel = RunOptionsPanel(ctx, values={"batch_size": 7})
        fields = _text_fields(panel.control)
        assert fields, "the Run options have number fields"
        for field in fields:
            assert _scroll_keyed_ancestors(panel.control, field) == []
        tile = panel.tiles["batch_size"]
        assert tile.control.key is None and tile.control.content.controls[0].key == ft.ScrollKey("batch_size")
    finally:
        store._saver.close()


# ---- owner #5: Run options on a stopped card, used by Resume -----------------------------------------------


def _ended_run(state: str, params: dict):
    return types.SimpleNamespace(live=False, state=state, last_snapshot=None, params=params, user_index=2,
                                 output_folder="", output_dir="", job_id="job9")


@needs_flet
@pytest.mark.parametrize("state", ["stopped", "failed"])
def test_stopped_card_shows_run_options_and_resume_uses_them(desktop_store_cls, isolated_env, state):
    from glossarion_mobile.ui.chat.cards import JobCard
    from glossarion_mobile.ui.chat.plan_card import RunOptionsPanel

    async def scenario():
        chat = TC._chat_view(desktop_store_cls, isolated_env)
        view, adapter, calls = chat.view, chat.adapter, chat.calls
        try:
            await TC._settle(calls)
            params = {"run": {"temp_root": str(isolated_env / "root")}, "user_index": 2,
                      "config_overrides": {"model": "gpt-x", "batch_size": 7, "translation_temperature": 0.4}}
            run = _ended_run(state, params)
            view.env.runs.run_for = lambda cid: run
            view._run_turn = lambda r: 2
            resubmitted: list = []

            async def resubmit(cid, profile_check=None, run_config=None, run_config_for=None):
                resubmitted.append((run_config, run_config_for))
                return None

            view.env.runs.resubmit = resubmit
            view.render_transcript()
            card = next(c for c in TC._cards(view, JobCard) if c.title_text.value == "book.epub")
            assert card.phase.name == state and ("resume" in card.action_buttons or "retry" in card.action_buttons)
            assert card.offers_resume and card.options_box.visible, "no Run options on a %s card" % state
            panels = [p for k, p in view.run_panels.items() if k[1] == "resume"]
            assert len(panels) == 1 and isinstance(panels[0], RunOptionsPanel)
            panel = panels[0]
            assert card.options_box.controls == [panel.control]
            # the values the job ran with (its Run options, not the chat's model)
            assert panel.values == {"batch_size": 7, "translation_temperature": 0.4}
            assert panel.tiles["batch_size"].field.value == "7"
            # a re-render keeps the same panel (its fields keep their state)
            view.render_transcript()
            again = next(c for c in TC._cards(view, JobCard) if c.title_text.value == "book.epub")
            assert again.options_box.controls[0] is panel.control
            assert panel.tiles["batch_size"].apply(12)
            item = TC._job_item(adapter)
            view._on_job_action("resume" if state == "stopped" else "retry", item)
            await TC._settle(calls)
            root_key = "run:" + str(isolated_env / "root")
            assert resubmitted == [({"batch_size": 12, "translation_temperature": 0.4}, root_key)]
            # the panel was released: a card of the next ending starts from that job's own values
            assert all(p is not panel for k, p in view.run_panels.items() if k[1] == "resume")
        finally:
            TC._close(chat)

    asyncio.run(scenario())


@needs_flet
def test_done_card_without_failures_has_no_run_options(desktop_store_cls, isolated_env):
    from glossarion_mobile.ui.chat.cards import JobCard

    async def scenario():
        chat = TC._chat_view(desktop_store_cls, isolated_env)
        view, calls = chat.view, chat.calls
        try:
            await TC._settle(calls)
            run = _ended_run("done", {"run": {"temp_root": "r"}, "config_overrides": {"batch_size": 7}})
            view.env.runs.run_for = lambda cid: run
            view._run_turn = lambda r: 2
            view.render_transcript()
            card = next(c for c in TC._cards(view, JobCard) if c.title_text.value == "book.epub")
            assert card.phase.name == "done" and not card.offers_resume
            assert not card.options_box.visible and not card.options_box.controls
        finally:
            TC._close(chat)

    asyncio.run(scenario())


def test_resubmit_replaces_the_run_options_and_keeps_the_chat_keys(desktop_store_cls, tmp_path):
    from pathlib import Path as _P

    jobs = TC.FakeJobService()
    adapter, runs = TC._runs(desktop_store_cls, tmp_path, jobs)
    try:
        run = asyncio.run(runs.send("5", text="안녕하세요", attachment=None, settings=TC.DirectTextSettings(),
                                    output_mode="text", overrides={"model": "gpt-x"},
                                    config_extra={"batch_size": 5, "translation_temperature": 0.3}))
        assert jobs.submitted[0].params["config_overrides"] == {"batch_size": 5, "translation_temperature": 0.3,
                                                                "model": "gpt-x"}
        jobs.publish("RUNNING")
        runs.request_stop("5")
        _P(run.run.expected_output).parent.mkdir(parents=True, exist_ok=True)
        _P(run.run.expected_output).write_text("Hello", encoding="utf-8")
        jobs.publish("DONE")
        TC._finish_all(runs)
        assert not run.live and run.state == "stopped"
        asyncio.run(runs.resubmit("5"))  # no run_config: the run's own Run options
        assert jobs.submitted[-1].params["config_overrides"] == {"batch_size": 5, "translation_temperature": 0.3,
                                                                 "model": "gpt-x"}
        jobs.publish("CANCELLED")
        TC._finish_all(runs)
        asyncio.run(runs.resubmit("5", run_config={"batch_size": 9}))
        assert jobs.submitted[-1].params["config_overrides"] == {"batch_size": 9, "model": "gpt-x"}
        jobs.publish("CANCELLED")
        TC._finish_all(runs)
        asyncio.run(runs.resubmit("5", run_config={}))  # "Only for this run" off: the config's values
        assert jobs.submitted[-1].params["config_overrides"] == {"model": "gpt-x"}
        jobs.publish("CANCELLED")
        TC._finish_all(runs)
        # Run options taken from another run's card (an older turn) are not applied to this run
        asyncio.run(runs.resubmit("5", run_config={"batch_size": 1}, run_config_for="run:elsewhere"))
        assert jobs.submitted[-1].params["config_overrides"] == {"model": "gpt-x"}
        jobs.publish("CANCELLED")
        TC._finish_all(runs)
        from glossarion_mobile.ui.chat.run_controller import turn_key
        own = turn_key(jobs.submitted[-1].params)
        asyncio.run(runs.resubmit("5", run_config={"batch_size": 2}, run_config_for=own))
        assert jobs.submitted[-1].params["config_overrides"] == {"batch_size": 2, "model": "gpt-x"}
        jobs.publish("CANCELLED")
        TC._finish_all(runs)
    finally:
        adapter.close()


# ---- the real app: the keyboard's resize keeps the Plan card's fields ------------------------------------


@needs_flet
def test_keyboard_resize_keeps_the_plan_cards_run_option_fields(app_env, tmp_path):
    """The real app on the fake session: a Plan card (attach an EPUB, Send), its Run options field focused, then
    the keyboard: ``page.on_resize`` with a shorter page and ``on_media_change``. The transcript is not
    re-rendered, its TextFields are the same objects, and none of them sits under a ScrollKey'd control (Flet
    re-keys those on the client's rebuild, which the keyboard's relayout always causes)."""
    import flows
    from glossarion_mobile.diagnostics import fixtures
    from glossarion_mobile.diagnostics.fake_llm_server import FAKE_MODEL

    epub = fixtures.build_tiny_epub(tmp_path / flows.EPUB_NAME, chapters=3)
    twin = _load("_glossarion_ui_flows_owner_card", Path(__file__).with_name("test_ui_flows.py"))
    tf = _load("_glossarion_tf_owner_card", Path(__file__).with_name("test_ui_foundations.py"))

    async def scenario():
        app, tester, driver = await twin._host_driver(tf, {epub.name: epub}, width=411)
        try:
            await flows.wait_home(driver)
            store = app.config_store
            store.set_many(flows.ui_config("http://127.0.0.1:9/v1", FAKE_MODEL))
            store.flush()
            await flows.go_home(driver)
            await driver.tap(tooltip="New chat")
            await driver.tap(tooltip=flows.ATTACH_TOOLTIP)
            await driver.pick_file(epub.name, lambda: driver.tap(key="attach-files"))
            await driver.wait(contains=epub.stem, timeout=60)
            await driver.tap(key="send-idle_ready", timeout=60)
            await driver.wait(text="Ready to translate", timeout=60)
            view = app.chat_view
            panel = next(p for k, p in view.run_panels.items() if k[1] == "plan")
            field = panel.tiles["batch_size"].field
            session = tester.session
            await session.dispatch_event(field._i, "focus", None)
            await asyncio.sleep(0.3)
            before = {f._i: f for f in _text_fields(view.transcript)}
            assert field._i in before
            renders = view._render_gen
            session.apply_page_patch({"height": 480})
            app.on_resize(types.SimpleNamespace(width=411, height=480))
            if callable(getattr(tester.page, "on_media_change", None)):
                tester.page.on_media_change(None)
            await asyncio.sleep(0.5)
            after = {f._i: f for f in _text_fields(view.transcript)}
            assert view._render_gen == renders, "the keyboard re-rendered the transcript"
            assert after.keys() == before.keys() and all(after[i] is before[i] for i in before)
            for text_field in after.values():
                assert _scroll_keyed_ancestors(view.transcript, text_field) == [], text_field
        finally:
            app.jobs.close()
            await tf._stop(app)

    asyncio.run(scenario())
