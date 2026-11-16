"""Owner's phone report of 2026-11-16: "Glossary generation is not complete."

The model picked on the phone needed an API key and none was saved. A book run with an automatic glossary then
showed "Glossary generation complete · 0 entries · Accept this glossary and start translation?" seven seconds
after it started, and Glossary › Continue ended as "Glossary extraction stopped (see the log)." The real reason
("❌ Error: Please enter your API key.") was only in the log: the glossary pass refuses every file without a key,
and the book run went on to announce the pass complete and to ask for the glossary's approval.

Now the book run ends at once with the reason and asks nothing, and both jobs show the reason on their card.

The real-app test drives both jobs through the app's own UI; nothing may reach the model.

Run from src/mobile:
    .venv/Scripts/python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic \
        tests_host/test_glossary_stage_incomplete.py
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
for _path in (APP_DIR, MOBILE_DIR / "tests", MOBILE_DIR / "extensions" / "flet_glossarion_native" / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

ft = pytest.importorskip("flet")


def _load(filename: str, alias: str):
    spec = importlib.util.spec_from_file_location(alias, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


I7 = _load("test_hotfix1_issue7.py", "_glossarion_i7_helpers_glossary_incomplete")
storage = I7.storage  # noqa: F811  (pytest fixtures)
app_env = I7.app_env
isolated = I7.isolated

#: A model name no provider route matches: it needs an API key (``UnifiedClient._model_needs_api_key``).
KEYED_MODEL = "glossarion-keyed-test-model"
NO_KEY = "Please enter your API key."


# ---- the error line of the two jobs ---------------------------------------------------------------------------


def test_the_job_error_is_the_reason_the_run_could_not_start():
    from glossarion_mobile.job_kinds.glossary import EXTRACTION_STOPPED, extraction_error, setup_error

    # the translation worker recorded why its glossary pre-pass refused
    owner = types.SimpleNamespace(_run_setup_error=NO_KEY)
    assert setup_error(owner) == NO_KEY
    assert extraction_error(owner) == f"Glossary extraction could not start: {NO_KEY}"
    # an extraction job: the shared check answers
    owner = types.SimpleNamespace(_glossary_setup_refusal=lambda: "No model is selected.")
    assert extraction_error(owner) == "Glossary extraction could not start: No model is selected."
    # nothing wrong with the set-up: the line the job always had
    for owner in (types.SimpleNamespace(), types.SimpleNamespace(_run_setup_error=None, _glossary_setup_refusal=lambda: None)):
        assert setup_error(owner) == "" and extraction_error(owner) == EXTRACTION_STOPPED

    def broken():
        raise RuntimeError("no config")

    assert setup_error(types.SimpleNamespace(_glossary_setup_refusal=broken)) == ""


# ---- the real app ------------------------------------------------------------------------------------------------


def _log_lines(js, job_id: str) -> list:
    with open(js.job_log_path(job_id), encoding="utf-8", errors="replace") as handle:
        return [line.rstrip("\n") for line in handle]


def test_a_book_run_without_an_api_key_ends_with_the_reason_and_asks_nothing(isolated):
    from glossarion_mobile.diagnostics.fake_llm_server import FakeLLMServer
    from glossarion_mobile.services.jobs import is_glossary_question

    picks, files = isolated
    tf = I7._load("test_ui_foundations.py", "_glossarion_tf_helpers_glossary_incomplete")

    async def scenario(server):
        run = await I7._start(tf, server, picks, files)
        app, driver = run.app, run.driver
        js = app.job_service
        questions = []
        unsubscribe = js.subscribe(lambda view: questions.append(view.active.question)
                                   if view.active is not None and view.active.question else None)
        try:
            # the phone's state: a keyed model, no key, the automatic glossary on
            app.config_store.set_many({"model": KEYED_MODEL, "api_key": "", "use_custom_openai_endpoint": False,
                                       "auto_glossary_mode": "full", "extraction_workers": 1})
            await I7._open_library(run)
            book = await I7._import_book(run, f"{I7.BOOK_STEM}.epub")
            book_page, _bid = await I7._open_chapters(run, book)

            # 1. the book run
            await driver.tap(key="ch-run", timeout=10)
            await driver.wait(key="start", timeout=30)
            await driver.tap(key="start", timeout=10)
            if await driver.exists(text="Keep translations running", timeout=2):
                await driver.tap(text="Not now")
            # the run ends within a second: it may already be in the history when this looks
            snap = await I7._until(lambda: next((s for s in [js.snapshot(), *js.view().history]
                                                 if s is not None and I7._kind(s) == "translate"), None), 60)
            assert snap is not None
            end = await I7._wait_end(js, snap.id)
            assert I7._state(end) == "FAILED", end
            assert end.error == f"The translation could not start: {NO_KEY}"
            log = _log_lines(js, snap.id)
            assert any(line.endswith(f"❌ Error: {NO_KEY}") for line in log), log[-30:]
            assert any("Translation cancelled: glossary generation could not start" in line for line in log)
            said_complete = [line for line in log if "glossary generation is complete" in line.lower()
                             or "glossary extraction complete" in line.lower() or "Retrying" in line]
            assert said_complete == []
            assert not [q for q in questions if is_glossary_question(str((q or {}).get("kind") or ""))]
            assert end.question is None

            # 2. Glossary › Continue / Extract glossary
            tab = book_page.glossary_tab
            assert await I7._until(lambda: tab.extract_wait_reason() is None, 20), tab.extract_wait_reason()
            extraction = await tab.extract()
            assert extraction and extraction != snap.id
            end = await I7._wait_end(js, extraction)
            assert I7._kind(end) == "extract_glossary" and I7._state(end) == "FAILED", end
            assert end.error == f"Glossary extraction could not start: {NO_KEY}"
            assert any(line.endswith(f"❌ Error: {NO_KEY}") for line in _log_lines(js, extraction))

            # nothing was sent to a model
            assert [r.kind for r in server.requests if r.kind in ("glossary", "translation")] == []
        except Exception:
            for row in run.tester.dump(200):
                print(row)
            raise
        finally:
            unsubscribe()
            app.jobs.close()
            await tf._stop(app)

    with FakeLLMServer() as server:
        asyncio.run(scenario(server))
