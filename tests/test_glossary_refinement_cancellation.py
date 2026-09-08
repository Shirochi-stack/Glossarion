"""Completed refinement work survives cancellation of unrelated requests."""

import csv
import io
import json
import threading
from types import SimpleNamespace

import pytest

import glossary_refinement as refinement


COLUMNS = ["type", "raw_name", "translated_name"]
TYPES = ["terms", "item", "character", "surnames"]


def entry(entry_type, refined=False):
    return {
        "type": entry_type,
        "raw_name": f"source {entry_type}",
        "translated_name": f"{'refined' if refined else 'original'} {entry_type}",
    }


def make_plan(chunk_types, mode="all"):
    chunks = []
    for selected_types in chunk_types:
        payload = io.StringIO()
        writer = csv.DictWriter(payload, fieldnames=COLUMNS, lineterminator="\n")
        writer.writerows(entry(entry_type) for entry_type in selected_types)
        chunks.append(refinement.RefinementPlannedChunk(
            payload=payload.getvalue(),
            entry_type="selected glossary entries" if mode == "all" else selected_types[0],
            selected_types=list(selected_types),
            columns=list(COLUMNS),
            token_count=len(payload.getvalue()),
            whole_type_chunk=len(selected_types) == 1,
        ))
    selected_types = list(dict.fromkeys(t for group in chunk_types for t in group))
    return refinement.RefinementPlan(
        selected_types=selected_types,
        chunking_mode=mode,
        chunks=chunks,
        per_type_counts={t: 1 for t in selected_types},
        available_tokens=10000,
    )


@pytest.fixture
def run_refinement(monkeypatch, tmp_path):
    monkeypatch.setenv("BATCH_TRANSLATION", "0")
    monkeypatch.setenv("BATCH_SIZE", "2")
    monkeypatch.setenv("GLOSSARY_CUSTOM_FIELDS", "[]")
    monkeypatch.setenv("GLOSSARY_REFINEMENT_SKIP_DEDUPE", "0")
    monkeypatch.setenv("GLOSSARY_REFINEMENT_SYSTEM_PROMPT", "Refine {fields} for {entries}.")
    monkeypatch.setenv("GLOSSARY_REFINEMENT_USER_PROMPT", "")
    progress_file = str(tmp_path / "glossary_progress.json")
    stop = threading.Event()

    def run(send, plan=None, glossary=None, selected_types=None):
        plan = plan or make_plan([TYPES[:2], TYPES[2:]])
        selected_types = selected_types or plan.selected_types
        return refinement.refine_glossary_entries(
            glossary or [entry(t) for t in TYPES],
            client=SimpleNamespace(model="refinement-test-model"),
            temp=0,
            mtoks=1000,
            check_stop=stop.is_set,
            chapter_splitter=SimpleNamespace(count_tokens=len),
            available_tokens=10000,
            chunk_timeout=5,
            parse_response_fn=json.loads,
            dedupe_fn=lambda entries: entries,
            custom_entry_types_fn=lambda: {t: {"enabled": True} for t in TYPES},
            send_fn=send,
            progress_file=progress_file,
            output_path=str(tmp_path / "glossary.csv"),
            log=lambda message: None,
            options=refinement.RefinementRunOptions(
                selected_types=selected_types,
                chunking_mode=plan.chunking_mode,
                run_when_disabled=True,
            ),
            plan=plan,
        )

    run.stop = stop
    run.progress = lambda: refinement.load_refinement_progress(progress_file)
    return run


def response(*entry_types):
    return json.dumps([entry(t, refined=True) for t in entry_types]), "stop", None


def assert_completed(progress, entry_types):
    for entry_type in entry_types:
        saved = progress[f"type::{entry_type}"]
        assert saved["status"] == "completed"
        assert saved["completed_chunks"] == saved["total_chunks"] == 1
        assert saved["entry_count_after"] == 1
        assert saved["output_identity_hash"]


def assert_partial_output(result):
    assert {e["type"]: e for e in result} == {
        t: entry(t, refined=t in TYPES[:2]) for t in TYPES
    }


@pytest.mark.parametrize("cancel_response", ["exception", "empty", "truncated"])
def test_cancelled_all_mode_chunk_preserves_completed_types(run_refinement, cancel_response):
    calls = []

    def send(*args, chunk_idx, **kwargs):
        calls.append(chunk_idx)
        if chunk_idx == 1:
            return response(*TYPES[:2])
        assert_completed(run_refinement.progress(), TYPES[:2])
        run_refinement.stop.set()
        if cancel_response == "exception":
            raise RuntimeError("request interrupted by user")
        if cancel_response == "empty":
            return "[]", "stop", None
        return json.dumps([entry("character", refined=True)]), "length", None

    result = run_refinement(send)

    assert calls == [1, 2]
    assert_partial_output(result)
    progress = run_refinement.progress()
    assert_completed(progress, TYPES[:2])
    for entry_type in TYPES[2:]:
        assert progress[f"type::{entry_type}"]["status"] == "in_progress"
        assert progress[f"type::{entry_type}"]["completed_chunks"] == 0
        assert progress[f"type::{entry_type}"]["total_chunks"] == 1
    aggregate = progress[f"all::{','.join(TYPES)}"]
    assert aggregate["status"] == "in_progress"
    assert aggregate["completed_chunks"] == 1
    assert aggregate["total_chunks"] == 2


def test_genuine_chunk_failure_only_fails_its_incomplete_types(run_refinement):
    def send(*args, chunk_idx, **kwargs):
        if chunk_idx == 1:
            return response(*TYPES[:2])
        raise RuntimeError("provider unavailable")

    result = run_refinement(send)

    assert_partial_output(result)
    progress = run_refinement.progress()
    assert_completed(progress, TYPES[:2])
    for entry_type in TYPES[2:]:
        saved = progress[f"type::{entry_type}"]
        assert saved["status"] == "failed"
        assert saved["error"] == "provider unavailable"
        assert saved["completed_chunks"] == 0
        assert saved["total_chunks"] == 1
    aggregate = progress[f"all::{','.join(TYPES)}"]
    assert aggregate["status"] == "failed"
    assert aggregate["completed_chunks"] == 1


def test_provider_user_stop_is_cancellation_without_local_stop_flag(run_refinement):
    def send(*args, chunk_idx, **kwargs):
        if chunk_idx == 1:
            return response(*TYPES[:2])
        raise RuntimeError("AuthGPT: Translation stopped by user")

    result = run_refinement(send)

    assert_partial_output(result)
    progress = run_refinement.progress()
    assert_completed(progress, TYPES[:2])
    for entry_type in TYPES[2:]:
        assert progress[f"type::{entry_type}"]["status"] == "in_progress"


def test_stop_without_completed_types_preserves_original_glossary(run_refinement):
    glossary = [dict(entry(t), description="Existing description") for t in TYPES]

    def send(*args, **kwargs):
        run_refinement.stop.set()
        raise RuntimeError("request interrupted by user")

    result = run_refinement(send, glossary=glossary)

    # Schema preprocessing must not trigger a save in the manual caller when
    # cancellation happened before any type produced a complete result.
    assert result == glossary
    assert all(row["description"] == "Existing description" for row in result)


@pytest.mark.parametrize("fail", [False, True])
def test_separate_plan_alias_uses_configured_type_progress(run_refinement, fail):
    def send(*args, **kwargs):
        if fail:
            raise RuntimeError("provider unavailable")
        return response("terms")

    result = run_refinement(
        send,
        plan=make_plan([["term"]], mode="separate"),
        selected_types=["terms"],
        glossary=[entry("terms")],
    )

    progress = run_refinement.progress()
    assert "type::term" not in progress
    assert result == [entry("terms", refined=not fail)]
    if fail:
        assert progress["type::terms"]["status"] == "failed"
        assert progress["type::terms"]["error"] == "provider unavailable"
    else:
        assert_completed(progress, ["terms"])


def test_type_spanning_cancelled_chunk_keeps_its_original_entries(run_refinement):
    plan = make_plan([["terms", "item"], ["terms", "character", "surnames"]])
    second_term = dict(entry("terms"), raw_name="source second term")
    plan.chunks[1].payload = plan.chunks[1].payload.replace("source terms", second_term["raw_name"])
    plan.per_type_counts["terms"] = 2
    glossary = [entry(t) for t in TYPES] + [second_term]

    def send(*args, chunk_idx, **kwargs):
        if chunk_idx == 1:
            return response("terms", "item")
        run_refinement.stop.set()
        raise RuntimeError("request interrupted by user")

    result = run_refinement(send, plan=plan, glossary=glossary)

    assert [e for e in result if e["type"] == "terms"] == [entry("terms"), second_term]
    assert next(e for e in result if e["type"] == "item") == entry("item", refined=True)
    progress = run_refinement.progress()
    assert_completed(progress, ["item"])
    assert progress["type::terms"]["status"] == "in_progress"
    assert progress["type::terms"]["completed_chunks"] == 1
    assert progress["type::terms"]["total_chunks"] == 2
    for entry_type in TYPES[2:]:
        assert progress[f"type::{entry_type}"]["completed_chunks"] == 0
        assert progress[f"type::{entry_type}"]["total_chunks"] == 1


@pytest.mark.parametrize("mode", ["all", "separate"])
def test_parallel_success_is_collected_after_another_request_stops(
    run_refinement, monkeypatch, mode,
):
    monkeypatch.setenv("BATCH_TRANSLATION", "1")
    successful_request_started = threading.Event()
    release_success = threading.Event()
    plan = make_plan([["terms"], ["character"]], mode=mode)

    def stopped_future_first(futures):
        # Both requests are running. Deliver the stopped result to the consumer
        # before the successful result, independent of the thread scheduler.
        successful, cancelled = list(futures)
        try:
            cancelled.result(timeout=5)
        finally:
            release_success.set()
        yield cancelled
        successful.result(timeout=5)
        yield successful

    monkeypatch.setattr(refinement, "as_completed", stopped_future_first)

    def send(messages, *args, **kwargs):
        if "source terms" in messages[-1]["content"]:
            successful_request_started.set()
            assert release_success.wait(timeout=5)
            return response("terms")
        assert successful_request_started.wait(timeout=5)
        run_refinement.stop.set()
        raise RuntimeError("request interrupted by user")

    result = run_refinement(send, plan=plan)

    by_type = {e["type"]: e for e in result}
    assert by_type["terms"] == entry("terms", refined=True)
    assert by_type["character"] == entry("character")
    progress = run_refinement.progress()
    assert_completed(progress, ["terms"])
    assert progress["type::character"]["status"] == "in_progress"


def test_cancellation_leaves_previously_completed_types_unchanged(run_refinement):
    initial = run_refinement(
        lambda *args, **kwargs: response(*TYPES[:2]),
        plan=make_plan([TYPES[:2]]),
    )
    previously_completed = {
        t: run_refinement.progress()[f"type::{t}"] for t in TYPES[:2]
    }
    calls = []

    def send(messages, *args, **kwargs):
        payload = messages[-1]["content"]
        calls.append(payload)
        assert "source terms" not in payload
        assert "source item" not in payload
        run_refinement.stop.set()
        raise RuntimeError("request interrupted by user")

    result = run_refinement(send, glossary=initial)

    assert len(calls) == 1
    assert_partial_output(result)
    progress = run_refinement.progress()
    for entry_type, saved in previously_completed.items():
        assert progress[f"type::{entry_type}"] == saved
    for entry_type in TYPES[2:]:
        assert progress[f"type::{entry_type}"]["status"] == "in_progress"
