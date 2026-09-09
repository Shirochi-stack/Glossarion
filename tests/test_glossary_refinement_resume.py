"""Preserve completed categories unless source-change reopening is enabled."""

import hashlib
import json
import unicodedata
from types import SimpleNamespace

import pytest

import glossary_refinement as refinement


TYPES = ["terms", "item", "character", "surnames"]


def entry(entry_type, raw_name=None):
    return {
        "type": entry_type,
        "raw_name": raw_name or f"source {entry_type}",
        "translated_name": f"translation {entry_type}",
    }


def v1_identity(entry_type, entries, mode):
    """Reproduce the persisted v1 format without calling the current hash helper."""
    payload = {
        "version": "raw-name-v1",
        "entry_type": unicodedata.normalize("NFC", entry_type).strip().casefold(),
        "chunking_mode": mode.strip().casefold(),
        "raw_names": sorted({
            unicodedata.normalize("NFC", row["raw_name"]).strip().casefold()
            for row in entries
        }),
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


@pytest.fixture
def resume(monkeypatch, tmp_path):
    monkeypatch.delenv("GLOSSARY_REFINEMENT_REOPEN_ON_SOURCE_CHANGE", raising=False)
    monkeypatch.setenv("GLOSSARY_REFINEMENT_ENABLED", "1")
    monkeypatch.setenv("GLOSSARY_REFINEMENT_TYPE_MODE", "all")
    monkeypatch.setenv("GLOSSARY_REFINEMENT_CHUNKING_MODE", "all")
    monkeypatch.setenv("GLOSSARY_CUSTOM_FIELDS", "[]")
    monkeypatch.setenv("GLOSSARY_REFINEMENT_SKIP_DEDUPE", "0")
    monkeypatch.setenv("GLOSSARY_REFINEMENT_SYSTEM_PROMPT", "Refine {fields} for {entries}.")
    monkeypatch.setenv("GLOSSARY_REFINEMENT_USER_PROMPT", "")
    monkeypatch.setenv("BATCH_TRANSLATION", "0")
    progress_file = tmp_path / "glossary_progress.json"
    calls = []
    logs = []

    def run(entries=None, *, mode="all", selected_types=None, active_types=None, force=False):
        entries = entries if entries is not None else [entry(t) for t in TYPES]
        active_types = active_types or TYPES

        def send(messages, *args, **kwargs):
            calls.append(messages[-1]["content"])
            refined = [dict(row, translated_name=f"refined {row['raw_name']}") for row in entries]
            return json.dumps(refined), "stop", None

        return refinement.refine_glossary_entries(
            entries,
            client=SimpleNamespace(model="resume-test-model"),
            temp=0,
            mtoks=1000,
            check_stop=lambda: False,
            chapter_splitter=SimpleNamespace(count_tokens=len),
            available_tokens=10000,
            chunk_timeout=5,
            parse_response_fn=json.loads,
            dedupe_fn=lambda rows: rows,
            custom_entry_types_fn=lambda: {t: {"enabled": True} for t in active_types},
            send_fn=send,
            progress_file=str(progress_file),
            output_path=str(tmp_path / "glossary.csv"),
            log=logs.append,
            options=refinement.RefinementRunOptions(
                selected_types=selected_types,
                chunking_mode=mode,
                force=force,
                run_when_disabled=True,
            ),
        )

    def seed(records):
        progress_file.write_text(json.dumps({"refinement": records}), encoding="utf-8")

    run.calls = calls
    run.logs = logs
    run.seed = seed
    run.progress = lambda: refinement.load_refinement_progress(str(progress_file))
    return run


@pytest.fixture
def reopen_on_source_change(resume, monkeypatch):
    monkeypatch.setenv("GLOSSARY_REFINEMENT_REOPEN_ON_SOURCE_CHANGE", "1")


@pytest.mark.parametrize(
    "old_mode,new_mode,change_delimiter",
    [("separate", "all", False), ("all", "separate", False), ("all", "all", True)],
)
def test_request_mode_or_delimiter_change_does_not_reopen_completed_type(
    resume, monkeypatch, old_mode, new_mode, change_delimiter,
):
    original = [entry("terms")]
    persisted = resume(original, mode=old_mode, selected_types=["terms"])
    assert len(resume.calls) == 1
    if change_delimiter:
        monkeypatch.setenv("GLOSSARY_REFINEMENT_USER_PROMPT", "Use this glossary schema: {fields1}")

    result = resume(persisted, mode=new_mode, selected_types=["terms"])

    assert len(resume.calls) == 1
    assert result == persisted
    assert resume.progress()["type::terms"]["identity_hash_version"] == "raw-name-v2"
    if change_delimiter:
        # Verify the changed setting really controls request serialization,
        # after proving an automatic resume still skips the completed type.
        resume(persisted, mode=new_mode, selected_types=["terms"], force=True)
        assert "\x1f" in resume.calls[-1]
        assert resume.progress()["type::terms"]["payload_delimiter"] == "unit_separator"


def test_configured_type_alias_finds_previously_completed_progress(resume):
    persisted = resume([entry("term")], active_types=["term"])
    renamed_type = [dict(row, type="terms") for row in persisted]

    result = resume(renamed_type, active_types=["terms"])

    assert len(resume.calls) == 1
    assert result == renamed_type


@pytest.mark.parametrize(
    "exact_status,alias_status,alias_updated,alias_completed,expected_requests",
    [
        ("completed", "failed", 200, None, 1),
        ("failed", "completed", 200, None, 0),
        ("failed", "completed", 50, 200, 0),
    ],
    ids=["newer-alias-failed", "newer-alias-completed", "legacy-manual-completion-newest"],
)
def test_latest_type_alias_state_controls_resume(
    resume, reopen_on_source_change, exact_status, alias_status, alias_updated,
    alias_completed, expected_requests,
):
    persisted = resume([entry("terms")], selected_types=["terms"])
    completed = resume.progress()["type::terms"]
    exact_record = dict(completed, status=exact_status, last_updated=100)
    alias_record = dict(
        completed,
        entry_type="term",
        status=alias_status,
        last_updated=alias_updated,
    )
    if alias_completed is not None:
        # The legacy manual action retained the old request timestamp and
        # hashes, recording the later user decision only in completed_at.
        alias_record.update({
            "completed_at": alias_completed,
            "identity_hash_version": "raw-name-v1",
            "input_identity_hash": "stale request identity",
            "output_identity_hash": "stale request identity",
        })
    resume.seed({"type::terms": exact_record, "type::term": alias_record})
    resume.calls.clear()

    resume(persisted, selected_types=["terms"])

    assert len(resume.calls) == expected_requests
    if alias_completed is not None:
        migrated = resume.progress()["type::term"]
        assert migrated["manually_marked_completed"] is True
        assert migrated["identity_hash_version"] == "raw-name-v2"


@pytest.mark.parametrize("matching_hash", ["input_identity_hash", "output_identity_hash"])
def test_v1_completed_hash_uses_saved_alias_mode_and_delimiter_then_migrates(
    resume, reopen_on_source_change, matching_hash,
):
    current_entries = [entry("terms")]
    old_hash = v1_identity("term", current_entries, "separate:unit_separator")
    record = {
        "entry_type": "term",
        "status": "completed",
        "identity_hash_version": "raw-name-v1",
        "input_identity_hash": "unrelated former source identity",
        "output_identity_hash": "unrelated former source identity",
        "chunking_mode": "separate",
        "payload_delimiter": "unit_separator",
        "entry_count_after": 1,
        "output_file": "glossary.csv",
    }
    record[matching_hash] = old_hash
    resume.seed({"type::term": record})

    result = resume(current_entries, active_types=["terms"], mode="all")

    assert resume.calls == []
    assert result == current_entries
    migrated = [
        row for key, row in resume.progress().items()
        if key.startswith("type::") and row.get("identity_hash_version") == "raw-name-v2"
    ]
    assert migrated
    assert migrated[0]["input_identity_hash"] == migrated[0]["output_identity_hash"]
    assert migrated[0]["input_identity_hash"] != old_hash

    # Once migrated, switching request settings again must still skip the type.
    resume(current_entries, active_types=["terms"], mode="separate")
    assert resume.calls == []


@pytest.mark.parametrize("reverted_shape", ["input", "output"])
def test_v1_migration_preserves_both_source_shapes_until_a_new_run(
    resume, reopen_on_source_change, reverted_shape,
):
    original_entries = [
        entry("terms", raw_name="old first source"),
        entry("terms", raw_name="old duplicate source"),
    ]
    refined_entries = [entry("terms", raw_name="merged renamed source")]
    saved_mode = "separate:unit_separator"
    resume.seed({"type::terms": {
        "entry_type": "terms",
        "status": "completed",
        "identity_hash_version": "raw-name-v1",
        "input_identity_hash": v1_identity("terms", original_entries, saved_mode),
        "output_identity_hash": v1_identity("terms", refined_entries, saved_mode),
        "chunking_mode": "separate",
        "payload_delimiter": "unit_separator",
        "entry_count_before": 2,
        "entry_count_after": 1,
        "output_file": "glossary.csv",
    }})

    assert resume(original_entries, selected_types=["terms"]) == original_entries
    assert resume.progress()["type::terms"]["identity_hash_version"] == "raw-name-v2"
    assert resume.calls == []

    # Refinement changed raw names and removed a duplicate. Loading that output
    # after migrating from the original input must still reuse completed work.
    assert resume(refined_entries, selected_types=["terms"]) == refined_entries
    assert resume.calls == []

    resume([entry("terms", raw_name="new third source shape")], selected_types=["terms"])
    assert len(resume.calls) == 1

    # A new completed run supersedes both historical identity sets. Neither
    # legacy input nor legacy output may bypass later source-change detection.
    reverted_entries = original_entries if reverted_shape == "input" else refined_entries
    resume(reverted_entries, selected_types=["terms"])
    assert len(resume.calls) == 2


def test_legacy_manual_completion_replaces_stale_hashes_once(resume, reopen_on_source_change):
    current_entries = [entry("terms")]
    resume.seed({"type::terms": {
        "entry_type": "terms",
        "status": "completed",
        "completed_at": 1700000000,
        "identity_hash_version": "raw-name-v1",
        "input_identity_hash": "stale failed-run input identity",
        "output_identity_hash": "stale former-run output identity",
        "entry_count_after": 99,
        "output_file": "glossary.csv",
    }})

    result = resume(current_entries, selected_types=["terms"])

    assert resume.calls == []
    assert result == current_entries
    migrated = resume.progress()["type::terms"]
    assert migrated["manually_marked_completed"] is True
    assert migrated["identity_hash_version"] == "raw-name-v2"
    assert migrated["input_identity_hash"] == migrated["output_identity_hash"]
    assert migrated["entry_count_after"] == len(current_entries)

    # The legacy timestamp must not exempt this category from future source
    # changes after its acknowledged source identities have been recorded.
    resume([entry("terms", raw_name="replacement source")], selected_types=["terms"])
    assert len(resume.calls) == 1


def test_v1_hash_mismatch_does_not_skip_same_count_source_replacement(
    resume, reopen_on_source_change,
):
    original_hash = v1_identity("terms", [entry("terms")], "separate:comma")
    resume.seed({"type::terms": {
        "entry_type": "terms",
        "status": "completed",
        "identity_hash_version": "raw-name-v1",
        "input_identity_hash": original_hash,
        "output_identity_hash": original_hash,
        "chunking_mode": "separate",
        "payload_delimiter": "comma",
        "entry_count_after": 1,
        "output_file": "glossary.csv",
    }})

    resume([entry("terms", raw_name="replacement source")], selected_types=["terms"])

    assert len(resume.calls) == 1
    assert "replacement source" in resume.calls[0]


@pytest.mark.parametrize("change", ["add", "replace"])
def test_new_or_replaced_source_reopens_only_affected_completed_type(
    resume, reopen_on_source_change, change,
):
    persisted = resume()
    assert len(resume.calls) == 1
    updated = [dict(row) for row in persisted]
    if change == "add":
        updated.append(entry("terms", raw_name="new source term"))
    else:
        updated[0]["raw_name"] = "replacement source term"
    resume.calls.clear()
    resume.logs.clear()

    resume(updated)

    assert len(resume.calls) == 1
    assert "source term" in resume.calls[0]
    for unaffected_type in TYPES[1:]:
        assert f"source {unaffected_type}" not in resume.calls[0]
    assert any("source" in line.lower() and "chang" in line.lower() for line in resume.logs)


@pytest.mark.parametrize("setting", [None, "0"], ids=["default-off", "explicitly-off"])
@pytest.mark.parametrize("completion", ["automatic", "manual", "legacy"])
def test_new_chapter_entries_keep_completed_types_and_original_identity_metadata(
    resume, monkeypatch, setting, completion,
):
    persisted = resume()
    records = resume.progress()
    if completion == "manual":
        for record in records.values():
            record["manually_marked_completed"] = True
            record["completed_at"] = 1700000000
    elif completion == "legacy":
        for record in records.values():
            record.pop("identity_hash_version", None)
            record.pop("input_identity_hash", None)
            record.pop("output_identity_hash", None)
    resume.seed(records)
    if setting is not None:
        monkeypatch.setenv("GLOSSARY_REFINEMENT_REOPEN_ON_SOURCE_CHANGE", setting)
    updated = persisted + [
        entry("character", "character first seen in chapter 447"),
        entry("surnames", "surname first seen in chapter 447"),
    ]
    resume.calls.clear()

    result = resume(updated)

    assert resume.calls == []
    assert result == updated
    assert resume.progress() == records


@pytest.mark.parametrize("change", ["add", "replace"])
def test_enabling_reopening_later_detects_changes_skipped_while_disabled(
    resume, monkeypatch, change,
):
    persisted = resume()
    records = resume.progress()
    updated = [dict(row) for row in persisted]
    if change == "add":
        updated.append(entry("character", "new chapter character"))
    else:
        next(row for row in updated if row["type"] == "character")["raw_name"] = (
            "replacement character"
        )
    resume.calls.clear()

    assert resume(updated) == updated
    assert resume.calls == []
    assert resume.progress() == records

    monkeypatch.setenv("GLOSSARY_REFINEMENT_REOPEN_ON_SOURCE_CHANGE", "1")
    resume(updated)

    assert len(resume.calls) == 1
    assert "character" in resume.calls[0]
    for unaffected_type in ("terms", "item", "surnames"):
        assert f"source {unaffected_type}" not in resume.calls[0]
    assert resume.progress()["type::character"]["input_identity_hash"] != (
        records["type::character"]["input_identity_hash"]
    )


@pytest.mark.parametrize("status", [None, "failed", "stopped", "not_refined"])
def test_reopening_disabled_still_processes_types_without_completion(resume, status):
    persisted = resume()
    records = resume.progress()
    if status is None:
        records.pop("type::character")
    else:
        records["type::character"]["status"] = status
    resume.seed(records)
    resume.calls.clear()

    resume(persisted)

    assert len(resume.calls) == 1
    assert "source character" in resume.calls[0]
    for unaffected_type in ("terms", "item", "surnames"):
        assert f"source {unaffected_type}" not in resume.calls[0]
    assert resume.progress()["type::character"]["status"] == "completed"


def test_manual_force_resends_completed_selection(resume):
    persisted = resume()
    resume.calls.clear()

    resume(persisted, selected_types=["terms"], force=True)

    assert len(resume.calls) == 1
    assert "source terms" in resume.calls[0]
    for unaffected_type in TYPES[1:]:
        assert f"source {unaffected_type}" not in resume.calls[0]


def test_automatic_scope_and_logs_identify_only_pending_types(resume):
    persisted = resume(selected_types=TYPES[:2])
    resume.calls.clear()
    resume.logs.clear()

    resume(persisted)

    assert len(resume.calls) == 1
    payload = resume.calls[0]
    for completed_type in TYPES[:2]:
        assert f"source {completed_type}" not in payload
        assert any(
            "skip" in line.lower() and completed_type in line.lower()
            for line in resume.logs
        )
    for pending_type in TYPES[2:]:
        assert f"source {pending_type}" in payload
    assert any(
        "pending" in line.lower() and all(t in line.lower() for t in TYPES[2:])
        for line in resume.logs
    )
