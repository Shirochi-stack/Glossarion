import json
import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from gender_tracking import (
    automatic_storage_gender,
    collapse_tracked_gender_variants,
    editor_gender_status,
    tracker_path_for_glossary,
)
from glossary_compressor import compress_glossary
import extract_glossary_from_epub as glossary_extractor

try:
    from GlossaryManager_GUI import (
        GlossaryManagerMixin,
        _apply_editor_gender_presentation,
        _gender_resolution_summary,
        _prepare_editor_gender_tracking,
    )
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication, QTreeWidgetItem
except ModuleNotFoundError as exc:
    if exc.name != "PySide6":
        raise
    GlossaryManagerMixin = None
    _apply_editor_gender_presentation = None
    _gender_resolution_summary = None
    _prepare_editor_gender_tracking = None


def _occurrence(gender, chapter, filename=None):
    return {
        "gender": gender,
        "chapter_num": chapter,
        "chapter_index": chapter - 1,
        "chapter_file": filename or f"chapter{chapter}.xhtml",
    }


def _tracker_entry(genders, decision="auto"):
    return {
        "raw_name": "루나",
        "translated_name": "Luna",
        "decision": decision,
        "occurrences": [
            _occurrence(gender, index + 1)
            for index, gender in enumerate(genders)
        ],
        "changes": [],
    }


def _tracker(genders, decision="auto"):
    return {
        "version": 2,
        "entries": {"루나": _tracker_entry(genders, decision)},
    }


def _write_tracker(glossary_path: Path, tracker):
    tracker_path = Path(tracker_path_for_glossary(str(glossary_path)))
    tracker_path.write_text(json.dumps(tracker, ensure_ascii=False), encoding="utf-8")
    return tracker_path


def test_auto_storage_uses_majority_and_stable_tie():
    assert automatic_storage_gender(_tracker_entry(["male", "female", "male"]), "female") == "male"
    assert automatic_storage_gender(_tracker_entry(["male", "female"]), "female") == "female"
    assert automatic_storage_gender(_tracker_entry(["male", "female"]), "") == "male"


def test_auto_storage_counts_unique_chapter_file_observations():
    entry = _tracker_entry(["male", "female"])
    entry["occurrences"].extend([
        dict(entry["occurrences"][0]),
        dict(entry["occurrences"][0]),
    ])

    assert automatic_storage_gender(entry, "female") == "female"


def test_editor_status_is_threshold_and_bias_aware():
    entry = _tracker_entry(["male"] * 9 + ["female"])

    suppressed = editor_gender_status(entry, "male", 10, "none")
    assert suppressed["label"] == "Male*"
    assert suppressed["unresolved"] is True

    preferred = editor_gender_status(entry, "male", 10, "female")
    assert preferred["label"] == "Male / Female"

    manual = dict(entry, decision="female")
    resolved = editor_gender_status(manual, "male", 10, "none")
    assert resolved["label"] == "Female"
    assert resolved["unresolved"] is False


def test_tracker_aware_collapse_keeps_one_row_and_plain_winner():
    entries = [
        {"type": "character", "raw_name": "루나", "translated_name": "Luna", "gender": "female"},
        {"type": "character", "raw_name": "루나", "translated_name": "Moon", "gender": "male", "description": "Hero"},
    ]
    collapsed, removed = collapse_tracked_gender_variants(
        entries,
        _tracker(["male", "female", "male"]),
        has_gender=lambda _entry: True,
        score_entry=lambda entry: len(entry),
    )

    assert removed == 1
    assert collapsed == [{
        "type": "character",
        "raw_name": "루나",
        "translated_name": "Luna",
        "gender": "male",
        "description": "Hero",
    }]


def test_tracker_aware_collapse_keeps_richer_fields_from_the_other_variant():
    entries = [
        {
            "type": "character",
            "raw_name": "루나",
            "translated_name": "Luna",
            "gender": "female",
            "description": "Brief",
        },
        {
            "type": "character",
            "raw_name": "루나",
            "translated_name": "Moon",
            "gender": "male",
            "description": "Detailed",
            "affiliation": "Guild",
        },
    ]

    collapsed, removed = collapse_tracked_gender_variants(
        entries,
        _tracker(["female", "male", "female"]),
        has_gender=lambda _entry: True,
        score_entry=lambda entry: len(entry),
    )

    assert removed == 1
    assert collapsed[0]["raw_name"] == "루나"
    assert collapsed[0]["translated_name"] == "Luna"
    assert collapsed[0]["gender"] == "female"
    assert collapsed[0]["description"] == "Detailed"
    assert collapsed[0]["affiliation"] == "Guild"


def test_dedupe_without_tracker_preserves_existing_gender_variant(monkeypatch):
    monkeypatch.delenv("GLOSSARY_SKIP_GENDER_TRACKING", raising=False)
    entries = [
        {"type": "character", "raw_name": "루나", "translated_name": "Luna", "gender": "male"},
        {"type": "character", "raw_name": "루나", "translated_name": "Luna", "gender": "female"},
    ]
    assert len(glossary_extractor.skip_duplicate_entries(entries)) == 2


def test_dedupe_with_one_gender_of_tracker_evidence_keeps_existing_behavior(tmp_path, monkeypatch):
    monkeypatch.delenv("GLOSSARY_SKIP_GENDER_TRACKING", raising=False)
    glossary_path = tmp_path / "Book_glossary.csv"
    _write_tracker(glossary_path, _tracker(["male", "male"]))
    entries = [
        {"type": "character", "raw_name": "루나", "translated_name": "Luna", "gender": "male"},
        {"type": "character", "raw_name": "루나", "translated_name": "Luna", "gender": "female"},
    ]

    result = glossary_extractor.skip_duplicate_entries(entries, glossary_path=str(glossary_path))

    assert len(result) == 2


def test_dedupe_ignores_tracker_when_tracking_is_disabled(tmp_path, monkeypatch):
    monkeypatch.setenv("GLOSSARY_SKIP_GENDER_TRACKING", "1")
    glossary_path = tmp_path / "Book_glossary.csv"
    _write_tracker(glossary_path, _tracker(["male", "female", "female"]))
    entries = [
        {"type": "character", "raw_name": "루나", "translated_name": "Luna", "gender": "male"},
        {"type": "character", "raw_name": "루나", "translated_name": "Luna", "gender": "female"},
    ]

    result = glossary_extractor.skip_duplicate_entries(entries, glossary_path=str(glossary_path))

    assert len(result) == 1


def test_dedupe_uses_tracker_majority_and_manual_override(tmp_path, monkeypatch):
    monkeypatch.delenv("GLOSSARY_SKIP_GENDER_TRACKING", raising=False)
    glossary_path = tmp_path / "Book_glossary.csv"
    entries = [
        {"type": "character", "raw_name": "루나", "translated_name": "Luna", "gender": "male"},
        {"type": "character", "raw_name": "루나", "translated_name": "Luna", "gender": "female"},
    ]
    tracker_path = _write_tracker(glossary_path, _tracker(["male", "female", "female"]))

    majority = glossary_extractor.skip_duplicate_entries(entries, glossary_path=str(glossary_path))
    assert len(majority) == 1
    assert majority[0]["gender"] == "female"

    tracker = json.loads(tracker_path.read_text(encoding="utf-8"))
    tracker["entries"]["루나"]["decision"] = "male"
    tracker_path.write_text(json.dumps(tracker, ensure_ascii=False), encoding="utf-8")
    manual = glossary_extractor.skip_duplicate_entries(entries, glossary_path=str(glossary_path))
    assert manual[0]["gender"] == "male"


def test_dedupe_updates_an_already_consolidated_row_after_majority_flips(tmp_path, monkeypatch):
    monkeypatch.delenv("GLOSSARY_SKIP_GENDER_TRACKING", raising=False)
    glossary_path = tmp_path / "Book_glossary.csv"
    tracker_path = _write_tracker(glossary_path, _tracker(["male", "female", "female"]))
    entries = [
        {"type": "character", "raw_name": "루나", "translated_name": "Luna", "gender": "female"},
    ]

    tracker = json.loads(tracker_path.read_text(encoding="utf-8"))
    tracker["entries"]["루나"]["occurrences"].extend([
        _occurrence("male", 4),
        _occurrence("male", 5),
    ])
    tracker_path.write_text(json.dumps(tracker, ensure_ascii=False), encoding="utf-8")

    updated = glossary_extractor.skip_duplicate_entries(entries, glossary_path=str(glossary_path))
    assert updated[0]["gender"] == "male"


def test_v1_tracker_defaults_to_auto_and_decision_update_preserves_history(tmp_path, monkeypatch):
    monkeypatch.delenv("GLOSSARY_SKIP_GENDER_TRACKING", raising=False)
    glossary_path = tmp_path / "Book_glossary.csv"
    tracker = _tracker(["male", "female"])
    tracker["version"] = 1
    del tracker["entries"]["루나"]["decision"]
    tracker_path = _write_tracker(glossary_path, tracker)

    assert glossary_extractor.set_gender_tracker_decisions(str(glossary_path), {"루나": "female"})
    updated = json.loads(tracker_path.read_text(encoding="utf-8"))
    assert updated["version"] == 2
    assert updated["entries"]["루나"]["decision"] == "female"
    assert len(updated["entries"]["루나"]["occurrences"]) == 2


@pytest.mark.parametrize("glossary_format", ["token", "legacy", "json"])
def test_compression_materializes_chapter_gender_from_one_stored_row(
    tmp_path,
    monkeypatch,
    glossary_format,
):
    monkeypatch.setenv("GLOSSARY_GENDER_NOISE_THRESHOLD", "0")
    monkeypatch.setenv("GLOSSARY_GENDER_TRACKING_BIAS", "none")
    glossary_path = tmp_path / "Book_glossary.csv"
    _write_tracker(glossary_path, _tracker(["male", "male", "female"]))

    if glossary_format == "token":
        content = (
            "Glossary Columns: raw_name, translated_name, gender, description\n\n"
            "=== CHARACTERS ===\n* 루나 = Luna [male]: Hero"
        )
        result = compress_glossary(
            content, "루나", glossary_format="csv", glossary_path=str(glossary_path),
            chapter_ref={"chapter_num": 3},
        )
        assert "[female]" in result
        assert "[male]" not in result
    elif glossary_format == "legacy":
        content = "type,raw_name,translated_name,gender\ncharacter,루나,Luna,male"
        result = compress_glossary(
            content, "루나", glossary_format="csv", glossary_path=str(glossary_path),
            chapter_ref={"chapter_num": 3},
        )
        assert result.splitlines()[-1].endswith(",female")
    else:
        content = [{"type": "character", "raw_name": "루나", "translated_name": "Luna", "gender": "male"}]
        result = compress_glossary(
            content, "루나", glossary_format="json", glossary_path=str(glossary_path),
            chapter_ref={"chapter_num": 3},
        )
        assert result[0]["gender"] == "female"


@pytest.mark.parametrize("glossary_format", ["token", "legacy", "json"])
def test_manual_compression_decision_overrides_chapter_history(
    tmp_path,
    monkeypatch,
    glossary_format,
):
    monkeypatch.setenv("GLOSSARY_GENDER_NOISE_THRESHOLD", "0")
    glossary_path = tmp_path / "Book_glossary.csv"
    _write_tracker(glossary_path, _tracker(["male", "male", "female"], decision="male"))

    if glossary_format == "token":
        content = (
            "Glossary Columns: raw_name, translated_name, gender, description\n\n"
            "=== CHARACTERS ===\n* 루나 = Luna [female]: Hero"
        )
        result = compress_glossary(
            content, "루나", glossary_format="csv", glossary_path=str(glossary_path),
            chapter_ref={"chapter_num": 3},
        )
        assert "[male]" in result
        assert "[female]" not in result
    elif glossary_format == "legacy":
        content = "type,raw_name,translated_name,gender\ncharacter,루나,Luna,female"
        result = compress_glossary(
            content, "루나", glossary_format="csv", glossary_path=str(glossary_path),
            chapter_ref={"chapter_num": 3},
        )
        assert result.splitlines()[-1].endswith(",male")
    else:
        content = [{"type": "character", "raw_name": "루나", "translated_name": "Luna", "gender": "female"}]
        result = compress_glossary(
            content, "루나", glossary_format="json", glossary_path=str(glossary_path),
            chapter_ref={"chapter_num": 3},
        )
        assert result[0]["gender"] == "male"


def test_compression_keeps_single_carrier_row_when_stored_gender_is_rare(tmp_path, monkeypatch):
    monkeypatch.setenv("GLOSSARY_GENDER_NOISE_THRESHOLD", "10")
    monkeypatch.setenv("GLOSSARY_GENDER_TRACKING_BIAS", "none")
    glossary_path = tmp_path / "Book_glossary.csv"
    _write_tracker(glossary_path, _tracker(["male"] * 10 + ["female"]))
    content = "type,raw_name,translated_name,gender\ncharacter,루나,Luna,female"

    result = compress_glossary(
        content, "루나", glossary_format="csv", glossary_path=str(glossary_path),
        chapter_ref={"chapter_num": 11},
    )

    assert result.splitlines()[-1].endswith(",male")


@pytest.mark.skipif(_prepare_editor_gender_tracking is None, reason="PySide6 is not installed")
def test_editor_consolidates_legacy_rows_in_memory_without_rewriting(tmp_path, monkeypatch):
    monkeypatch.delenv("GLOSSARY_SKIP_GENDER_TRACKING", raising=False)
    glossary_path = tmp_path / "Book_glossary.csv"
    original_text = (
        "type,raw_name,translated_name,gender\n"
        "character,루나,Luna,male\n"
        "character,루나,Luna,female\n"
    )
    glossary_path.write_text(original_text, encoding="utf-8")
    _write_tracker(glossary_path, _tracker(["male", "female", "female"]))
    entries = [
        {"type": "character", "raw_name": "루나", "translated_name": "Luna", "gender": "male"},
        {"type": "character", "raw_name": "루나", "translated_name": "Luna", "gender": "female"},
    ]

    prepared, tracker, _tracker_path, collapsed = _prepare_editor_gender_tracking(
        entries,
        str(glossary_path),
        {"character": {"enabled": True, "has_gender": True}},
    )

    assert tracker is not None
    assert collapsed == 1
    assert len(prepared) == 1
    assert prepared[0]["gender"] == "female"
    assert glossary_path.read_text(encoding="utf-8") == original_text


@pytest.mark.skipif(_apply_editor_gender_presentation is None, reason="PySide6 is not installed")
def test_editor_gender_presentation_uses_pink_until_manually_resolved():
    QApplication.instance() or QApplication([])
    tracker_entry = _tracker_entry(["male"] * 9 + ["female"])
    item = QTreeWidgetItem(["1", "female"])

    unresolved = editor_gender_status(tracker_entry, "male", 10, "none")
    _apply_editor_gender_presentation(item, unresolved, 1)

    assert item.text(1) == "Male*"
    assert item.background(0).color().alpha() == 72

    tracker_entry["decision"] = "female"
    resolved = editor_gender_status(tracker_entry, "male", 10, "none")
    _apply_editor_gender_presentation(item, resolved, 1)

    assert item.text(1) == "Female"
    assert item.data(0, Qt.BackgroundRole) is None


@pytest.mark.skipif(_gender_resolution_summary is None, reason="PySide6 is not installed")
def test_gender_resolution_summary_includes_history_and_latest_five_flips():
    tracker_entry = _tracker_entry(["male", "female", "female"])
    tracker_entry["changes"] = [
        {"from": "male", "to": "female", "chapter_num": index, "chapter_file": f"c{index}.xhtml"}
        for index in range(1, 7)
    ]

    summary = _gender_resolution_summary(tracker_entry, "male", 10, "none")

    assert summary["calculated_auto_gender"] == "female"
    assert summary["total"] == 3
    assert summary["genders"]["male"]["count"] == 1
    assert summary["genders"]["female"]["count"] == 2
    assert summary["genders"]["male"]["first"]["chapter_num"] == 1
    assert summary["genders"]["female"]["last"]["chapter_num"] == 3
    assert summary["flip_count"] == 6
    assert [event["chapter_num"] for event in summary["latest_flips"]] == [2, 3, 4, 5, 6]


@pytest.mark.skipif(GlossaryManagerMixin is None, reason="PySide6 is not installed")
def test_double_click_routes_only_through_gender_resolver_before_normal_editing():
    calls = []

    class Dummy:
        glossary_column_fields = []

        @staticmethod
        def _open_gender_resolution_for_item(item, column, require_gender_column=False):
            calls.append((item, column, require_gender_column))
            return column == 4

    item = object()
    GlossaryManagerMixin._on_tree_double_click(Dummy(), item, 4)
    GlossaryManagerMixin._on_tree_double_click(Dummy(), item, 2)

    assert calls == [(item, 4, True), (item, 2, True)]


def test_editor_gender_resolution_is_wired_to_gender_double_click_and_context_menu():
    source = Path(glossary_extractor.__file__).with_name("GlossaryManager_GUI.py").read_text(encoding="utf-8")
    double_click_source = source[source.index("def _on_tree_double_click"):]

    assert "require_gender_column=True" in double_click_source
    assert '"Resolve Gender…"' in source
    assert "pink.setAlpha(72)" in source
    assert "Counts are unique chapter/file tracker observations" in source
