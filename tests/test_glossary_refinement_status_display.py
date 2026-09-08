"""Keep completed refinement types visible and saved when another type stops."""

import ast
import copy
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

import chapter_splitter
import extract_glossary_from_epub as extractor
import glossary_refinement
import Retranslation_GUI as gui_module


@pytest.fixture(scope="module")
def glossary_progress_codes():
    source_path = Path(gui_module.__file__)
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    functions = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name not in {
            "_gp_refinement_rows", "_gp_apply_mark_completed_to_progress",
            "_apply_gp_mark_completed_result",
        }:
            continue
        target = node
        if node.name == "_apply_gp_mark_completed_result":
            target = ast.parse(
                "def _make_handler():\n    gp_data = {}\n    return _apply_gp_mark_completed_result"
            ).body[0]
            target.body.insert(1, node)
        functions[node.name] = compile(
            ast.fix_missing_locations(ast.Module(body=[target], type_ignores=[])),
            str(source_path), "exec",
        )
    return functions


@pytest.fixture(scope="module")
def refinement_rows_code(glossary_progress_codes):
    return glossary_progress_codes["_gp_refinement_rows"]


@pytest.mark.parametrize(
    ("pending_status", "aggregate_status"),
    [("in_progress", "partially_in_progress"), ("failed", "refine_failed")],
)
def test_completed_refinement_rows_do_not_inherit_another_types_interruption(
    refinement_rows_code, pending_status, aggregate_status,
):
    counts = {
        "character": 331,
        "terms": 2007,
        "concept": 6,
        "equipment": 8,
        "item": 6,
        "locations": 215,
        "nicknames": 128,
        "skill": 21,
        "surnames": 85,
        "titles": 163,
    }
    aggregate_key = f"all::{','.join(counts)}"
    expected = {
        aggregate_key: {
            "entry_type": "All Entry Types",
            "is_aggregate": True,
            "selected_types": list(counts),
            "entry_count_before": sum(counts.values()),
            "current_entry_count": sum(counts.values()),
            "status": "not_refined",
        },
    }
    progress = {
        aggregate_key: {
            "status": pending_status,
            "completed_chunks": 1,
            "total_chunks": 2,
        },
    }
    for entry_type, count in counts.items():
        key = f"type::{entry_type}"
        expected[key] = {
            "entry_type": entry_type,
            "entry_count_before": count,
            "current_entry_count": count,
            "status": "not_refined",
        }
        unfinished = entry_type in {"character", "surnames"}
        progress[key] = {
            "entry_type": entry_type,
            "status": pending_status if unfinished else "completed",
            "model_name": "gemini-2.5-pro",
            "entry_count_before": count,
            "entry_count_after": count,
            "completed_chunks": 0 if unfinished else 1,
            "total_chunks": 1,
        }
    original_progress = copy.deepcopy(progress)
    namespace = dict(vars(gui_module))
    namespace.update({
        "_glossary_refinement_expected_entries": lambda _entries: copy.deepcopy(expected),
        "_gp_glossary_entries": lambda _data: [],
        "_refinement_type_key": gui_module._glossary_refinement_type_key,
    })
    exec(refinement_rows_code, namespace)

    rows = {
        key: (display, status)
        for key, display, status in namespace["_gp_refinement_rows"]({"refinement": progress})
    }

    assert rows[aggregate_key][1] == aggregate_status
    for entry_type in counts:
        display, status = rows[f"type::{entry_type}"]
        if entry_type in {"character", "surnames"}:
            assert status == ("refine_failed" if pending_status == "failed" else pending_status)
            assert "chunks 0/1" in display
        else:
            assert status == "completed"
            assert "Completed" in display
            assert "Refine Failed" not in display
            assert "chunks" not in display
    assert progress == original_progress


@pytest.mark.parametrize("latest_status", ["completed", "failed"])
def test_refinement_rows_use_latest_state_across_type_aliases(refinement_rows_code, latest_status):
    expected = {
        "type::terms": {
            "entry_type": "terms", "status": "not_refined",
            "entry_count_before": 1, "current_entry_count": 1,
        },
    }
    progress = {
        "type::terms": {
            "entry_type": "terms", "last_updated": 100,
            "status": "failed" if latest_status == "completed" else "completed",
        },
        "type::term": {
            "entry_type": "term", "last_updated": 200, "status": latest_status,
        },
    }
    namespace = dict(vars(gui_module))
    namespace.update({
        "_glossary_refinement_expected_entries": lambda _entries: expected,
        "_gp_glossary_entries": lambda _data: [],
    })
    exec(refinement_rows_code, namespace)

    rows = namespace["_gp_refinement_rows"]({"refinement": progress})

    assert rows[0][0] == "type::terms"
    assert rows[0][2] == ("refine_failed" if latest_status == "failed" else latest_status)


@pytest.mark.parametrize("changed", [False, True])
@pytest.mark.parametrize("save_legacy_json", [False, True])
def test_manual_stop_saves_completed_type_results_only_when_changed(
    monkeypatch, tmp_path, changed, save_legacy_json,
):
    monkeypatch.setattr(os, "environ", dict(os.environ))
    gui = gui_module.RetranslationMixin()
    gui.config = {"glossary_output_legacy_json": save_legacy_json}
    gui.model_var = "gemini-2.5-pro"
    gui.stop_requested = False
    logs = []
    gui.append_log = logs.append
    entries = [
        {"type": "character", "raw_name": "A", "translated_name": "A"},
        {"type": "surnames", "raw_name": "B", "translated_name": "B"},
        {"type": "titles", "raw_name": "C", "translated_name": "original title"},
    ]
    saved_csv = []
    saved_json = []
    monkeypatch.setattr(gui_module, "parse_glossary_file", lambda _path: entries)
    monkeypatch.setattr(extractor, "create_client_with_multi_key_support", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(extractor, "_effective_glossary_output_limit", lambda *_args: 2048)
    monkeypatch.setattr(extractor, "_compute_safe_input_tokens", lambda *_args: 1024)
    monkeypatch.setattr(extractor, "is_stop_requested", lambda: False)
    monkeypatch.setattr(chapter_splitter, "ChapterSplitter", lambda **_kwargs: object())
    monkeypatch.setattr(extractor, "save_glossary_csv", lambda result, path: saved_csv.append((copy.deepcopy(result), path)))
    monkeypatch.setattr(extractor, "save_glossary_json", lambda result, path: saved_json.append((copy.deepcopy(result), path)))

    def refine(result, **_kwargs):
        gui.stop_requested = True
        if changed:
            result[-1]["translated_name"] = "refined title"
        return result

    monkeypatch.setattr(glossary_refinement, "refine_glossary_entries", refine)
    glossary_path = str(tmp_path / "book.csv")
    json_path = str(tmp_path / "book.json")

    gui._run_manual_glossary_refinement(
        glossary_path,
        str(tmp_path / "book_glossary_progress.json"),
        SimpleNamespace(chunking_mode="all"),
        None,
    )

    expected_saves = [(entries, json_path)] if changed else []
    assert saved_csv == expected_saves
    assert saved_json == (expected_saves if save_legacy_json else [])
    if changed:
        assert "completed entry types saved" in logs[-1]
    else:
        assert "saved glossary was left unchanged" in logs[-1]


@pytest.mark.parametrize("mark_all", [False, True])
@pytest.mark.parametrize("existing_refinement", [False, True])
def test_marking_refinement_completed_persists_current_identities_for_auto_resume(
    glossary_progress_codes, monkeypatch, tmp_path, mark_all, existing_refinement,
):
    import json

    monkeypatch.setenv("BATCH_TRANSLATION", "0")
    monkeypatch.setenv("GLOSSARY_CUSTOM_FIELDS", "[]")
    monkeypatch.setenv("GLOSSARY_REFINEMENT_SKIP_DEDUPE", "1")
    monkeypatch.setenv("GLOSSARY_REFINEMENT_SYSTEM_PROMPT", "Refine {fields} for {entries}.")
    monkeypatch.setenv("GLOSSARY_REFINEMENT_USER_PROMPT", "")
    entry_types = ["terms", "character", "surnames"]
    entries = [
        {"type": entry_type, "raw_name": f"source {entry_type}", "translated_name": entry_type}
        for entry_type in entry_types
    ]
    # Configured type names and persisted glossary section headings may differ.
    entries[0]["type"] = "term"
    aggregate_key = f"all::{','.join(entry_types)}"
    expected = {
        aggregate_key: {
            "entry_type": "All Entry Types", "is_aggregate": True,
            "selected_types": entry_types, "status": "not_refined",
        },
        **{
            f"type::{entry_type}": {"entry_type": entry_type, "status": "not_refined"}
            for entry_type in entry_types
        },
    }
    original = {
        "refinement": {
            key: {
                **info, "status": "failed", "total_chunks": 2, "completed_chunks": 0,
                "entry_count_before": 90, "entry_count_after": 80,
                "input_identity_hash": "stale input", "output_identity_hash": "stale output",
                "error": "cancelled", "reason": "no_entries",
                "legacy_identity_hashes": ["old completed source set"],
                "legacy_identity_entry_type": "old type",
                "legacy_identity_hash_mode": "separate:unit_separator",
            }
            for key, info in expected.items()
        },
    }
    if not existing_refinement:
        original = {}
    progress_path = tmp_path / "glossary_progress.json"
    progress_path.write_text(json.dumps(original), encoding="utf-8")
    output_path = str(tmp_path / "glossary.csv")
    config = {"glossary_refinement_chunking_mode": "all"}
    namespace = dict(vars(gui_module))
    namespace.update({
        "self": SimpleNamespace(config=config),
        "panel_state": {"_glossary_path": output_path},
        "_gp_load_progress_dict": lambda path: json.loads(Path(path).read_text(encoding="utf-8")),
        "_gp_glossary_entries": lambda _data: copy.deepcopy(entries),
        "_glossary_refinement_expected_entries": lambda _entries: copy.deepcopy(expected),
        "_refinement_type_key": gui_module._glossary_refinement_type_key,
        "_gp_row_updates_for_targets": lambda *_args: {},
        "_gp_stats_for_dict": lambda *_args: {},
    })
    exec(glossary_progress_codes["_gp_apply_mark_completed_to_progress"], namespace)

    result = namespace["_gp_apply_mark_completed_to_progress"](
        str(progress_path), [("refinement", aggregate_key if mark_all else "type::terms")],
    )

    assert result["changed"]
    assert result["refresh_refinement_rows"] is True
    saved = json.loads(progress_path.read_text(encoding="utf-8"))["refinement"]
    marked_types = entry_types if mark_all else ["terms"]
    for entry_type in marked_types:
        info = saved[f"type::{entry_type}"]
        assert info["status"] == "completed"
        assert info["manually_marked_completed"] is True
        assert info["entry_count_before"] == info["entry_count_after"] == 1
        if existing_refinement:
            assert info["completed_chunks"] == info["total_chunks"] == 2
        assert info["identity_hash_version"] == glossary_refinement._IDENTITY_HASH_VERSION
        assert info["input_identity_hash"] == info["output_identity_hash"]
        assert info["input_identity_hash"] not in {"stale input", "stale output"}
        assert "error" not in info and "reason" not in info
        assert not any(key.startswith("legacy_identity_") for key in info)
    if mark_all:
        assert saved[aggregate_key]["entry_count_after"] == len(entries)
    else:
        for entry_type in ["character", "surnames"]:
            if existing_refinement:
                assert saved[f"type::{entry_type}"] == original["refinement"][f"type::{entry_type}"]
            else:
                assert f"type::{entry_type}" not in saved

    requests = []

    def send(messages, *_args, **_kwargs):
        requests.append(messages)
        return json.dumps(entries[1:]), "stop", None

    glossary_refinement.refine_glossary_entries(
        entries,
        client=SimpleNamespace(model="test-model"), temp=0, mtoks=1000,
        check_stop=lambda: False, chapter_splitter=SimpleNamespace(count_tokens=len),
        available_tokens=10000, chunk_timeout=5, parse_response_fn=json.loads,
        dedupe_fn=lambda value: value,
        custom_entry_types_fn=lambda: {entry_type: {"enabled": True} for entry_type in entry_types},
        send_fn=send, progress_file=str(progress_path), output_path=output_path,
        log=lambda _message: None,
        options=glossary_refinement.RefinementRunOptions(
            selected_types=entry_types, chunking_mode="all", run_when_disabled=True,
        ),
    )

    assert len(requests) == (0 if mark_all else 1)
    if requests:
        assert "source terms" not in requests[0][-1]["content"]
        assert "source character" in requests[0][-1]["content"]
        assert "source surnames" in requests[0][-1]["content"]


def test_manual_completion_immediately_refreshes_all_refinement_rows(glossary_progress_codes):
    calls = []
    updated_data = {"refinement": {"type::terms": {"status": "completed"}}}
    namespace = dict(vars(gui_module))
    namespace.update({
        "panel_state": {},
        "gp_listbox": SimpleNamespace(
            setUpdatesEnabled=lambda enabled: calls.append(("updates_enabled", enabled)),
            viewport=lambda: SimpleNamespace(update=lambda: calls.append(("paint",))),
        ),
        "_refresh_refinement_rows": lambda data, keep_updates_disabled: calls.append(
            ("refresh_all", data, keep_updates_disabled)
        ),
        "_apply_gp_stats": lambda _stats: None,
    })
    exec(glossary_progress_codes["_apply_gp_mark_completed_result"], namespace)
    apply_result = namespace["_make_handler"]()

    apply_result({
        "data": updated_data, "refresh_refinement_rows": True,
        "row_updates": {}, "stats": {},
    })

    assert calls == [
        ("refresh_all", updated_data, True),
        ("updates_enabled", True),
        ("paint",),
    ]
