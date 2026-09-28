"""The completion gate must use the same refinement type scope as extraction."""

import glossary_usage
from glossary_translation_gate import refinement_complete


def test_completed_active_types_ignore_inactive_glossary_entries(tmp_path, monkeypatch):
    glossary_path = tmp_path / "glossary.json"
    glossary_path.write_text("[]", encoding="utf-8")
    monkeypatch.setattr(glossary_usage, "parse_glossary_file", lambda path: [
        {"type": "character"}, {"type": "terms"}, {"type": "items"},
        {"type": "organizations"},
    ])
    progress = {"refinement": {
        "type::character": {"entry_type": "character", "status": "completed"},
        "type::terms": {"entry_type": "terms", "status": "completed"},
    }}
    config = {
        "glossary_refinement_enabled": True,
        "glossary_refinement_type_mode": "all",
        "custom_entry_types": {
            "character": {"enabled": True},
            "term": {"enabled": True},
            "items": {"enabled": False},
            "organizations": {"enabled": False},
        },
    }

    assert refinement_complete(progress, str(glossary_path), config) == (True, "complete")

    config["custom_entry_types"]["items"]["enabled"] = True
    assert refinement_complete(progress, str(glossary_path), config) == (
        False, "refinement is incomplete for item"
    )


def test_selected_refinement_scope_ignores_unselected_active_types(tmp_path, monkeypatch):
    glossary_path = tmp_path / "glossary.json"
    glossary_path.write_text("[]", encoding="utf-8")
    monkeypatch.setattr(glossary_usage, "parse_glossary_file", lambda path: [
        {"type": "character"}, {"type": "items"},
    ])
    config = {
        "glossary_refinement_enabled": True,
        "glossary_refinement_type_mode": "selected",
        "glossary_refinement_selected_types": ["character"],
        "custom_entry_types": {
            "character": {"enabled": True},
            "items": {"enabled": True},
        },
    }
    progress = {"refinement": {
        "type::character": {"status": "completed"},
    }}

    assert refinement_complete(progress, str(glossary_path), config) == (True, "complete")
