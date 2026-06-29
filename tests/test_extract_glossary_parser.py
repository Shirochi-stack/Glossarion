import json

from extract_glossary_from_epub import parse_api_response, skip_duplicate_entries


def _set_glossary_env(monkeypatch):
    monkeypatch.setenv("GLOSSARY_CUSTOM_FIELDS", json.dumps(["description"]))
    monkeypatch.setenv(
        "GLOSSARY_CUSTOM_ENTRY_TYPES",
        json.dumps(
            {
                "item": {"enabled": True, "has_gender": False},
                "system_term": {"enabled": True, "has_gender": False},
                "title": {"enabled": True, "has_gender": True},
                "character": {"enabled": True, "has_gender": True},
            }
        ),
    )
    monkeypatch.setenv("GLOSSARY_ENTRY_TYPE_FILTER_MODE", "strict")


def test_no_header_non_gender_description_with_and_without_blank_gender(monkeypatch):
    _set_glossary_env(monkeypatch)
    response = "\n".join(
        [
            'item,\ub9c8\uc815\uc11d,Mana Stone,"Crystals containing magical energy."',
            'item,\ub9c8\uc774\ud06c,Microphone,,"A device created by Damian."',
            'system_term,\ud06c\ub85c\uc2dc\uc548\ub825,Crossian Calendar,,"The chronological system used within the Crossian Principality."',
        ]
    )

    entries = parse_api_response(response)

    assert [entry["translated_name"] for entry in entries] == [
        "Mana Stone",
        "Microphone",
        "Crossian Calendar",
    ]
    assert [entry["description"] for entry in entries] == [
        "Crystals containing magical energy.",
        "A device created by Damian.",
        "The chronological system used within the Crossian Principality.",
    ]


def test_no_header_gender_type_description_without_gender_column(monkeypatch):
    _set_glossary_env(monkeypatch)
    response = 'title,\uc18c\ub4dc\ub9c8\uc2a4\ud130,Sword Master,"The pinnacle of swordsmanship."'

    entries = parse_api_response(response)

    assert len(entries) == 1
    assert entries[0]["gender"] == "Unknown"
    assert entries[0]["description"] == "The pinnacle of swordsmanship."


def test_no_header_gender_type_keeps_real_gender_before_description(monkeypatch):
    _set_glossary_env(monkeypatch)
    response = 'character,\ub2e4\ubbf8\uc548,Damian,male,"A creator of mana devices."'

    entries = parse_api_response(response)

    assert len(entries) == 1
    assert entries[0]["gender"] == "male"
    assert entries[0]["description"] == "A creator of mana devices."


def test_dedup_keeps_later_description_when_description_is_active(monkeypatch):
    _set_glossary_env(monkeypatch)
    monkeypatch.setenv("GLOSSARY_USE_ADVANCED_DETECTION", "0")
    monkeypatch.setenv("GLOSSARY_DEDUPE_TRANSLATIONS", "0")

    entries = [
        {"type": "item", "raw_name": "\ub9c8\uc774\ud06c", "translated_name": "Microphone"},
        {
            "type": "item",
            "raw_name": "\ub9c8\uc774\ud06c",
            "translated_name": "Microphone",
            "description": "A device created by Damian.",
        },
    ]

    result = skip_duplicate_entries(entries)

    assert len(result) == 1
    assert result[0]["description"] == "A device created by Damian."
