import json

import pytest

from glossary_compressor import compress_glossary


TOKEN_GLOSSARY = (
    "Glossary Columns: raw_name, translated_name, gender, description\n\n"
    "=== CHARACTERS ===\n"
    "* 루나 = Luna [female]: Hero\n"
    "* 카일 = Kyle [male]: Knight\n"
    "=== TERMS ===\n"
    "* 마법 = Magic: Power"
)
LEGACY_GLOSSARY = (
    "type,raw_name,translated_name,gender\n"
    "character,루나,Luna,female\n"
    "character,카일,Kyle,male\n"
    "term,마법,Magic,"
)
JSON_GLOSSARY = [
    {"type": "character", "raw_name": "루나", "translated_name": "Luna", "gender": "female"},
    {"type": "character", "raw_name": "카일", "translated_name": "Kyle", "gender": "male"},
    {"type": "term", "raw_name": "마법", "translated_name": "Magic"},
]

RAW_SOURCE = "루나와 카일은 마법을 배웠다."
# Luna and Magic are applied; 카일 was left untranslated.
TRANSLATED_OUTPUT = "<p>Luna learned magic with 카일.</p>"


def _names(result):
    if isinstance(result, list):
        return {entry["translated_name"] for entry in result}
    return {
        line.split("=")[1].split("[")[0].split(":")[0].strip()
        if "* " in line else line.split(",")[2]
        for line in result.splitlines()
        if line.startswith("* ") or line.startswith(("character,", "term,"))
    }


@pytest.mark.parametrize("engine", ["new", "legacy"])
@pytest.mark.parametrize(
    "content,fmt",
    [(TOKEN_GLOSSARY, "csv"), (LEGACY_GLOSSARY, "csv"), (JSON_GLOSSARY, "json")],
)
def test_translated_text_drops_only_applied_entries(monkeypatch, engine, content, fmt):
    monkeypatch.setenv("GLOSSARY_MATCH_ENGINE", engine)
    monkeypatch.setenv("GLOSSARY_SKIP_GENDER_TRACKING", "1")
    stats = {}

    result = compress_glossary(
        content, RAW_SOURCE, glossary_format=fmt,
        translated_text=TRANSLATED_OUTPUT, stats=stats,
    )

    assert _names(result) == {"Kyle"}
    assert stats["excluded_applied"] == 2


@pytest.mark.parametrize(
    "content,fmt",
    [(TOKEN_GLOSSARY, "csv"), (LEGACY_GLOSSARY, "csv"), (JSON_GLOSSARY, "json")],
)
def test_without_translated_text_every_raw_match_is_kept(monkeypatch, content, fmt):
    monkeypatch.setenv("GLOSSARY_SKIP_GENDER_TRACKING", "1")
    stats = {}

    result = compress_glossary(content, RAW_SOURCE, glossary_format=fmt, stats=stats)

    assert _names(result) == {"Luna", "Kyle", "Magic"}
    assert stats["excluded_applied"] == 0


def test_all_applied_sends_no_glossary_instead_of_relaxing(monkeypatch, capsys):
    monkeypatch.setenv("GLOSSARY_MATCH_ENGINE", "new")
    monkeypatch.setenv("GLOSSARY_SKIP_GENDER_TRACKING", "1")
    stats = {}

    result = compress_glossary(
        LEGACY_GLOSSARY, RAW_SOURCE, glossary_format="csv",
        translated_text="<p>Luna and Kyle learned magic.</p>", stats=stats,
    )

    assert result == ""
    assert stats["excluded_applied"] == 3
    assert "already applied" in capsys.readouterr().out
    assert "relaxed" not in capsys.readouterr().out


def test_latin_translation_needs_a_word_boundary(monkeypatch):
    monkeypatch.setenv("GLOSSARY_SKIP_GENDER_TRACKING", "1")

    result = compress_glossary(
        JSON_GLOSSARY, "카일", glossary_format="json",
        translated_text="<p>Kyleford is a town.</p>",
    )

    assert _names(result) == {"Kyle"}


def test_raw_not_in_source_is_dropped_before_the_applied_check(monkeypatch):
    monkeypatch.setenv("GLOSSARY_SKIP_GENDER_TRACKING", "1")
    stats = {}

    result = compress_glossary(
        JSON_GLOSSARY, "루나", glossary_format="json",
        translated_text="<p>Kyle</p>", stats=stats,
    )

    assert _names(result) == {"Luna"}
    assert stats["excluded_applied"] == 0
    assert json.dumps(result, ensure_ascii=False)
