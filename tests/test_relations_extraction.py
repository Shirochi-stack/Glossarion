"""Relations Tracker, extraction side (extract_glossary_from_epub, GlossaryManager, glossary_document).

No network: build_prompt, parse_api_response, the writers, the tracker write side and the
Minimal response path are called directly with temp folders and a patched environment.

Run (repository root)::

    python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic \\
        tests/test_relations_extraction.py
"""

from __future__ import annotations

import csv
import json
import os
import re
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import relations_tracking as rt  # noqa: E402

_ENV_KEYS = (
    rt.ENV_ENABLE, rt.ENV_FIELD, rt.ENV_HISTORY_MODE,
    "GLOSSARY_SYSTEM_PROMPT", "GLOSSARY_CUSTOM_FIELDS", "GLOSSARY_CUSTOM_ENTRY_TYPES",
    "GLOSSARY_TARGET_LANGUAGE", "GLOSSARY_INCLUDE_GENDER_CONTEXT", "GLOSSARY_INCLUDE_DESCRIPTION",
    "GLOSSARY_USE_LEGACY_CSV", "GLOSSARY_OUTPUT_LEGACY_JSON", "GLOSSARY_INCLUDE_BOOK_TITLE",
    "GLOSSARY_SKIP_GENDER_TRACKING", "GLOSSARY_OUTPUT_BACKUP_DIR", "GLOSSARY_SKIP_ALL_VALIDATION",
    "GLOSSARY_SKIP_IDENTICAL_ENTRIES", "GLOSSARY_CJK_SCRIPT_FILTER", "EPUB_PATH",
)


@pytest.fixture(autouse=True)
def _env(tmp_path):
    """Isolated env (own MonkeyPatch so it holds for the whole test); Relations Tracker off."""
    mp = pytest.MonkeyPatch()
    for key in _ENV_KEYS:
        mp.delenv(key, raising=False)
    for key, sub in (("GLOSSARION_APP_DIR", "_app"), ("GLOSSARION_LIBRARY_DIR", "_lib"),
                     ("OUTPUT_DIRECTORY", "_out")):
        (tmp_path / sub).mkdir(exist_ok=True)
        mp.setenv(key, str(tmp_path / sub))
    mp.setenv("GLOSSARY_INCLUDE_BOOK_TITLE", "0")
    mp.setenv("GLOSSARY_CUSTOM_FIELDS", json.dumps(["description"]))
    try:
        yield mp
    finally:
        mp.undo()


@pytest.fixture(scope="module")
def eg():
    import extract_glossary_from_epub
    return extract_glossary_from_epub


@pytest.fixture(scope="module")
def gm():
    import GlossaryManager
    return GlossaryManager


def _on(mp, field=None):
    mp.setenv(rt.ENV_ENABLE, "1")
    if field:
        mp.setenv(rt.ENV_FIELD, field)


def _without_relations_line(text):
    return text.replace("- {relations_instruction}\n", "")


def _tracker_without_time(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    data.pop("updated_at", None)
    return data


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

def test_default_prompts_carry_the_placeholder_under_subject_tracking(eg):
    for prompt in (eg.DEFAULT_GLOSSARY_PROMPT, eg.DEFAULT_AUTO_GLOSSARY_PROMPT):
        lines = prompt.split("\n")
        idx = lines.index("- {relations_instruction}")
        assert "Subject Tracking" in lines[idx - 1] or lines[idx - 1] == "- {subject_tracking_instruction}"
        assert prompt.count(rt.RELATIONS_PLACEHOLDER) == 1


@pytest.mark.parametrize("custom_fields", [["description"], [], ["description", "notes"]])
@pytest.mark.parametrize("gender_ctx", ["0", "1"])
def test_build_prompt_is_identical_with_toggle_off(eg, _env, monkeypatch, custom_fields, gender_ctx):
    _env.setenv("GLOSSARY_CUSTOM_FIELDS", json.dumps(custom_fields))
    _env.setenv("GLOSSARY_INCLUDE_GENDER_CONTEXT", gender_ctx)
    system_now, user_now = eg.build_prompt("서윤은 도현을 보았다.")
    monkeypatch.setattr(eg, "DEFAULT_GLOSSARY_PROMPT", _without_relations_line(eg.DEFAULT_GLOSSARY_PROMPT))
    system_before, user_before = eg.build_prompt("서윤은 도현을 보았다.")
    assert system_now == system_before
    assert user_now == user_before
    assert rt.RELATIONS_PLACEHOLDER not in system_now and "`relations`" not in system_now
    assert not re.search(r"\brelations\b", system_now)


def test_custom_prompts_without_the_placeholder_are_untouched_either_way(eg, _env):
    _env.setenv("GLOSSARY_SYSTEM_PROMPT", "Extract names as CSV.\n{fields}\n- keep it short")
    off, _ = eg.build_prompt("x")
    _on(_env)
    on, _ = eg.build_prompt("x")
    # The relations column is still requested through {fields}, but no instruction is added.
    assert rt.RELATIONS_PLACEHOLDER not in on and "The `relations` column" not in on
    assert off.replace("description", "description, relations") == on


def test_minimal_prompt_line_vanishes_when_off(eg):
    prompt = eg.DEFAULT_AUTO_GLOSSARY_PROMPT
    assert eg._apply_relations_placeholder(prompt, enabled=False) == _without_relations_line(prompt)
    assert eg._apply_relations_placeholder(prompt) == _without_relations_line(prompt)


def test_build_prompt_with_toggle_on_has_instruction_and_column(eg, _env):
    _on(_env)
    _env.setenv("GLOSSARY_TARGET_LANGUAGE", "Spanish")
    system, _ = eg.build_prompt("text")
    assert rt.RELATIONS_PLACEHOLDER not in system
    assert "The `relations` column: for character entries only" in system
    assert "in Spanish, 1–3 words" in system
    assert "extra LAST column" not in system  # Balanced/Full: the column comes from {fields}
    assert "Columns:\ntype, raw_name, translated_name, gender, description, relations" in system
    # The instruction sits on its own bullet line right under subject tracking.
    lines = system.split("\n")
    idx = next(i for i, line in enumerate(lines) if line.startswith("- The `relations` column"))
    assert "Subject Tracking" in lines[idx - 1]


def test_build_prompt_keeps_a_relations_field_once_and_uses_its_name(eg, _env):
    _on(_env, field="Bonds")
    _env.setenv("GLOSSARY_CUSTOM_FIELDS", json.dumps(["description", "bonds"]))
    system, _ = eg.build_prompt("text")
    # build_prompt joins the {fields} parts with a literal backslash-n
    header = re.search(r"Columns:\n(.+?)(?:\\n|\n)", system).group(1)
    assert header == "type, raw_name, translated_name, gender, description, bonds"
    assert "The `Bonds` column" in system


def _instruction_example_rows(system):
    line = next(l for l in system.split("\n") if l.startswith("Example rows that mirror each other"))
    return [next(csv.reader([row])) for row in re.findall(r"`([^`]+)`", line)]


def test_build_prompt_examples_put_relations_in_their_own_column(eg, _env):
    """With description on, every other example row ends at the description; a relations
    example of the same length made the model write the relations there."""
    _on(_env)
    _env.setenv("GLOSSARY_CUSTOM_FIELDS", json.dumps(["description", "relations"]))
    system, _ = eg.build_prompt("text")
    assert ("`relations` is its own column, column 6 of 6 (type, raw_name, translated_name, gender, "
            "description, relations): never write relationships in another column.") in system
    rows = _instruction_example_rows(system)
    assert [len(row) for row in rows] == [6, 6]
    assert rows[0][5] == "뢤사난: Older Sister" and ":" not in rows[0][4]

    # Without a description field the relations are the fifth column, and the rows say so.
    _env.setenv("GLOSSARY_CUSTOM_FIELDS", json.dumps([]))
    system, _ = eg.build_prompt("text")
    assert "column 5 of 5 (type, raw_name, translated_name, gender, relations)" in system
    assert [len(row) for row in _instruction_example_rows(system)] == [5, 5]


def test_instruction_example_names_are_the_prompt_example_names(eg):
    """Made-up names only: a real-looking example name came back as a glossary entry."""
    for example in rt._EXAMPLE_ROWS:
        assert (f"character,{example['raw_name']},{example['translated_name']},{example['gender']},"
                in eg._DESCRIPTION_EXAMPLE_TEXT)


def test_minimal_prompt_declares_the_last_column(eg, _env):
    _on(_env)
    text = eg._apply_relations_placeholder(eg.DEFAULT_AUTO_GLOSSARY_PROMPT, declare_column=True)
    assert "- Add a `relations` column as one extra LAST column of every row. The `relations` column" in text


def test_warning_once_when_on_and_placeholder_missing(eg, _env, capsys):
    assert eg.warn_missing_relations_placeholder("no placeholder") is False
    _on(_env)
    assert eg.warn_missing_relations_placeholder(eg.DEFAULT_GLOSSARY_PROMPT) is False
    assert eg.warn_missing_relations_placeholder("no placeholder") is True
    out = capsys.readouterr().out
    assert out.count("Relations Tracker is on") == 1 and "- {relations_instruction}" in out


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

RESPONSE = (
    "type,raw_name,translated_name,gender,description,relations\n"
    'character,서윤,Seo-yun,female,"Heir of the Baek trading house",도현: Older Brother; 하린: Best Friend\n'
    'character,도현,Do-hyeon,male,"Royal guard captain",서윤: Younger Sister\n'
    'term,청월검,Azure Moon Sword,,"Heirloom blade",\n'
)


def test_parse_api_response_keeps_the_relations_column(eg, _env):
    _on(_env)
    entries = eg.parse_api_response(RESPONSE)
    by_raw = {e["raw_name"]: e for e in entries}
    assert by_raw["서윤"]["relations"] == "도현: Older Brother; 하린: Best Friend"
    assert by_raw["도현"]["relations"] == "서윤: Younger Sister"


def test_parse_api_response_without_header_maps_the_appended_field(eg, _env):
    _on(_env)
    entries = eg.parse_api_response(
        'character,서윤,Seo-yun,female,"Heir",도현: Older Brother\n'
        'character,도현,Do-hyeon,male,"Captain"\n'
    )
    by_raw = {e["raw_name"]: e for e in entries}
    assert by_raw["서윤"].get("relations") == "도현: Older Brother"
    assert "relations" not in by_raw["도현"]


# ---------------------------------------------------------------------------
# Never persisted
# ---------------------------------------------------------------------------

def _entries():
    return [
        {"type": "character", "raw_name": "서윤", "translated_name": "Seo-yun", "gender": "female",
         "description": "Heir", "relations": "도현: Older Brother"},
        {"type": "character", "raw_name": "도현", "translated_name": "Do-hyeon", "gender": "male",
         "description": "Captain", "Relations": "서윤: Younger Sister"},
        {"type": "term", "raw_name": "청월검", "translated_name": "Azure Moon Sword", "description": "Blade"},
    ]


@pytest.mark.parametrize("legacy", ["0", "1"])
def test_glossary_writers_never_write_relations(eg, _env, tmp_path, legacy):
    _on(_env)
    _env.setenv("GLOSSARY_USE_LEGACY_CSV", legacy)
    _env.setenv("GLOSSARY_OUTPUT_LEGACY_JSON", "1")
    _env.setenv("GLOSSARY_CUSTOM_FIELDS", json.dumps(["description", "relations"]))
    out = tmp_path / "Book_glossary.csv"
    eg.save_glossary_csv(_entries(), str(out))
    eg.save_glossary_json(_entries(), str(tmp_path / "Book_glossary.json"))
    for path in (out, tmp_path / "Book_glossary.json"):
        text = path.read_text(encoding="utf-8")
        assert "Seo-yun" in text
        assert "relations" not in text.lower(), path.name
        assert "Older Brother" not in text and "Younger Sister" not in text


@pytest.mark.parametrize("legacy", ["0", "1"])
def test_writers_with_toggle_off_still_write_a_user_relations_field(eg, _env, tmp_path, legacy):
    """Off means off: a user's own 'relations' custom field is an ordinary field."""
    _env.setenv("GLOSSARY_USE_LEGACY_CSV", legacy)
    _env.setenv("GLOSSARY_CUSTOM_FIELDS", json.dumps(["description", "relations"]))
    out = tmp_path / "Book_glossary.csv"
    eg.save_glossary_csv(_entries(), str(out))
    text = out.read_text(encoding="utf-8")
    assert "relations" in text and "Older Brother" in text
    assert not (tmp_path / "Book_relations_tracker.json").exists()


def _doc(fmt, entries):
    return types.SimpleNamespace(
        current_glossary_data=entries,
        current_glossary_format=fmt,
        current_glossary_sections=[],
        _pending_gender_decisions={},
        _gender_variants_pending_save=0,
        current_gender_tracker_data=None,
        current_gender_tracker_path="",
    )


@pytest.mark.parametrize("fmt", ["token_csv", "list"])
def test_editor_save_never_writes_relations(_env, tmp_path, fmt):
    import glossary_document as gd
    config = {rt.CONFIG_ENABLE: True, "custom_glossary_fields": ["description", "relations"]}
    path = tmp_path / "Book_glossary.csv"
    assert gd.save_document(_doc(fmt, _entries()), str(path), config) is True
    text = path.read_text(encoding="utf-8")
    assert "Seo-yun" in text and "relations" not in text.lower() and "Older Brother" not in text
    assert config["custom_glossary_fields"] == ["description", "relations"]  # caller's config untouched


def test_editor_save_with_toggle_off_is_unchanged(_env, tmp_path):
    import glossary_document as gd
    config = {"custom_glossary_fields": ["description", "relations"]}
    path = tmp_path / "Book_glossary.csv"
    gd.save_document(_doc("token_csv", _entries()), str(path), config)
    assert "Older Brother" in path.read_text(encoding="utf-8")
    assert gd.relations_free_save_config(_doc("list", _entries()), config) is config


def test_saved_glossary_with_a_relations_column_still_loads(_env, tmp_path):
    import glossary_document as gd
    path = tmp_path / "Book_glossary.csv"
    path.write_text(
        "Glossary Columns: raw_name, translated_name, gender, description, relations\n\n"
        "=== CHARACTERS ===\n* 서윤 = Seo-yun [Female]: Heir (relations: 도현: Older Brother)\n",
        encoding="utf-8",
    )
    payload = gd.parse_glossary_file(str(path), {rt.CONFIG_ENABLE: True, "custom_glossary_fields": ["description"]})
    assert [e["raw_name"] for e in payload["entries"]] == ["서윤"]


# ---------------------------------------------------------------------------
# Tracker write side
# ---------------------------------------------------------------------------

def _chapter_entries(label):
    return [
        {"type": "character", "raw_name": "서윤", "translated_name": "Seo-yun", "relations": f"하린: {label}"},
        {"type": "character", "raw_name": "하린", "translated_name": "Ha-rin"},
    ]


def test_update_relations_tracker_is_off_by_default(eg, tmp_path):
    entries = _chapter_entries("Best Friend")
    before = json.dumps(entries, ensure_ascii=False)
    assert eg.update_relations_tracker(entries, str(tmp_path / "Book_glossary.csv"), chapter_num=1) == {}
    assert json.dumps(entries, ensure_ascii=False) == before
    assert eg.update_relations_tracker_for_group(entries, str(tmp_path / "Book_glossary.csv"), [], 0) == []
    assert json.dumps(entries, ensure_ascii=False) == before
    assert not any(rt.is_tracker_file(p) for p in tmp_path.rglob("*"))


def test_update_relations_tracker_writes_pops_and_is_order_independent(eg, _env, tmp_path, capsys):
    _on(_env)
    chapters = [(0, 1, "ch001.xhtml", "Best Friend"), (39, 40, "ch040.xhtml", "Enemy"),
                (44, 45, "ch045.xhtml", "Enemy")]
    results = {}
    for name, order in (("forward", chapters), ("backward", list(reversed(chapters)))):
        folder = tmp_path / name
        folder.mkdir()
        output = folder / "Book_glossary.csv"
        for index, number, filename, label in order:
            entries = _chapter_entries(label)
            eg.update_relations_tracker(entries, str(output), source_path="book.epub",
                                        chapter_index=index, chapter_num=number, chapter_file=filename)
            assert all("relations" not in e for e in entries)
        tracker_path = folder / "Book_relations_tracker.json"
        assert tracker_path.exists()
        data = _tracker_without_time(tracker_path)
        data.pop("glossary_path")
        results[name] = data
    assert results["forward"] == results["backward"]
    # 서윤's row says what 서윤 is to 하린: stored on 하린's entry (what 서윤 is to 하린).
    rel = results["forward"]["entries"]["하린"]["relations"]["서윤"]
    assert [o["label"] for o in rel["occurrences"]] == ["Best Friend", "Enemy", "Enemy"]
    assert rel["changes"][0]["from"] == "Best Friend" and rel["changes"][0]["chapter_num"] == 40
    assert "서윤" not in results["forward"]["entries"]
    out = capsys.readouterr().out
    assert "🔗 Relations tracker: Ch. 1 +1 new" in out
    # The log names the stored edge: on 하린's row, 서윤 went from Best Friend to Enemy.
    assert "🔗 Relations tracker: Ch. 40 1 changed (하린 → 서윤: Best Friend → Enemy)" in out


def test_no_tracker_file_without_relations(eg, _env, tmp_path):
    _on(_env)
    entries = [{"type": "term", "raw_name": "청월검", "translated_name": "Azure Moon Sword", "relations": ""}]
    eg.update_relations_tracker(entries, str(tmp_path / "Book_glossary.csv"), chapter_num=1)
    assert entries == [{"type": "term", "raw_name": "청월검", "translated_name": "Azure Moon Sword"}]
    assert not (tmp_path / "Book_relations_tracker.json").exists()


def test_merged_group_bucketing(eg, _env, tmp_path):
    _on(_env)
    units = [
        (10, "서윤은 오빠 도현과 함께 있었다."),
        (11, "서윤과 하린이 처음 만났다."),
        (12, "무진이 나타났다."),
    ]
    entries = [
        {"type": "character", "raw_name": "서윤", "translated_name": "Seo-yun",
         "relations": "도현: Older Brother; 하린: Best Friend; 무진: Father"},
        {"type": "character", "raw_name": "도현", "translated_name": "Do-hyeon"},
        {"type": "character", "raw_name": "하린", "translated_name": "Ha-rin"},
        {"type": "character", "raw_name": "설화", "translated_name": "Seol-hwa", "relations": "하린: Master"},
    ]
    output = tmp_path / "Book_glossary.csv"
    eg.update_relations_tracker_for_group(
        entries, str(output), units, 10,
        chapter_positions={10: 11, 11: 12, 12: 13},
        chapter_filenames={10: "c11.xhtml", 11: "c12.xhtml", 12: "c13.xhtml"},
    )
    assert all("relations" not in e for e in entries)
    data = _tracker_without_time(tmp_path / "Book_relations_tracker.json")
    entries = data["entries"]
    # Each item of 서윤's row is stored on the other character's entry, target 서윤.
    assert "서윤" not in entries
    # both names in the chapter -> that chapter
    assert entries["도현"]["relations"]["서윤"]["occurrences"][0]["chapter_num"] == 11
    assert entries["하린"]["relations"]["서윤"]["occurrences"][0]["chapter_num"] == 12
    assert entries["하린"]["relations"]["서윤"]["target_translated"] == "Seo-yun"
    assert entries["하린"]["translated_name"] == "Ha-rin"
    # target never shown with the source -> the first chapter with the source's own name
    assert entries["무진"]["relations"]["서윤"]["occurrences"][0]["chapter_num"] == 11
    assert entries["무진"]["relations"]["서윤"]["occurrences"][0]["chapter_file"] == "c11.xhtml"
    assert entries["무진"]["translated_name"] == ""  # no row: filled by the glossary sync later
    # source in no chapter text -> the parent chapter
    assert entries["하린"]["relations"]["설화"]["occurrences"][0]["chapter_index"] == 10


def test_each_chapters_description_is_recorded_beside_the_relations(eg, _env, tmp_path):
    """The diagram shows the description a character had at the chapter it looks at."""
    _on(_env)
    output = tmp_path / "Book_glossary.csv"
    tracker_path = tmp_path / "Book_relations_tracker.json"
    # No relation anywhere yet: descriptions alone do not start a tracker file.
    rows = [{"type": "character", "raw_name": "민", "translated_name": "Min", "description": "A playful hunter"}]
    eg.update_relations_tracker(rows, str(output), chapter_index=1, chapter_num=2, chapter_file="c2.xhtml")
    assert not tracker_path.exists()
    rows = [{"type": "character", "raw_name": "민", "translated_name": "Min",
             "description": "Team leader of the Night Wind clan", "relations": "진: Team Leader"},
            {"type": "character", "raw_name": "진", "translated_name": "Jin", "description": "A new sniper",
             "relations": "민: Subordinate"},
            {"type": "term", "raw_name": "마석", "translated_name": "Mana Stone", "description": "A crystal"}]
    eg.update_relations_tracker(rows, str(output), chapter_index=7, chapter_num=8, chapter_file="c8.xhtml")
    # The description stays on the rows (it is a glossary column); only relations are taken off.
    assert [r.get("description") for r in rows] == ["Team leader of the Night Wind clan", "A new sniper", "A crystal"]
    assert all("relations" not in r for r in rows)
    # A later chapter with a new description and no relations is recorded too, now that a tracker exists.
    rows = [{"type": "character", "raw_name": "민", "translated_name": "Min", "description": "Feared by the Red Tigers"}]
    eg.update_relations_tracker(rows, str(output), chapter_index=16, chapter_num=17, chapter_file="c17.xhtml")
    tracker = rt.load_tracker(str(tracker_path))
    assert [(d["chapter_num"], d["text"]) for d in rt.description_history(tracker, "민")] == [
        (8, "Team leader of the Night Wind clan"), (17, "Feared by the Red Tigers")]
    assert rt.description_at(tracker, "진", 8)["text"] == "A new sniper"
    assert "마석" not in tracker["entries"]
    # Nothing of it reaches the glossary file beyond its own description column.
    eg.save_glossary_csv([{"type": "character", "raw_name": "민", "translated_name": "Min", "gender": "female",
                           "description": "Feared by the Red Tigers"}], str(output))
    assert "Team leader of the Night Wind clan" not in output.read_text(encoding="utf-8")


def test_merged_group_files_a_description_under_the_chapter_that_names_the_entry(eg, _env, tmp_path):
    _on(_env)
    units = [(10, "서윤은 혼자였다."), (11, "서윤과 하린이 처음 만났다.")]
    entries = [
        {"type": "character", "raw_name": "서윤", "translated_name": "Seo-yun", "description": "The heir",
         "relations": "하린: Best Friend"},
        {"type": "character", "raw_name": "하린", "translated_name": "Ha-rin", "description": "A healer"},
        {"type": "character", "raw_name": "설화", "translated_name": "Seol-hwa", "description": "Named nowhere"},
    ]
    output = tmp_path / "Book_glossary.csv"
    eg.update_relations_tracker_for_group(
        entries, str(output), units, 10, chapter_positions={10: 11, 11: 12},
        chapter_filenames={10: "c11.xhtml", 11: "c12.xhtml"})
    assert [e["description"] for e in entries] == ["The heir", "A healer", "Named nowhere"]
    tracker = rt.load_tracker(str(tmp_path / "Book_relations_tracker.json"))
    assert [(d["chapter_num"], d["text"]) for d in rt.description_history(tracker, "서윤")] == [(11, "The heir")]
    assert [(d["chapter_num"], d["text"]) for d in rt.description_history(tracker, "하린")] == [(12, "A healer")]
    assert rt.description_history(tracker, "설화")[0]["chapter_num"] == 11  # the parent chapter
    # The relation itself is still filed under the chapter that shows both names.
    assert tracker["entries"]["하린"]["relations"]["서윤"]["occurrences"][0]["chapter_num"] == 12


def test_sync_after_dedup_rekeys_merged_names(eg, _env, tmp_path):
    _on(_env)
    output = tmp_path / "Book_glossary.csv"
    eg.update_relations_tracker(
        [{"raw_name": "서윤님", "translated_name": "Seo-yun", "relations": "도현: Older Brother"},
         {"raw_name": "하린", "translated_name": "Ha-rin", "relations": "서윤님: Best Friend"}],
        str(output), chapter_index=0, chapter_num=1, chapter_file="c1.xhtml",
    )
    # dedup kept "서윤" (honorific-stripped match); save_glossary_csv syncs the tracker
    eg.save_glossary_csv(
        [{"type": "character", "raw_name": "서윤", "translated_name": "Seo-yun", "gender": "female"},
         {"type": "character", "raw_name": "도현", "translated_name": "Do-hyeon", "gender": "male"},
         {"type": "character", "raw_name": "하린", "translated_name": "Ha-rin", "gender": "female"}],
        str(output),
    )
    data = _tracker_without_time(tmp_path / "Book_relations_tracker.json")
    assert "서윤님" not in data["entries"]
    assert "서윤님" in data["entries"]["서윤"]["aliases"]
    assert set(data["entries"]["서윤"]["relations"]) == {"하린"}
    assert set(data["entries"]["도현"]["relations"]) == {"서윤"}
    assert data["entries"]["도현"]["relations"]["서윤"]["target_raw"] == "서윤"


def test_sync_is_a_noop_when_off(eg, _env, tmp_path):
    _on(_env)
    output = tmp_path / "Book_glossary.csv"
    eg.update_relations_tracker([{"raw_name": "서윤님", "relations": "도현: Older Brother"}], str(output),
                                chapter_num=1)
    before = (tmp_path / "Book_relations_tracker.json").read_bytes()
    _env.delenv(rt.ENV_ENABLE)
    assert eg.sync_relations_tracker_with_glossary([{"raw_name": "서윤"}], str(output)) is None
    assert (tmp_path / "Book_relations_tracker.json").read_bytes() == before


def test_editor_save_syncs_the_tracker(eg, _env, tmp_path):
    import glossary_document as gd
    _on(_env)
    output = tmp_path / "Book_glossary.csv"
    eg.update_relations_tracker([{"raw_name": "도현", "relations": "서윤님: Older Brother"}], str(output),
                                chapter_num=1)
    _env.delenv(rt.ENV_ENABLE)  # the editor reads its config, not the environment
    entries = [{"type": "character", "raw_name": "서윤", "translated_name": "Seo-yun", "gender": "female"}]
    gd.save_document(_doc("token_csv", entries), str(output), {rt.CONFIG_ENABLE: True})
    data = _tracker_without_time(tmp_path / "Book_relations_tracker.json")
    assert list(data["entries"]) == ["서윤"]


# ---------------------------------------------------------------------------
# Minimal mode (GlossaryManager)
# ---------------------------------------------------------------------------

def _sink(gm, known=()):
    return {"field": "relations", "columns": list(gm._MINIMAL_DEFAULT_COLUMNS), "known": set(known), "cells": []}


def test_minimal_detaches_the_last_cell(gm, _env):
    sink = _sink(gm)
    lines = [
        "character,서윤,Seo-yun,female,Heir of the house,도현: Older Brother; 하린: Best Friend",
        "character,무진,Mu-jin,male,Head of the house",                     # no relations cell
        "character,하린,Ha-rin,female,Role: apprentice healer",             # description with a colon
        "term,청월검,Azure Moon Sword,,Heirloom blade,",                      # empty relations cell
        "character,도현,Do-hyeon,male,Guard, captain of the gate,서윤: Younger Sister",  # unquoted comma
        'character,설화,Seol-hwa,female,"A master, of herbs"',                # quoted comma, no relations
    ]
    out = gm._detach_relations_cells(lines, ",", sink)
    assert out == [
        "character,서윤,Seo-yun,female,Heir of the house",
        "character,무진,Mu-jin,male,Head of the house",
        "character,하린,Ha-rin,female,Role: apprentice healer",
        "term,청월검,Azure Moon Sword,,Heirloom blade",
        "character,도현,Do-hyeon,male,Guard, captain of the gate",
        'character,설화,Seol-hwa,female,"A master, of herbs"',
    ]
    assert sink["cells"] == [
        ("서윤", "Seo-yun", "도현: Older Brother; 하린: Best Friend"),
        ("도현", "Do-hyeon", "서윤: Younger Sister"),
    ]


def test_minimal_detach_by_returned_header(gm, _env):
    sink = _sink(gm)
    lines = [
        "type,raw_name,translated_name,gender,relations,description",
        "character,서윤,Seo-yun,female,도현: Older Brother,Heir",
        "character,도현,Do-hyeon,male,,Captain",
    ]
    assert gm._detach_relations_cells(lines, ",", sink) == [
        "type,raw_name,translated_name,gender,description",
        "character,서윤,Seo-yun,female,Heir",
        "character,도현,Do-hyeon,male,Captain",
    ]
    assert sink["cells"] == [("서윤", "Seo-yun", "도현: Older Brother")]


def test_minimal_response_path_and_recording(gm, _env, tmp_path):
    _on(_env)
    _env.setenv("GLOSSARY_INCLUDE_DESCRIPTION", "1")
    sink = _sink(gm)
    response = (
        "character,서윤,Seo-yun,female,Heir,도현: Older Brother; 하린: Best Friend\n"
        "character,도현,Do-hyeon,male,Captain\n"
        "character,하린,Ha-rin,female,Role: healer\n"
    )
    lines = gm._process_ai_response(response, "", 1, False, 0.9, "korean", "all", relations_sink=sink)
    rows = [line.split(gm.GLOSSARY_SEP) for line in lines[1:]]
    assert [r[4] for r in rows] == ["Heir", "Captain", "Role: healer"]
    assert all(len(r) == 5 for r in rows)
    assert len(sink["cells"]) == 1

    other_dir = tmp_path / "other_book"
    with gm._minimal_relations_lock:
        gm._minimal_relations_cells[gm._minimal_relations_key(str(tmp_path))] = list(sink["cells"])
        gm._minimal_relations_cells[gm._minimal_relations_key(str(other_dir))] = [("X", "X", "Y: Z")]
    chapters = [
        {"body": "<p>서윤과 도현.</p>", "num": 1, "filename": "c1.xhtml"},
        {"body": "<p>서윤은 하린을 만났다.</p>", "num": 2, "filename": "c2.xhtml"},
    ]
    gm._record_minimal_relations(str(tmp_path), chapters)
    data = _tracker_without_time(tmp_path / "relations_tracker.json")
    # Minimal cells are stored on the reverse edge too: what 서윤 is to 도현 / 하린.
    entries = data["entries"]
    assert entries["도현"]["relations"]["서윤"]["occurrences"][0]["chapter_num"] == 1
    assert entries["하린"]["relations"]["서윤"]["occurrences"][0]["chapter_num"] == 2
    assert entries["하린"]["relations"]["서윤"]["occurrences"][0]["chapter_file"] == "c2.xhtml"
    assert entries["하린"]["relations"]["서윤"]["target_translated"] == "Seo-yun"
    assert set(entries) == {"도현", "하린"}
    # Another run's cells (another output folder) are neither recorded here nor dropped.
    key_other = gm._minimal_relations_key(str(other_dir))
    assert gm._minimal_relations_key(str(tmp_path)) not in gm._minimal_relations_cells
    assert gm._minimal_relations_cells.pop(key_other) == [("X", "X", "Y: Z")]


def test_minimal_filler_and_surname_labels_are_dropped(eg, _env, tmp_path):
    _on(_env)
    cells = [("로웰", "Lowell", "고드윈: Surname"), ("리리엣", "Liliet", "바알: Niece; 길버트: Acquaintance")]
    units = [(0, "로웰 고드윈. 리리엣과 바알, 길버트.")]
    eg.record_relations_for_chapters(cells, units, 0, str(tmp_path / "glossary.csv"))
    data = _tracker_without_time(tmp_path / "relations_tracker.json")
    assert set(data["entries"]) == {"바알"}
    assert data["entries"]["바알"]["relations"]["리리엣"]["occurrences"][0]["label"] == "Niece"


def test_tracker_writers_take_a_cross_process_lock(eg, _env, tmp_path, monkeypatch):
    """Recording and the editor/dedup sync hold one lock file per tracker (not in the glossary folder)."""
    import contextlib as _contextlib
    held = []

    @_contextlib.contextmanager
    def _fake_lock(path):
        held.append(path)
        yield

    monkeypatch.setattr(eg, "_locked_glossary_progress_file", _fake_lock)
    _on(_env)
    glossary = str(tmp_path / "Book_glossary.csv")
    eg.update_relations_tracker([{"raw_name": "서윤", "translated_name": "Seo-yun", "relations": "도현: Brother"}],
                                glossary, chapter_index=0, chapter_num=1, chapter_file="c1.xhtml")
    eg.sync_relations_tracker_with_glossary([{"raw_name": "서윤", "translated_name": "Seo-yun"}], glossary)
    with eg._relations_tracker_guard(rt.tracker_path_for_glossary(glossary)):
        with eg._relations_tracker_guard(rt.tracker_path_for_glossary(glossary)):  # nested: taken once
            pass
    assert len(held) == 3 and len(set(held)) == 1
    assert not held[0].startswith(str(tmp_path))
    assert os.listdir(tmp_path) and not [n for n in os.listdir(tmp_path) if n.endswith(".lock")]


def test_add_minimal_pass_environment_drops_the_relations_field(eg, _env):
    _env.setenv("GLOSSARY_CUSTOM_FIELDS", json.dumps(["description", "relations"]))
    with eg._minimal_pass_environment():  # tracker off: custom fields untouched
        assert json.loads(os.environ["GLOSSARY_CUSTOM_FIELDS"]) == ["description", "relations"]
        assert rt.ENV_ENABLE not in os.environ
    _on(_env)
    with eg._minimal_pass_environment():
        assert os.environ[rt.ENV_ENABLE] == "0"
        assert json.loads(os.environ["GLOSSARY_CUSTOM_FIELDS"]) == ["description"]
    assert os.environ[rt.ENV_ENABLE] == "1"
    assert json.loads(os.environ["GLOSSARY_CUSTOM_FIELDS"]) == ["description", "relations"]


def test_minimal_detach_respects_quoted_commas(gm, _env):
    """A quoted description with a comma is one cell, not the relations cell (review blocker)."""
    sink = _sink(gm)
    lines = [
        'character,서윤,Seo-yun,female,"Heir of the house, daughter of 무진: a merchant"',
        'character,도현,Do-hyeon,male,"Captain, of the gate",서윤: Younger Sister',
    ]
    assert gm._detach_relations_cells(lines, ",", sink) == [
        'character,서윤,Seo-yun,female,"Heir of the house, daughter of 무진: a merchant"',
        'character,도현,Do-hyeon,male,"Captain, of the gate"',
    ]
    assert sink["cells"] == [("도현", "Do-hyeon", "서윤: Younger Sister")]

    # A returned header that names the field: a quoted comma and no relations cell keeps
    # its whole description; an unquoted overflow that is not relations is left alone too.
    sink = _sink(gm)
    lines = [
        "type,raw_name,translated_name,gender,description,relations",
        'character,서윤,Seo-yun,female,"Heir, of the house"',
        "character,무진,Mu-jin,male,Head, of the house",
        "character,도현,Do-hyeon,male,Captain,서윤: Younger Sister",
    ]
    assert gm._detach_relations_cells(lines, ",", sink) == [
        "type,raw_name,translated_name,gender,description",
        'character,서윤,Seo-yun,female,"Heir, of the house"',
        "character,무진,Mu-jin,male,Head, of the house",
        "character,도현,Do-hyeon,male,Captain",
    ]
    assert sink["cells"] == [("도현", "Do-hyeon", "서윤: Younger Sister")]


def test_minimal_response_path_unchanged_without_sink(gm, _env):
    _env.setenv("GLOSSARY_INCLUDE_DESCRIPTION", "1")
    response = "character,서윤,Seo-yun,female,Heir\ncharacter,도현,Do-hyeon,male,Role: Captain\n"
    assert gm._process_ai_response(response, "", 1, False, 0.9, "korean", "all") == \
        gm._process_ai_response(response, "", 1, False, 0.9, "korean", "all", relations_sink=None)


def test_minimal_sink_reads_the_prompt_columns(gm, _env):
    sink = gm._minimal_relations_sink("Columns in this exact order: type,raw_name,translated_name\n", [], False)
    assert sink["columns"] == ["type", "raw_name", "translated_name"]
    sink = gm._minimal_relations_sink("{fields}", ["description", "relations"], True,
                                      "Glossary Columns: raw_name\n=== CHARACTERS ===\n* 서윤 = Seo-yun\n")
    assert sink["columns"][-1] == "relations" and sink["known"] == {"서윤"}


# ---------------------------------------------------------------------------
# Sidecar hygiene
# ---------------------------------------------------------------------------

def test_tracker_sidecar_is_a_known_glossary_artifact(tmp_path):
    import glossary_paths as gp
    import translation_pipeline as tp
    import unified_glossary as ug
    assert gp.glossary_book_base_from_output("My Book_relations_tracker.json") == "My Book"
    assert gp.legacy_glossary_base_from_filename("My Book_relations_tracker.json") == "My Book"
    assert gp.is_relations_tracker_file("relations_tracker.json")
    assert tp._is_relations_tracker_file("x_relations_tracker.json")
    assert not tp._is_relations_tracker_file("x_glossary.csv")
    assert not ug.is_book_glossary_filename("Book_relations_tracker.json")

    root = tmp_path / "Glossary"
    root.mkdir()
    (root / "Book_glossary.csv").write_text("x", encoding="utf-8")
    (root / "Book_relations_tracker.json").write_text("{}", encoding="utf-8")
    gp.migrate_all_legacy_glossary_files(str(root))
    assert (root / "Book" / "Book_relations_tracker.json").exists()

    result = gp.rename_auto_glossary_artifacts_for_book_rename([str(root)], "Book", "Novel")
    assert (root / "Novel" / "Novel_relations_tracker.json").exists()
    assert any(dst.endswith("Novel_relations_tracker.json") for _src, dst in result["moved"])


def test_delete_list_includes_the_relations_tracker(tmp_path, monkeypatch):
    import glossary_files as gf
    root = Path(os.environ["OUTPUT_DIRECTORY"])
    monkeypatch.setattr(gf, "_get_app_dir", lambda: str(root))
    book_dir = root / "Glossary" / "Book"
    book_dir.mkdir(parents=True)
    for name in ("Book_glossary.csv", "Book_gender_tracker.json", "Book_relations_tracker.json"):
        (book_dir / name).write_text("x", encoding="utf-8")
    files = gf.collect_glossary_files_for_inputs(
        [str(tmp_path / "Book.epub")], config={}, guess_glossary=lambda _p: None,
    )
    names = sorted(os.path.basename(p) for _b, p in files)
    assert "Book_relations_tracker.json" in names and "Book_gender_tracker.json" in names
