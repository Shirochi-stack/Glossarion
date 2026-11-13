import copy
import csv
import io
import json
import os
import re
import sys
import time

import pytest

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import relations_tracking as rt  # noqa: E402


ON = {rt.ENV_ENABLE: "1"}


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in (rt.ENV_ENABLE, rt.ENV_FIELD, rt.ENV_HISTORY_MODE, "GLOSSARY_TARGET_LANGUAGE"):
        monkeypatch.delenv(name, raising=False)


def _entry(raw, translated, cell, **extra):
    entry = {"type": "character", "raw_name": raw, "translated_name": translated, "gender": "female",
             "description": "x", "relations": cell}
    entry.update(extra)
    return entry


def _related(tracker):
    """The names that have a relation (an entry may also hold only its descriptions)."""
    return {key for key, item in tracker["entries"].items() if item["relations"]}


def _model_rows(rows):
    """The fixtures state the stored meaning: on X's row, ``Y: what Y is to X``. The model
    writes the other way round (on Y's row, ``X: what Y is to X``), so every item moves to
    the other character's row (a row is added for a character that has none)."""
    rows = copy.deepcopy(rows)
    by_key = {rt.tracker_key(r.get("raw_name")): r for r in rows}
    moved = {key: [] for key in by_key}
    extra = []
    for row in rows:
        for target, label in rt.parse_relations_cell(row.get("relations")):
            key = rt.tracker_key(target)
            if key not in by_key:
                by_key[key] = {"type": "character", "raw_name": target, "translated_name": ""}
                moved[key] = []
                extra.append(by_key[key])
            moved[key].append(f"{row['raw_name']}: {label}")
    for key, row in by_key.items():
        if "relations" in row or moved[key]:
            row["relations"] = "; ".join(moved[key])
    return rows + extra


def _record(tracker, chapter, rows, chapter_file=None):
    return rt.record_relations(
        tracker, _model_rows(rows), chapter_index=chapter - 1, chapter_num=chapter,
        chapter_file=chapter_file if chapter_file is not None else f"ch{chapter:03d}.xhtml",
    )


def _story():
    """서윤's relation to 하린: Best Friend (1-10), Rival (20), Enemy (40)."""
    return [
        (1, [_entry("서윤", "Seo-yun", "도현: Older Brother; 하린: Best Friend; 무진: Father"),
             _entry("도현", "Do-hyeon", "서윤: Younger Sister; 무진: Father"),
             _entry("하린", "Ha-rin", "서윤: Best Friend"),
             _entry("무진", "Mu-jin", "서윤: Daughter; 도현: Son")]),
        (10, [_entry("서윤", "Seo-yun", "하린: best friend")]),
        (20, [_entry("서윤", "Seo-yun", "하린: Rival")]),
        (21, [_entry("서윤", "Seo-yun", "하린: Rival")]),
        (40, [_entry("서윤", "Seo-yun", "하린: Enemy")]),
    ]


def _story_tracker(order=None):
    tracker = rt.empty_tracker()
    story = _story()
    for index in (order or range(len(story))):
        chapter, rows = story[index]
        _record(tracker, chapter, rows)
    return tracker


# --------------------------------------------------------------------------- settings / files

def test_settings_read_snapshot_then_env(monkeypatch):
    assert rt.tracking_enabled() is False
    assert rt.tracking_enabled(ON) is True
    assert rt.tracking_enabled({rt.CONFIG_ENABLE: True}) is True
    monkeypatch.setenv(rt.ENV_ENABLE, "1")
    assert rt.tracking_enabled() is True
    assert rt.tracking_enabled({rt.ENV_ENABLE: "0"}) is False
    assert rt.field_name() == "relations"
    assert rt.field_name({rt.ENV_FIELD: "  "}) == "relations"
    assert rt.field_name({rt.ENV_FIELD: "Bonds"}) == "Bonds"
    assert rt.history_mode() == "previous"
    assert rt.history_mode({rt.ENV_HISTORY_MODE: "FULL"}) == "full"
    assert rt.history_mode({rt.ENV_HISTORY_MODE: "junk"}) == "previous"


def test_tracker_paths_and_names(tmp_path):
    folder = str(tmp_path)
    assert rt.tracker_path_for_glossary(os.path.join(folder, "Book_glossary.csv")) == os.path.join(
        folder, "Book_relations_tracker.json")
    assert rt.tracker_path_for_glossary(os.path.join(folder, "glossary.csv")) == os.path.join(
        folder, "relations_tracker.json")
    assert rt.tracker_path_for_glossary(os.path.join(folder, "Other.json")) == os.path.join(
        folder, "Other_relations_tracker.json")
    assert rt.tracker_path_for_glossary("") == ""
    assert rt.is_tracker_file("Book_relations_tracker.json")
    assert rt.is_tracker_file(os.path.join(folder, "relations_tracker.json"))
    assert not rt.is_tracker_file("Book_gender_tracker.json")
    assert not rt.is_tracker_file("Book_glossary.csv")


def test_load_tracker_missing_and_broken(tmp_path):
    assert rt.load_tracker(str(tmp_path / "missing.json")) == rt.empty_tracker()
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    assert rt.load_tracker(str(broken)) == rt.empty_tracker()
    good = tmp_path / "good.json"
    good.write_text(json.dumps(_story_tracker(), ensure_ascii=False), encoding="utf-8")
    assert "서윤" in rt.load_tracker(str(good))["entries"]


# --------------------------------------------------------------------------- parsing

def test_parse_relations_cell_variants():
    assert rt.parse_relations_cell("도현: Older Brother; 하린: Best Friend") == [
        ("도현", "Older Brother"), ("하린", "Best Friend")]
    assert rt.parse_relations_cell("도현：형；하린：친구") == [("도현", "형"), ("하린", "친구")]
    assert rt.parse_relations_cell('"도현: Older Brother;"') == [("도현", "Older Brother")]
    assert rt.parse_relations_cell("'도현': 'Father'") == [("도현", "Father")]
    assert rt.parse_relations_cell("A: Father, B: Mother") == [("A", "Father"), ("B", "Mother")]
    assert rt.parse_relations_cell("junk; : nothing; 도현:; :Father; 하린 Best Friend") == []
    assert rt.parse_relations_cell("") == []
    assert rt.parse_relations_cell(None) == []
    long_label = "A: " + "word " * 12
    assert rt.parse_relations_cell(long_label) == []
    assert rt.parse_relations_cell("A: Father; a: Uncle") == [("A", "Father")]


def test_labels_equivalent():
    assert rt.labels_equivalent("Best Friend", "best-friend")
    assert rt.labels_equivalent("Older Brother", "Older Brothers")
    assert not rt.labels_equivalent("Mother", "Father")
    assert not rt.labels_equivalent("Older Sister", "Older Brother")
    assert not rt.labels_equivalent("Friend", "Enemy")
    # Adding "ex-" (or another qualifier) is a real change, not wording noise.
    for old, new in (("Husband", "Ex-husband"), ("Fiancé", "Ex-fiancé"), ("Girlfriend", "Ex-girlfriend"),
                     ("Boyfriend", "Ex-boyfriend"), ("Disciple", "Ex-disciple"), ("Husband", "Exhusband"),
                     ("Wife", "Former Wife"), ("Father", "Stepfather")):
        assert not rt.labels_equivalent(old, new), (old, new)
        assert not rt.labels_equivalent(new, old), (new, old)
    assert rt.labels_equivalent("Ex-husband", "Ex husband")


def test_ex_prefix_is_recorded_as_a_change():
    tracker = rt.empty_tracker()
    _record(tracker, 1, [_entry("A", "Aa", "B: Husband")])
    _record(tracker, 5, [_entry("A", "Aa", "B: Ex-husband")])
    state = rt.relations_at(tracker, "A", {"chapter_num": 6})[0]
    assert state["label"] == "Ex-husband" and state["former"] == ["Husband"]


# --------------------------------------------------------------------------- recording

def test_record_pops_field_and_skips_self_reference():
    tracker = rt.empty_tracker()
    rows = [_entry("서윤", "Seo-yun", "서윤: Sister; 도현: Younger Sister"), _entry("도현", "Do-hyeon", ""),
            {"type": "term", "raw_name": "검", "translated_name": "Sword", "Relations": "  "}]
    stats = rt.record_relations(tracker, rows, chapter_index=0, chapter_num=1, chapter_file="ch001.xhtml")
    assert all("relations" not in r and "Relations" not in r for r in rows)
    assert stats["pairs"] == 1 and stats["new"] == 1
    # 서윤's row says what 서윤 is to 도현; it is stored on 도현's entry.
    assert _related(tracker) == {"도현"}
    assert tracker["entries"]["서윤"]["relations"] == {}  # kept for its description only
    rels = tracker["entries"]["도현"]["relations"]
    assert list(rels) == ["서윤"]
    assert rels["서윤"]["target_translated"] == "Seo-yun"
    assert rels["서윤"]["occurrences"][0]["label"] == "Younger Sister"
    assert tracker["entries"]["도현"]["translated_name"] == "Do-hyeon"


def test_model_direction_is_stored_on_the_reverse_edge():
    """Father's row ``딸: Father`` + daughter's row ``아빠: Daughter`` -> entries[딸][아빠] = Father."""
    tracker = rt.empty_tracker()
    rt.record_relations(tracker, [_entry("아빠", "Dad", "딸: Father"), _entry("딸", "Girl", "아빠: Daughter")],
                        chapter_num=1, chapter_file="ch001.xhtml")
    entries = tracker["entries"]
    assert entries["딸"]["relations"]["아빠"]["occurrences"][0]["label"] == "Father"
    assert entries["아빠"]["relations"]["딸"]["occurrences"][0]["label"] == "Daughter"
    assert rt.render_relations(rt.relations_at(tracker, "딸", 1)) == "Dad: Father"
    # The other character needs no row in this response; its entry is still created.
    tracker = rt.empty_tracker()
    stats = rt.record_relations(tracker, [_entry("바알", "Baal", "리리엣: Uncle")], chapter_num=1)
    assert stats["new"] == 1
    liliet = tracker["entries"]["리리엣"]
    assert liliet["raw_name"] == "리리엣" and liliet["translated_name"] == ""
    assert liliet["relations"]["바알"]["target_translated"] == "Baal"
    assert tracker["entries"]["바알"]["relations"] == {}
    # A later sync fills the translated name in.
    rt.sync_tracker_with_glossary(tracker, [{"type": "character", "raw_name": "리리엣", "translated_name": "Liliet"},
                                            {"type": "character", "raw_name": "바알", "translated_name": "Baal"}])
    assert tracker["entries"]["리리엣"]["translated_name"] == "Liliet"


def test_filler_labels_never_enter_the_tracker():
    tracker = rt.empty_tracker()
    cell = ("B: Acquaintance; C: stranger; D: UNKNOWN; E: None; F: n/a; G: Surname; H: Same person; "
            "I: self; J: Friend")
    stats = rt.record_relations(tracker, [_entry("A", "Aa", cell)], chapter_num=1)
    assert stats["pairs"] == 1 and _related(tracker) == {"j"}
    assert all(rt.is_filler_label(label) for label in rt.FILLER_LABELS)
    assert not rt.is_filler_label("Friend") and not rt.is_filler_label("Acquaintance's Son")
    # Only filler: nothing is recorded at all.
    tracker = rt.empty_tracker()
    assert rt.record_relations(tracker, [_entry("A", "Aa", "B: Acquaintance")], chapter_num=1)["pairs"] == 0
    assert _related(tracker) == set()


def test_record_keeps_field_when_asked():
    rows = [_entry("서윤", "Seo-yun", "도현: Brother")]
    rt.record_relations(rt.empty_tracker(), rows, chapter_num=1, pop_field=False)
    assert rows[0]["relations"] == "도현: Brother"


def test_out_of_order_arrival_matches_in_order():
    in_order = _story_tracker()
    shuffled = _story_tracker(order=[4, 2, 0, 3, 1])
    repeated = _story_tracker(order=[4, 2, 0, 3, 1, 0, 4])
    assert json.dumps(in_order, sort_keys=True, ensure_ascii=False) == json.dumps(
        shuffled, sort_keys=True, ensure_ascii=False)
    assert json.dumps(in_order, sort_keys=True, ensure_ascii=False) == json.dumps(
        repeated, sort_keys=True, ensure_ascii=False)


def test_first_wording_wins_in_chapter_order():
    tracker = _story_tracker(order=[1, 0])  # chapter 10 ("best friend") arrives first
    occurrences = tracker["entries"]["서윤"]["relations"]["하린"]["occurrences"]
    assert [o["label"] for o in occurrences] == ["Best Friend", "Best Friend"]
    assert occurrences[1]["wording"] == "best friend"


def test_change_history_and_stats():
    tracker = rt.empty_tracker()
    story = _story()
    for chapter, rows in story[:4]:
        _record(tracker, chapter, rows)
    stats = _record(tracker, 40, story[4][1])
    assert stats["changed"] == 1
    assert stats["changes"] == [{"raw_name": "서윤", "target_raw": "하린", "from": "Rival", "to": "Enemy",
                                 "chapter_num": 40, "chapter_file": "ch040.xhtml"}]
    changes = tracker["entries"]["서윤"]["relations"]["하린"]["changes"]
    assert [(c["from"], c["to"], c["chapter_num"]) for c in changes] == [
        ("Best Friend", "Rival", 20), ("Rival", "Enemy", 40)]


def test_blip_filter():
    tracker = rt.empty_tracker()
    for chapter, label in ((1, "Friend"), (2, "Friend"), (3, "Enemy"), (4, "Friend"), (5, "Friend")):
        _record(tracker, chapter, [_entry("A", "Aa", f"B: {label}")])
    rel = tracker["entries"]["a"]["relations"]["b"]
    assert rel["changes"] == []
    blips = [o for o in rel["occurrences"] if o.get("blip")]
    assert [(o["label"], o["chapter_num"]) for o in blips] == [("Enemy", 3)]
    state = rt.relations_at(tracker, "A", {"chapter_num": 3})
    assert state[0]["label"] == "Friend" and state[0]["former"] == []
    # A two-chapter run is a real change.
    _record(tracker, 6, [_entry("A", "Aa", "B: Enemy")])
    _record(tracker, 7, [_entry("A", "Aa", "B: Enemy")])
    assert [(c["from"], c["to"]) for c in tracker["entries"]["a"]["relations"]["b"]["changes"]] == [
        ("Friend", "Enemy")]


def test_entries_ordered_by_first_occurrence():
    tracker = rt.empty_tracker()
    _record(tracker, 5, [_entry("Z", "Zed", "Y: Friend")])
    _record(tracker, 1, [_entry("M", "Em", "N: Friend")])
    assert list(tracker["entries"]) == ["m", "z"]


# --------------------------------------------------------------------------- reading

def test_history_modes():
    tracker = _story_tracker()
    ref = {"chapter_num": 45, "chapter_file": "ch045.xhtml"}

    def ha_rin(mode):
        return next(r for r in rt.relations_at(tracker, "서윤", ref, mode) if r["target_raw"] == "하린")

    assert ha_rin("previous")["label"] == "Enemy"
    assert ha_rin("previous")["former"] == ["Rival"]
    assert ha_rin("full")["former"] == ["Rival", "Best Friend"]
    assert ha_rin("current")["former"] == []
    assert ha_rin("previous")["changed_at"] == 40
    assert ha_rin("previous")["since"] == 40


def test_cutoff_never_reveals_later_chapter():
    tracker = _story_tracker()
    early = {r["target_raw"]: r for r in rt.relations_at(tracker, "서윤", {"chapter_num": 5})}
    assert early["하린"]["label"] == "Best Friend" and early["하린"]["former"] == []
    assert early["하린"]["changed_at"] is None
    at_39 = {r["target_raw"]: r for r in rt.relations_at(tracker, "서윤", {"chapter_num": 39})}
    assert at_39["하린"]["label"] == "Rival"
    # Before the first observation there is nothing (no fallback to a later chapter).
    assert rt.relations_at(tracker, "서윤", {"chapter_num": 0}) == []
    assert rt.relations_at(tracker, "서윤", {"chapter_num": 0.5, "chapter_file": "prologue.xhtml"}) == []


def test_same_chapter_file_decides_over_a_different_numbering():
    """Extraction filed relations under its own numbers (101, 120, 140); translation calls the
    same files 1, 20, 40. The request's own file decides, as in the gender tracker."""
    tracker = rt.empty_tracker()
    for chapter, label in ((101, "Best Friend"), (120, "Rival"), (140, "Enemy")):
        rt.record_relations(tracker, _model_rows([_entry("서윤", "Seo-yun", f"하린: {label}")]),
                            chapter_index=chapter - 101, chapter_num=chapter,
                            chapter_file=f"ch{chapter:04d}.xhtml")

    def label_at(ref):
        found = rt.relations_at(tracker, "서윤", ref)
        return found[0]["label"] if found else None

    assert label_at({"chapter_num": 1, "chapter_file": "ch0101.xhtml"}) == "Best Friend"
    assert label_at({"chapter_num": 20, "chapter_file": "ch0120.xhtml"}) == "Rival"
    assert label_at({"chapter_num": 40, "chapter_file": "ch0140.xhtml"}) == "Enemy"
    # Reverse offset: translation's number is ahead of the tracker's; the file still bounds it.
    assert label_at({"chapter_num": 500, "chapter_file": "ch0120.xhtml"}) == "Rival"
    # Merged group 20-39 whose parent file is known: the end shifts by the file's offset.
    assert label_at({"chapter_num": 20, "chapter_num_end": 39, "chapter_file": "ch0120.xhtml"}) == "Rival"
    assert label_at({"chapter_num": 20, "chapter_num_end": 40, "chapter_file": "ch0120.xhtml"}) == "Enemy"
    # An unknown file falls back to the number.
    assert label_at({"chapter_num": 1, "chapter_file": "other.xhtml"}) is None

    text = "Glossary Columns: raw_name, translated_name\n=== CHARACTERS ===\n* 서윤 = Seo-yun\n* 하린 = Ha-rin\n"
    _out, stats = rt.inject_relations_column(text, None, {"chapter_num": 40, "chapter_file": "ch0140.xhtml"},
                                             settings=ON, tracker=tracker)
    assert stats["cutoff"] == 40 and stats["relations"] == 1
    assert stats["changed_here"] == ["Seo-yun → Ha-rin: Enemy [formerly Rival]"]


def test_chapter_num_end_extends_cutoff():
    tracker = _story_tracker()
    merged = {"chapter_num": 38, "chapter_num_end": 40, "chapter_file": "ch038.xhtml"}
    state = {r["target_raw"]: r for r in rt.relations_at(tracker, "서윤", merged)}
    assert state["하린"]["label"] == "Enemy"
    cutoff = rt.chapter_cutoff(merged)
    assert cutoff["chapter_num"] == 40 and cutoff["start"] == 38


def test_cutoff_by_chapter_file_only():
    tracker = _story_tracker()
    state = {r["target_raw"]: r for r in rt.relations_at(tracker, "서윤", {"chapter_file": "ch020.xhtml"})}
    assert state["하린"]["label"] == "Rival"
    # A file not in the tracker is placed by the number in its name.
    state = {r["target_raw"]: r for r in rt.relations_at(tracker, "서윤", {"chapter_file": "ch030.xhtml"})}
    assert state["하린"]["label"] == "Rival"
    assert rt.relations_at(tracker, "서윤", {"chapter_file": "prologue.xhtml"}) == []


def test_no_cutoff():
    tracker = _story_tracker()
    for ref in (None, {}, "", {"use_storage_gender": True}, {"chapter_num": None}):
        assert rt.chapter_cutoff(ref) is None
        assert rt.relations_at(tracker, "서윤", ref) == []
    assert rt.chapter_cutoff(7)["chapter_num"] == 7


def test_render_relations():
    rels = [{"target_raw": "도현", "target_translated": "Do-hyeon", "label": "Older Brother", "former": []},
            {"target_raw": "하린", "target_translated": "Ha-rin", "label": "Enemy", "former": ["Best Friend"]},
            {"target_raw": "X", "target_translated": "", "label": "Friend", "former": []}]
    assert rt.render_relations(rels[:2]) == "Do-hyeon: Older Brother; Ha-rin: Enemy [formerly Best Friend]"
    names = {"도현": "DH"}
    assert rt.render_relations(rels, names.get) == "DH: Older Brother"


def test_character_history_and_bounds():
    tracker = _story_tracker()
    history = {h["target_raw"]: h for h in rt.character_history(tracker, "서윤")}
    timeline = history["하린"]["timeline"]
    assert [(t["label"], t["from_chapter"], t["to_chapter"]) for t in timeline] == [
        ("Best Friend", 1, 10), ("Rival", 20, 21), ("Enemy", 40, 40)]
    assert history["하린"]["current"] == "Enemy"
    assert rt.chapter_bounds(tracker) == (1, 40)
    assert rt.chapter_bounds(rt.empty_tracker()) == (None, None)


# --------------------------------------------------------------------------- injection

TOKEN_TEXT = (
    "Glossary Columns: raw_name, translated_name, gender, description, title\n"
    "\n"
    "=== CHARACTERS ===\n"
    "* 서윤 = Seo-yun [Female]: Heir of the Baek trading house (title: Young Lady)\n"
    "* 도현 = Do-hyeon [Male]: Royal guard captain\n"
    "* 하린 = Ha-rin [Female]\n"
    "* 설화 = Seol-hwa [Female]: Healer\n"
    "\n"
    "=== TERMS ===\n"
    "* 청월검 = Azure Moon Sword: Heirloom blade\n"
    "\n"
)


def _legacy_csv_text():
    # Built with csv.writer exactly like save_glossary_csv's legacy branch
    # (trailing empty cells beyond the header are popped).
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["type", "raw_name", "translated_name", "gender", "description"])
    writer.writerow(["character", "서윤", "Seo-yun", "Female", "Heir, of the house"])
    writer.writerow(["character", "도현", "Do-hyeon", "Male", "Captain"])
    writer.writerow(["character", "하린", "Ha-rin", "Female"])
    writer.writerow(["term", "청월검", "Azure Moon Sword", "", "Blade"])
    return buffer.getvalue()


def test_injection_token_efficient():
    tracker = _story_tracker()
    text, stats = rt.inject_relations_column(TOKEN_TEXT, "Book_glossary.csv", {"chapter_num": 45}, ON, tracker)
    lines = text.split("\n")
    assert lines[0] == "Glossary Columns: raw_name, translated_name, gender, description, title, relations"
    assert lines[3] == ("* 서윤 = Seo-yun [Female]: Heir of the Baek trading house (title: Young Lady) "
                        "(relations: Do-hyeon: Older Brother; Ha-rin: Enemy [formerly Rival])")
    assert lines[4] == "* 도현 = Do-hyeon [Male]: Royal guard captain (relations: Seo-yun: Younger Sister)"
    assert lines[5] == "* 하린 = Ha-rin [Female] (relations: Seo-yun: Best Friend)"
    assert lines[6] == "* 설화 = Seol-hwa [Female]: Healer"
    assert lines[9] == "* 청월검 = Azure Moon Sword: Heirloom blade"
    # 무진 is not in the text, so Father/Daughter relations are not shown.
    assert "Mu-jin" not in text and "Father" not in text
    assert stats == {"enabled": True, "entries": 3, "relations": 4, "cutoff": 45, "changed_here": []}
    # The parser the app uses still reads the identities.
    import glossary_usage
    parsed = glossary_usage._parse_token_entry_line(lines[3])
    assert parsed["raw_name"] == "서윤" and parsed["translated_name"] == "Seo-yun"


def test_injection_changed_here_and_log_line():
    tracker = _story_tracker()
    ref = {"chapter_num": 38, "chapter_num_end": 41, "chapter_file": "ch038.xhtml"}
    _text, stats = rt.inject_relations_column(TOKEN_TEXT, "Book_glossary.csv", ref, ON, tracker)
    assert stats["changed_here"] == ["Seo-yun → Ha-rin: Enemy [formerly Rival]"]
    assert rt.relations_log_line(stats) == (
        "🔗 Relations: 4 shown for 3 characters (up to Ch. 41) | changed here: "
        "Seo-yun → Ha-rin: Enemy [formerly Rival]")
    _text, early = rt.inject_relations_column(TOKEN_TEXT, "Book_glossary.csv", {"chapter_num": 0}, ON, tracker)
    assert early["relations"] == 0 and _text == TOKEN_TEXT
    assert rt.relations_log_line(early) == "🔗 Relations: none recorded up to Ch. 0"
    assert rt.relations_log_line({"enabled": False}) is None
    assert rt.relations_log_line(None) is None


def test_injection_legacy_csv():
    tracker = _story_tracker()
    source = _legacy_csv_text()
    text, stats = rt.inject_relations_column(source, "Book_glossary.csv", {"chapter_num": 5}, ON, tracker)
    rows = list(csv.reader(io.StringIO(text)))
    assert rows[0] == ["type", "raw_name", "translated_name", "gender", "description", "relations"]
    assert rows[1] == ["character", "서윤", "Seo-yun", "Female", "Heir, of the house",
                       "Do-hyeon: Older Brother; Ha-rin: Best Friend"]
    assert rows[3] == ["character", "하린", "Ha-rin", "Female", "", "Seo-yun: Best Friend"]
    assert rows[4] == ["term", "청월검", "Azure Moon Sword", "", "Blade"]
    assert stats["entries"] == 3


def test_injection_legacy_unit_separator():
    tracker = _story_tracker()
    sep = "\x1f"
    source = sep.join(["type", "raw_name", "translated_name", "gender", "description"]) + "\n" + sep.join(
        ["character", "서윤", "Seo-yun", "Female", "Heir, of the house"]) + "\n" + sep.join(
        ["character", "하린", "Ha-rin", "Female", "Healer"]) + "\n"
    text, _stats = rt.inject_relations_column(source, "g.csv", {"chapter_num": 5}, ON, tracker)
    lines = text.split("\n")
    assert lines[0].split(sep)[-1] == "relations"
    assert lines[1].split(sep) == ["character", "서윤", "Seo-yun", "Female", "Heir, of the house",
                                   "Ha-rin: Best Friend"]


def test_injection_json_formats():
    tracker = _story_tracker()
    as_list = json.dumps([
        {"type": "character", "raw_name": "서윤", "translated_name": "Seo-yun", "gender": "Female"},
        {"type": "character", "raw_name": "무진", "translated_name": "Mu-jin", "gender": "Male"},
        {"type": "term", "raw_name": "청월검", "translated_name": "Azure Moon Sword"},
    ], ensure_ascii=False, indent=2)
    text, stats = rt.inject_relations_column(as_list, "g.json", {"chapter_num": 45}, ON, tracker)
    data = json.loads(text)
    assert data[0]["relations"] == "Mu-jin: Father"
    assert data[1]["relations"] == "Seo-yun: Daughter"
    assert "relations" not in data[2]
    assert stats["entries"] == 2

    as_dict = json.dumps({"metadata": {"v": 1}, "entries": {
        "서윤": {"translated_name": "Seo-yun"}, "도현": {"translated_name": "Do-hyeon"}}}, ensure_ascii=False)
    text, _ = rt.inject_relations_column(as_dict, "g.json", {"chapter_num": 45},
                                         {rt.ENV_ENABLE: "1", rt.ENV_FIELD: "bonds"}, tracker)
    data = json.loads(text)
    assert data["entries"]["서윤"]["bonds"] == "Do-hyeon: Older Brother"
    assert data["metadata"] == {"v": 1}

    # {"metadata": ..., "entries": [ ... ]} and lists under other keys.
    entries_list = json.dumps({"metadata": {"v": 1}, "entries": [
        {"raw_name": "서윤", "translated_name": "Seo-yun"}, {"raw_name": "도현", "translated_name": "Do-hyeon"}]},
        ensure_ascii=False)
    text, stats = rt.inject_relations_column(entries_list, "g.json", {"chapter_num": 45}, ON, tracker)
    data = json.loads(text)
    assert data["entries"][0]["relations"] == "Do-hyeon: Older Brother" and stats["relations"] == 2
    grouped = json.dumps({"metadata": {"v": 1}, "characters": [
        {"raw_name": "서윤", "translated_name": "Seo-yun"}, {"raw_name": "도현", "translated_name": "Do-hyeon"}]},
        ensure_ascii=False)
    text, stats = rt.inject_relations_column(grouped, "g.json", {"chapter_num": 45}, ON, tracker)
    assert json.loads(text)["characters"][1]["relations"] == "Seo-yun: Younger Sister"


def test_injection_off_or_without_data_returns_text_unchanged(tmp_path):
    tracker = _story_tracker()
    text, stats = rt.inject_relations_column(TOKEN_TEXT, "g.csv", {"chapter_num": 45}, {}, tracker)
    assert text is TOKEN_TEXT and stats == {"enabled": False}
    text, stats = rt.inject_relations_column(TOKEN_TEXT, "g.csv", None, ON, tracker)
    assert text is TOKEN_TEXT and stats == {"enabled": False}
    text, stats = rt.inject_relations_column(TOKEN_TEXT, "g.csv", {"use_storage_gender": True}, ON, tracker)
    assert text is TOKEN_TEXT and stats == {"enabled": False}
    missing = str(tmp_path / "Book_glossary.csv")
    text, stats = rt.inject_relations_column(TOKEN_TEXT, missing, {"chapter_num": 45}, ON)
    assert text is TOKEN_TEXT and stats == {"enabled": False}
    text, stats = rt.inject_relations_column(TOKEN_TEXT, "g.csv", {"chapter_num": 45}, ON, rt.empty_tracker())
    assert text is TOKEN_TEXT and stats == {"enabled": False}


def test_injection_reads_tracker_file_beside_glossary(tmp_path):
    glossary = tmp_path / "Book_glossary.csv"
    glossary.write_text(TOKEN_TEXT, encoding="utf-8")
    with open(rt.tracker_path_for_glossary(str(glossary)), "w", encoding="utf-8") as handle:
        json.dump(_story_tracker(), handle, ensure_ascii=False)
    text, stats = rt.inject_relations_column(TOKEN_TEXT, str(glossary), {"chapter_num": 5}, ON)
    assert stats["enabled"] and "(relations: Do-hyeon: Older Brother; Ha-rin: Best Friend)" in text


def test_injection_keeps_crlf_line_endings():
    tracker = _story_tracker()
    source = TOKEN_TEXT.replace("\n", "\r\n")
    text, _stats = rt.inject_relations_column(source, "g.csv", {"chapter_num": 5}, ON, tracker)
    assert text.count("\r\n") == source.count("\r\n")
    assert "\r\n* 하린 = Ha-rin [Female] (relations: Seo-yun: Best Friend)\r\n" in text


# --------------------------------------------------------------------------- sync

def test_sync_rekeys_honorific_variant_and_identical_translation():
    tracker = rt.empty_tracker()
    _record(tracker, 1, [_entry("서윤님", "Lady Seo-yun", "도현: Older Brother; 하린양: Best Friend"),
                         _entry("도현", "Do-hyeon", "서윤님: Younger Sister"),
                         _entry("백하린", "Ha-rin", "서윤님: Best Friend")])
    _record(tracker, 2, [_entry("서윤", "Seo-yun", "도현: Older Brother; 유령: Ghost")])
    glossary = [
        {"type": "character", "raw_name": "서윤", "translated_name": "Seo-yun"},
        {"type": "character", "raw_name": "도현", "translated_name": "Do-hyeon"},
        {"type": "character", "raw_name": "하린", "translated_name": "Ha-rin"},
    ]

    def strip(raw):
        for suffix in ("님", "양"):
            if raw.endswith(suffix):
                return raw[: -len(suffix)]
        return raw

    stats = rt.sync_tracker_with_glossary(tracker, glossary, strip_honorifics=strip)
    entries = tracker["entries"]
    assert "서윤님" not in entries and "백하린" not in entries
    seo = entries["서윤"]
    assert seo["translated_name"] == "Seo-yun"
    assert "서윤님" in seo["aliases"]
    brother = seo["relations"]["도현"]["occurrences"]
    assert [(o["label"], o["chapter_num"]) for o in brother] == [("Older Brother", 1), ("Older Brother", 2)]
    # Target re-keyed by honorific strip, entry re-keyed by identical translation.
    assert "하린" in seo["relations"] and "하린양" not in seo["relations"]
    assert entries["하린"]["relations"]["서윤"]["target_raw"] == "서윤"
    assert entries["도현"]["relations"]["서윤"]["target_translated"] == "Seo-yun"
    # Unresolvable target stays untouched.
    assert seo["relations"]["유령"]["target_raw"] == "유령"
    assert stats["unlisted"] == 1
    assert stats["rekeyed"] + stats["merged"] >= 4
    # Reading uses the surviving names.
    state = {r["target_raw"]: r for r in rt.relations_at(tracker, "서윤", {"chapter_num": 2})}
    assert state["하린"]["label"] == "Best Friend"


def test_sync_without_strip_uses_translation_only():
    tracker = rt.empty_tracker()
    _record(tracker, 1, [_entry("서윤님", "Seo-yun", "도현: Brother")])
    stats = rt.sync_tracker_with_glossary(tracker, [{"raw_name": "서윤", "translated_name": "Seo-yun"}])
    assert "서윤" in tracker["entries"] and stats["rekeyed"] == 1


def test_sync_follows_refinement_to_full_names():
    """Glossary refinement turns given names into full names (바알 -> 바알 아르젠트)."""
    tracker = rt.empty_tracker()
    _record(tracker, 1, [_entry("바알", "Baal", "리리엣: Niece; 에드먼드: Younger Brother"),
                         _entry("에드먼드", "Edmund", "바알: Older Brother; 아르젠트: Family"),
                         _entry("리리엣", "Liliet", "바알: Uncle")])
    glossary = [
        {"type": "character", "raw_name": "바알 아르젠트", "translated_name": "Baal Argent"},
        {"type": "character", "raw_name": "에드먼드 아르젠트", "translated_name": "Edmund Argent"},
        {"type": "character", "raw_name": "리리엣", "translated_name": "Liliet"},
        {"type": "term", "raw_name": "리리엣 영지", "translated_name": "Liliet Estate"},
    ]
    stats = rt.sync_tracker_with_glossary(tracker, glossary)
    assert set(tracker["entries"]) == {"바알 아르젠트", "에드먼드 아르젠트", "리리엣"}
    baal = tracker["entries"]["바알 아르젠트"]
    assert baal["aliases"] == ["바알"] and set(baal["relations"]) == {"리리엣", "에드먼드 아르젠트"}
    assert set(tracker["entries"]["리리엣"]["relations"]) == {"바알 아르젠트"}
    # "아르젠트" is a word of two names: ambiguous, so it stays unlisted.
    assert "아르젠트" in tracker["entries"]["에드먼드 아르젠트"]["relations"]
    assert stats["unlisted"] == 1
    # Translated one-word names resolve the same way.
    tracker = rt.empty_tracker()
    _record(tracker, 1, [_entry("발", "Baal", "리리엣: Niece")])
    rt.sync_tracker_with_glossary(tracker, glossary)
    assert "바알 아르젠트" in tracker["entries"]


# --------------------------------------------------------------------------- placeholder

def test_apply_relations_placeholder():
    prompt = "Rules:\n- Keep names.\n- {relations_instruction}\n- End.\n"
    assert rt.apply_relations_placeholder(prompt, enabled=False) == "Rules:\n- Keep names.\n- End.\n"
    assert rt.apply_relations_placeholder("A\n{relations_instruction}\nB", enabled=False) == "A\nB"
    expanded = rt.apply_relations_placeholder(prompt, enabled=True, field="bonds", language="Spanish")
    assert "{relations_instruction}" not in expanded
    assert "The `bonds` column" in expanded and "in Spanish" in expanded
    assert "LAST column" not in expanded
    declared = rt.apply_relations_placeholder(prompt, enabled=True, declare_column=True)
    assert "one extra LAST column of every row" in declared and "The `relations` column" in declared
    assert rt.apply_relations_placeholder("No placeholder", enabled=True) == "No placeholder"
    assert rt.apply_relations_placeholder(None, enabled=True) is None


def test_placeholder_follows_env(monkeypatch):
    prompt = "- {relations_instruction}\n"
    assert rt.apply_relations_placeholder(prompt) == ""
    monkeypatch.setenv(rt.ENV_ENABLE, "1")
    monkeypatch.setenv("GLOSSARY_TARGET_LANGUAGE", "French")
    assert "in French" in rt.apply_relations_placeholder(prompt)


def test_prompts_missing_placeholder():
    prompts = {"balanced_full": "x\n- {relations_instruction}\n", "minimal": "no"}
    assert rt.prompts_missing_placeholder(prompts) == ["minimal"]
    assert rt.prompts_missing_placeholder({"balanced_full": None}) == ["balanced_full"]
    assert rt.prompts_missing_placeholder({}) == []


def test_fresh_desktop_config_gets_both_toggles_and_the_relations_column():
    config = {rt.FRESH_DEFAULTS_KEY: list(rt.FRESH_DEFAULT_KEYS)}
    assert rt.apply_fresh_defaults(config) is True
    assert config == {"custom_glossary_fields": ["description", "relations"],
                      rt.CONFIG_ENABLE: True, rt.CONFIG_MINIMAL: True}
    # a reset config (the kept keys plus the mark): the same, the kept keys untouched
    reset = {"model": "m", "api_key": "k", rt.FRESH_DEFAULTS_KEY: list(rt.FRESH_DEFAULT_KEYS)}
    assert rt.apply_fresh_defaults(reset) is True
    assert reset["model"] == "m" and reset[rt.CONFIG_ENABLE] is True and rt.FRESH_DEFAULTS_KEY not in reset
    # a toggle the config already holds is kept
    held = {rt.FRESH_DEFAULTS_KEY: list(rt.FRESH_DEFAULT_KEYS), rt.CONFIG_ENABLE: False}
    rt.apply_fresh_defaults(held)
    assert held[rt.CONFIG_ENABLE] is False and held[rt.CONFIG_MINIMAL] is True and "custom_glossary_fields" not in held


def test_existing_desktop_config_is_not_touched():
    import copy

    for config in ({}, {"model": "m"}, {"model": "m", rt.CONFIG_ENABLE: False},
                   {"custom_glossary_fields": ["description"], "auto_update_check": True}):
        before = copy.deepcopy(config)
        assert rt.apply_fresh_defaults(config) is False
        assert config == before and json.dumps(config) == json.dumps(before)


def test_desktop_startup_seeds_only_a_fresh_config(tmp_path, monkeypatch):
    """owner_state._init_config_state (TranslatorGUI.__init__'s config block; HeadlessOwner runs it too)."""
    import app_paths
    from headless_owner import HeadlessOwner

    for key in ("GLOSSARION_APP_DIR", "GLOSSARION_LIBRARY_DIR", "OUTPUT_DIRECTORY"):
        (tmp_path / key).mkdir()
        monkeypatch.setenv(key, str(tmp_path / key))
    config_file = tmp_path / "config.json"
    monkeypatch.setattr(app_paths, "CONFIG_FILE", str(config_file))
    fresh = HeadlessOwner({})  # no config.json
    assert fresh.config[rt.CONFIG_ENABLE] is True and fresh.config[rt.CONFIG_MINIMAL] is True
    assert fresh.config["custom_glossary_fields"] == ["description", "relations"]
    assert fresh.custom_glossary_fields == ["description", "relations"]
    assert fresh.glossary_enable_relations_tracker_var is True
    assert fresh.glossary_minimal_include_relations_var is True
    assert rt.FRESH_DEFAULTS_KEY not in fresh.config
    # an existing config without the keys: read as off, nothing added
    old = HeadlessOwner({"model": "gpt-x"})
    assert not old.config.get(rt.CONFIG_ENABLE) and not old.config.get(rt.CONFIG_MINIMAL)
    assert old.glossary_enable_relations_tracker_var is False and old.glossary_minimal_include_relations_var is False
    assert "relations" not in old.config.get("custom_glossary_fields", [])
    # an empty config while config.json exists (unreadable and protected): not fresh
    config_file.write_text("{}", encoding="utf-8")
    assert not HeadlessOwner({}).config.get(rt.CONFIG_ENABLE)
    # the reset mark (Reset Settings to Defaults) works on a non-empty config too
    reset = HeadlessOwner({"model": "gpt-x", rt.FRESH_DEFAULTS_KEY: list(rt.FRESH_DEFAULT_KEYS)})
    assert reset.config[rt.CONFIG_ENABLE] is True and reset.config[rt.CONFIG_MINIMAL] is True


def test_desktop_reset_writes_the_fresh_mark():
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "src"
    settings = (src / "other_settings.py").read_text(encoding="utf-8-sig")
    start = settings.index("    def _reset_config_to_defaults():")
    reset = settings[start:settings.index("reset_btn = QPushButton", start)]
    mark = reset.index("keys_to_preserve[relations_tracking.FRESH_DEFAULTS_KEY]")
    assert mark < reset.index("_atomic_json_write(config_path, keys_to_preserve)")


def test_minimal_prompt_is_checked_only_with_its_toggle(monkeypatch):
    prompts = {"balanced_full": "b", "minimal": "m"}
    assert rt.prompts_to_check(prompts, {rt.CONFIG_MINIMAL: False}) == {"balanced_full": "b"}
    assert rt.prompts_to_check(prompts, {rt.CONFIG_MINIMAL: True}) == prompts
    assert rt.prompts_to_check(prompts, {rt.ENV_MINIMAL: "1"}) == prompts
    assert rt.prompts_to_check(None) == {}
    monkeypatch.delenv(rt.ENV_MINIMAL, raising=False)
    monkeypatch.setenv(rt.ENV_ENABLE, "1")
    assert rt.prompts_to_check(prompts) == {"balanced_full": "b"}
    assert not rt.minimal_tracking_enabled()
    monkeypatch.setenv(rt.ENV_MINIMAL, "1")
    assert rt.minimal_tracking_enabled() and rt.prompts_to_check(prompts) == prompts
    monkeypatch.setenv(rt.ENV_ENABLE, "0")
    assert not rt.minimal_tracking_enabled()


def test_instruction_text_is_one_block():
    text = rt.instruction_text()
    assert text.startswith("The `relations` column: for character entries only")
    assert text.count("\n") == 1
    # The model writes what THIS row's character is to the other one, shown both ways.
    assert "role is what THIS row's character is to that other character" in text
    assert "a father's row has `<his daughter's raw_name>: Father`" in text
    assert "the daughter's row has `<her father's raw_name>: Daughter`" in text
    assert "never filler such as Acquaintance, Stranger or Unknown, and never a note such as Surname" in text
    # Without a known layout: the four fixed columns and the relations.
    assert text.endswith(
        "\nExample rows that mirror each other: `character,이히리ᐐ 나애,Dihirit Ade,female,뢤사난: Older Sister` "
        "and `character,뢤사난,Kim Sang-hyu,male,이히리ᐐ 나애: Younger Brother`")
    # Recording the example rows gives the tracker's own meaning (she is his Older Sister).
    tracker = rt.empty_tracker()
    rt.record_relations(tracker, [_entry("이히리ᐐ 나애", "Dihirit Ade", "뢤사난: Older Sister"),
                                  _entry("뢤사난", "Kim Sang-hyu", "이히리ᐐ 나애: Younger Brother")],
                        chapter_num=1)
    assert rt.render_relations(rt.relations_at(tracker, "뢤사난", 1)) == "Dihirit Ade: Older Sister"


def _example_rows(text):
    line = text.split("\n")[-1]
    return [next(csv.reader([row])) for row in re.findall(r"`([^`]+)`", line)]


def test_instruction_examples_follow_the_prompt_columns():
    """A relation written in the description's place taught the model to merge the two."""
    columns = ["type", "raw_name", "translated_name", "gender", "description", "relations"]
    text = rt.instruction_text(columns=columns)
    assert text.endswith(
        '\nExample rows that mirror each other: `character,이히리ᐐ 나애,Dihirit Ade,female,"Guild leader",'
        '뢤사난: Older Sister` and `character,뢤사난,Kim Sang-hyu,male,"Swordsman",이히리ᐐ 나애: Younger Brother`')
    rows = _example_rows(text)
    assert [len(row) for row in rows] == [6, 6]
    assert rows[0][5] == "뢤사난: Older Sister" and rows[1][5] == "이히리ᐐ 나애: Younger Brother"
    for row in rows:
        assert row[0] == "character" and row[3] in ("female", "male")
        assert row[4] and ":" not in row[4]  # a description holds the column's place, not a relation

    # SHORT: a longer instruction made a 7B model that followed the short one ignore it
    # (owner, 2026-11-13). The whole text with description and relations columns is within
    # 40 characters of what it was with the five-column examples, and it is still two lines.
    base = rt.instruction_text()
    assert text.count("\n") == 1 and len(text) - len(base) <= 40 and len(text) < 1300
    assert "is its own column" not in text and "column 6 of 6" not in text

    # The field keeps its place and its name wherever the user put it; other custom fields stay empty.
    text = rt.instruction_text(field="bonds", columns=["type", "raw_name", "translated_name", "gender",
                                                       "Bonds", "notes", "description"])
    rows = _example_rows(text)
    assert [len(row) for row in rows] == [7, 7]
    assert rows[0][4] == "뢤사난: Older Sister" and rows[0][5] == "" and rows[0][6] == "Guild leader"

    # A layout without the field (Minimal's fixed list): the relations are the extra last column.
    text = rt.instruction_text(declare_column=True,
                               columns=["type", "raw_name", "translated_name", "gender", "description"])
    assert text.startswith("Add a `relations` column as one extra LAST column of every row. ")
    assert [len(row) for row in _example_rows(text)] == [6, 6]


# --------------------------------------------------------------------------- reciprocal flips

def _flip_tracker(order):
    """A→B is Uncle (B is A's uncle) in chapters 1-8, then the model slips to "Niece" (the
    reverse edge's label) for two chapters, a run the blip filter would not catch."""
    observations = [
        (1, [("A", "B", "Uncle"), ("B", "A", "Niece")]),
        (3, [("A", "B", "Uncle"), ("B", "A", "Niece")]),
        (6, [("A", "B", "Uncle"), ("B", "A", "Niece")]),
        (8, [("A", "B", "Uncle"), ("B", "A", "Niece")]),
        (9, [("A", "B", "Niece")]),
        (10, [("A", "B", "Niece")]),
    ]
    tracker = rt.empty_tracker()
    for index in order:
        chapter, items = observations[index]
        rows = {}
        for row, target, label in items:
            rows.setdefault(row, []).append(f"{target}: {label}")
        _record(tracker, chapter, [_entry(r, r.lower() * 2, "; ".join(c)) for r, c in rows.items()])
    return tracker


def test_reciprocal_flip_is_ignored_like_a_blip():
    tracker = _flip_tracker(range(6))
    rel = tracker["entries"]["a"]["relations"]["b"]
    flips = [(o["label"], o["chapter_num"]) for o in rel["occurrences"] if o.get("flip")]
    assert flips == [("Niece", 9), ("Niece", 10)]
    assert rel["changes"] == []
    assert not any(o.get("flip") for o in tracker["entries"]["b"]["relations"]["a"]["occurrences"])
    state = rt.relations_at(tracker, "A", 10)[0]
    assert state["label"] == "Uncle" and state["former"] == []
    history = rt.character_history(tracker, "A")[0]
    assert [t["label"] for t in history["timeline"]] == ["Uncle"]
    assert [(b["label"], b.get("flip")) for b in history["blips"]] == [("Niece", True)] * 2
    edge = rt.build_graph(tracker)["edges"][0]
    assert {edge["label_ab"], edge["label_ba"]} == {"Uncle", "Niece"} and edge["changed"] is False


def test_reciprocal_flip_is_order_independent():
    expected = json.dumps(_flip_tracker(range(6)), sort_keys=True, ensure_ascii=False)
    for order in ([5, 4, 3, 2, 1, 0], [2, 0, 5, 1, 4, 3], [3, 1, 5, 0, 2, 4, 2]):
        assert json.dumps(_flip_tracker(order), sort_keys=True, ensure_ascii=False) == expected


def test_same_label_both_ways_is_never_a_flip():
    tracker = rt.empty_tracker()
    for chapter, label in ((1, "Friend"), (2, "Friend"), (3, "Enemy"), (4, "Enemy"), (5, "Enemy")):
        _record(tracker, chapter, [_entry("A", "Aa", f"B: {label}"), _entry("B", "Bb", "A: Friend")])
    rel = tracker["entries"]["a"]["relations"]["b"]
    assert not any(o.get("flip") for o in rel["occurrences"])
    assert [(c["from"], c["to"]) for c in rel["changes"]] == [("Friend", "Enemy")]
    # A label the reverse edge does not use is a real change, not a flip.
    tracker = rt.empty_tracker()
    for chapter, label in ((1, "Master"), (2, "Master"), (5, "Enemy"), (6, "Enemy")):
        _record(tracker, chapter, [_entry("A", "Aa", f"B: {label}"), _entry("B", "Bb", "A: Disciple")])
    rel = tracker["entries"]["a"]["relations"]["b"]
    assert not any(o.get("flip") for o in rel["occurrences"])
    assert [(c["from"], c["to"]) for c in rel["changes"]] == [("Master", "Enemy")]


# --------------------------------------------------------------------------- custom fields

def test_apply_relations_field():
    assert rt.apply_relations_field(["description"], True) == ["description", "relations"]
    assert rt.apply_relations_field(["relations", "description"], True) == ["relations", "description"]
    assert rt.apply_relations_field(["Relations", "description", "relations"], True) == ["relations", "description"]
    assert rt.apply_relations_field(["description", "relations", "title"], False) == ["description", "title"]
    assert rt.apply_relations_field(["description", "relations", "title"], True, name="bonds",
                                    previous_name="relations") == ["description", "bonds", "title"]
    assert rt.apply_relations_field(["bonds", "relations"], False, name="bonds",
                                    previous_name="relations") == []
    original = ["description"]
    rt.apply_relations_field(original, True)
    assert original == ["description"]


def test_is_relations_field_and_persisted_fields():
    assert rt.is_relations_field("Relations")
    assert not rt.is_relations_field("description")
    assert rt.is_relations_field("bonds", {rt.CONFIG_FIELD: "bonds"})
    assert rt.is_relations_field("bonds", {rt.ENV_FIELD: "Bonds"})
    fields = ["description", "relations", "title"]
    assert rt.persisted_custom_fields(fields) == fields
    assert rt.persisted_custom_fields(fields, ON) == ["description", "title"]
    assert rt.persisted_custom_fields(fields, {rt.CONFIG_ENABLE: True, rt.CONFIG_FIELD: "title"}) == [
        "description", "relations"]


def test_relations_field_never_takes_a_glossary_column():
    for name in ("gender", "Raw_Name", "translated_name", "type", "description"):
        assert rt.field_name_problem(name, ["notes"]) == f"'{name}' is a built-in glossary column."
        # A config that names one anyway falls back to "relations", so no real column is claimed.
        assert rt.field_name({rt.ENV_FIELD: name}) == "relations"
        assert not rt.is_relations_field(name, {rt.ENV_FIELD: name})
    assert rt.persisted_custom_fields(["description"], {rt.ENV_ENABLE: "1", rt.ENV_FIELD: "description"}) == [
        "description"]
    assert rt.field_name_problem("", []) == "The relations field needs a name."
    assert rt.field_name_problem("Notes", ["notes"]) == "'Notes' is already a custom field."
    assert rt.field_name_problem("bonds", ["description"]) == ""


def test_enable_keeps_the_users_own_column():
    assert rt.enable_field_name(["description"]) == "relations"
    assert rt.enable_field_name(["description", "Relations"]) == "relations_tracker"
    assert rt.enable_field_name(["relations", "relations_tracker"]) == "relations_tracker_2"
    assert rt.enable_field_name(["bonds"], "bonds") == "bonds_tracker"


# --------------------------------------------------------------------------- graph

def test_build_graph_pairs_and_cutoff():
    tracker = _story_tracker()
    graph = rt.build_graph(tracker)
    edges = {(e["a"], e["b"]): e for e in graph["edges"]}
    # 서윤/도현 described from both sides -> one edge with both labels.
    pair = edges[tuple(sorted(("서윤", "도현")))]
    if pair["a"] == "서윤":
        assert (pair["label_ab"], pair["label_ba"]) == ("Older Brother", "Younger Sister")
    else:
        assert (pair["label_ab"], pair["label_ba"]) == ("Younger Sister", "Older Brother")
    ha = edges[tuple(sorted(("서윤", "하린")))]
    assert ha["changed"] is True
    assert graph["chapters"] == [1, 40]
    assert len(graph["edges"]) == len(edges) == 4
    early = rt.build_graph(tracker, up_to_chapter=5)
    ha_early = {(e["a"], e["b"]): e for e in early["edges"]}[tuple(sorted(("서윤", "하린")))]
    assert ha_early["changed"] is False
    assert "Best Friend" in (ha_early["label_ab"], ha_early["label_ba"])
    assert rt.build_graph(tracker, up_to_chapter=0)["nodes"] == []
    degrees = {n["id"]: n["degree"] for n in graph["nodes"]}
    assert degrees["서윤"] == 3


def test_build_graph_unlisted():
    tracker = _story_tracker()
    glossary = [{"raw_name": "서윤", "translated_name": "Seo-yun", "gender": "Female"},
                {"raw_name": "도현", "translated_name": "Do-hyeon"}]
    graph = rt.build_graph(tracker, glossary)
    assert {n["id"] for n in graph["nodes"]} == {"서윤", "도현"}
    node = next(n for n in graph["nodes"] if n["id"] == "서윤")
    assert node["gender"] == "Female" and node["label"] == "Seo-yun" and node["listed"]
    full = rt.build_graph(tracker, glossary, show_unlisted=True)
    listed = {n["id"]: n["listed"] for n in full["nodes"]}
    assert listed["하린"] is False and listed["서윤"] is True


def _big_graph(count):
    nodes = [{"id": f"n{i}"} for i in range(count)]
    edges = [{"a": f"n{i}", "b": f"n{(i * 7 + 3) % count}"} for i in range(count) if (i * 7 + 3) % count != i]
    edges += [{"a": f"n{i}", "b": f"n{i + 1}"} for i in range(0, count - 1, 2)]
    return {"nodes": nodes, "edges": edges}


@pytest.mark.parametrize("count", [0, 1, 2, 300])
def test_layout_bounds_and_determinism(count):
    graph = _big_graph(count) if count else {"nodes": [], "edges": []}
    started = time.perf_counter()
    first = rt.layout_graph(graph, 800, 600, seed=3)
    elapsed = time.perf_counter() - started
    assert first == rt.layout_graph(graph, 800, 600, seed=3)
    assert len(first) == count
    for x, y in first.values():
        assert 0.0 <= x <= 800 and 0.0 <= y <= 600
    if count == 1:
        assert first["n0"] == (400.0, 300.0)
    if count == 2:
        assert first["n0"] != first["n1"]
    if count == 300:
        assert elapsed < 3.0  # generous for slow CI; typically well under a second


def test_layout_disconnected_and_focus():
    graph = {"nodes": [{"id": c} for c in "abcdef"],
             "edges": [{"a": "a", "b": "b"}, {"a": "c", "b": "d"}]}
    positions = rt.layout_graph(graph, seed=1)
    assert len(set(positions.values())) == 6
    radial = rt.layout_graph(graph, 1000, 700, focus="a")
    assert radial["a"] == (500.0, 350.0)
    for x, y in radial.values():
        assert 0.0 <= x <= 1000 and 0.0 <= y <= 700
    import random
    state = random.getstate()
    rt.layout_graph(_big_graph(50), seed=9)
    assert random.getstate() == state


def test_character_history_stops_at_a_chapter():
    """The diagram's history panel shows what was known at the slider's chapter, no later."""
    tracker = _story_tracker()

    def harin(chapter):
        rows = {r["target_raw"]: r for r in rt.character_history(tracker, "서윤", chapter)}
        return rows.get("하린")

    whole = harin(None)
    assert whole["current"] == "Enemy" and [run["label"] for run in whole["timeline"]] == [
        "Best Friend", "Rival", "Enemy"]
    assert whole == {r["target_raw"]: r for r in rt.character_history(tracker, "서윤")}["하린"]
    early = harin(15)
    assert early["current"] == "Best Friend" and len(early["timeline"]) == 1
    assert early["timeline"][0]["to_chapter"] == 10 and early["changes"] == []
    middle = harin(25)
    assert middle["current"] == "Rival" and [c["to"] for c in middle["changes"]] == ["Rival"]
    assert harin(40)["current"] == "Enemy"
    # A relation first seen later is left out, and before the first chapter there is nothing.
    late = rt.empty_tracker()
    _record(late, 3, [_entry("a", "A", "b: Friend")])
    _record(late, 9, [_entry("a", "A", "c: Rival")])
    assert [r["target_raw"] for r in rt.character_history(late, "a", 5)] == ["b"]
    assert [r["target_raw"] for r in rt.character_history(late, "a")] == ["b", "c"]
    assert rt.character_history(late, "a", 2) == []


def test_edge_label_text_both_directions():
    edge = {"label_ab": "Enemy", "former_ab": ["Best Friend"], "label_ba": "Father", "former_ba": []}
    assert rt.edge_label_text(edge) == "Enemy / Father"
    assert rt.edge_label_text(edge, True) == "Enemy [Best Friend] / Father"
    assert rt.edge_label_text({"label_ab": "", "label_ba": "Rival"}) == "? / Rival"
    # The same word from both sides is said once.
    assert rt.edge_label_text({"label_ab": "Teammate", "label_ba": "teammate"}) == "Teammate"
    same = {"label_ab": "Enemy", "former_ab": ["Friend"], "label_ba": "Enemy", "former_ba": []}
    assert rt.edge_label_text(same) == "Enemy"
    assert rt.edge_label_text(same, True) == "Enemy [Friend] / Enemy"
    assert rt.edge_label_text({"label_ab": "", "label_ba": ""}) == "? / ?"


def _team_graph():
    """The shape of a real book: a leader with five subordinates, four of the team all
    related to each other, and one pair with no line to anyone else."""
    hub, team, leaves = "min", ["park", "chunhu", "junho", "minju"], ["a", "b", "c", "d", "e"]
    nodes = [{"id": n} for n in [hub] + team + leaves + ["x", "y"]]
    edges = [{"a": hub, "b": n} for n in team + leaves]
    edges += [{"a": a, "b": b} for i, a in enumerate(team) for b in team[i + 1:]]
    edges.append({"a": "x", "b": "y"})
    return {"nodes": nodes, "edges": edges}


def test_small_layout_keeps_the_tidiest_try():
    graph = _team_graph()
    edge_list = [(e["a"], e["b"]) for e in graph["edges"]]
    width, height = 1200.0, 800.0
    positions = rt.layout_graph(graph, width, height)
    assert positions == rt.layout_graph(graph, width, height)
    # Kept: the best of several tries, never worse than the first one alone.
    import random
    first = rt._force_positions([n["id"] for n in graph["nodes"]], edge_list, width, height,
                                random.Random(0), rt._REFINED_LAYOUT_ITERATIONS)
    assert rt.layout_untidiness(positions, edge_list) <= rt.layout_untidiness(first, edge_list)
    # Five names all related to each other cannot be drawn without one crossing; more than
    # a couple, or a line through a name, is the mess this layout exists to avoid.
    assert rt.layout_untidiness(positions, edge_list) <= 3.0

    def gap(a, b):
        return ((positions[a][0] - positions[b][0]) ** 2 + (positions[a][1] - positions[b][1]) ** 2) ** 0.5
    lengths = [gap(a, b) for a, b in edge_list]
    names = list(positions)
    # No two names on top of each other, no line too short to carry its label.
    assert min(gap(a, b) for i, a in enumerate(names) for b in names[i + 1:]) > 60
    assert min(lengths) > 90
    # The pair with no line to the others sits beside them, not in a far corner.
    others = [n for n in names if n not in ("x", "y")]
    assert min(gap("x", n) for n in others) < 2.5 * (sum(lengths) / len(lengths))


def test_layout_untidiness_counts_crossings_and_lines_through_names():
    square = {"a": (0.0, 0.0), "b": (100.0, 100.0), "c": (0.0, 100.0), "d": (100.0, 0.0)}
    assert rt.layout_untidiness(square, [("a", "b"), ("c", "d")]) == 1.0  # the diagonals cross
    assert rt.layout_untidiness(square, [("a", "c"), ("d", "b")]) == 0.0
    line = {"a": (0.0, 0.0), "b": (200.0, 0.0), "c": (100.0, 5.0)}
    assert rt.layout_untidiness(line, [("a", "b")]) == 2.0  # c sits on the a-b line


def test_small_layout_is_quick_at_its_largest():
    graph = _big_graph(rt._REFINED_LAYOUT_MAX_NODES)
    started = time.perf_counter()
    positions = rt.layout_graph(graph, 1600, 1100)
    assert len(positions) == rt._REFINED_LAYOUT_MAX_NODES
    assert time.perf_counter() - started < 5.0  # generous for slow CI; about half a second here


def test_reset_question_is_one_question_for_one_or_both_prompts():
    one = rt.reset_question(["Minimal"])
    assert one == ("The Minimal prompt has no {relations_instruction} line, so the AI will not return relations."
                   "\n\nReset the Minimal prompt to default?")
    both = rt.reset_question(["Balanced/Full", "Minimal"])
    assert both == ("The Balanced/Full and Minimal prompts have no {relations_instruction} line, so the AI will "
                    "not return relations.\n\nReset both prompts to default?")


# --------------------------------------------------------------------------- descriptions per chapter

def _describe(tracker, chapter, raw, text, cell="", **extra):
    row = {"type": "character", "raw_name": raw, "translated_name": raw.title(), "description": text,
           "relations": cell}
    row.update(extra)
    return rt.record_relations(tracker, [row], chapter_index=chapter - 1, chapter_num=chapter,
                               chapter_file=f"ch{chapter:03d}.xhtml")


def test_description_written_at_each_chapter_is_kept_and_read_by_chapter():
    tracker = rt.empty_tracker()
    # Out of order, as parallel requests arrive.
    assert _describe(tracker, 12, "min", "Leader of Team 1, now  feared\nby the Red Tiger clan")["descriptions"] == 1
    _describe(tracker, 3, "min", "A playful hunter", cell="jin: Team Leader")
    _describe(tracker, 8, "min", "Team leader of the Night Wind clan")
    history = rt.description_history(tracker, "Min")
    assert [(d["chapter_num"], d["text"]) for d in history] == [
        (3, "A playful hunter"), (8, "Team leader of the Night Wind clan"),
        (12, "Leader of Team 1, now feared by the Red Tiger clan")]
    # The one written at that chapter, else the last before it; never a later one.
    assert rt.description_at(tracker, "min", 8)["text"] == "Team leader of the Night Wind clan"
    assert rt.description_at(tracker, "min", 11)["chapter_num"] == 8
    assert rt.description_at(tracker, "min", 2) is None
    assert rt.description_at(tracker, "min")["chapter_num"] == 12
    assert rt.description_at(tracker, "nobody", 5) is None
    # The same chapter recorded again: unchanged text adds nothing, new text replaces it.
    assert _describe(tracker, 8, "min", "Team leader of the Night Wind clan")["descriptions"] == 0
    assert _describe(tracker, 8, "min", "Team leader, sniper's partner")["descriptions"] == 1
    assert [d["text"] for d in rt.description_history(tracker, "min") if d["chapter_num"] == 8] == [
        "Team leader, sniper's partner"]
    # The description stays on the row: it is a glossary column, only relations are taken off.
    row = {"type": "character", "raw_name": "jin", "description": "A sniper", "relations": "min: Subordinate"}
    rt.record_relations(tracker, [row], chapter_num=9)
    assert row == {"type": "character", "raw_name": "jin", "description": "A sniper"}


def test_descriptions_are_kept_for_characters_and_related_names_only():
    tracker = rt.empty_tracker()
    rows = [{"type": "term", "raw_name": "검", "description": "A sword"},
            {"type": "titles", "raw_name": "대장", "description": "A rank", "relations": "min: Superior"},
            {"type": "character", "raw_name": "min", "description": ""},
            {"type": "character", "raw_name": "jin", "description": "A sniper"},
            {"type": "character", "raw_name": "long", "description": "word " * 400}]
    stats = rt.record_relations(tracker, rows, chapter_num=4, chapter_file="ch004.xhtml")
    assert stats["descriptions"] == 3
    assert "검" not in tracker["entries"]                      # not a character, no relation
    assert rt.description_at(tracker, "대장")["text"] == "A rank"  # has a relation
    assert rt.description_at(tracker, "jin")["text"] == "A sniper"
    assert rt.description_at(tracker, "min") is None
    long_text = rt.description_at(tracker, "long")["text"]
    assert len(long_text) == rt.MAX_DESCRIPTION_CHARS and long_text.endswith("…")
    # Names without relations do not become diagram nodes.
    assert {n["id"] for n in rt.build_graph(tracker)["nodes"]} == {"min", "대장"}


def test_descriptions_follow_a_name_merged_by_the_glossary_sync():
    tracker = rt.empty_tracker()
    _describe(tracker, 1, "서윤님", "The heir, as first seen", cell="도현: Younger Sister")
    _describe(tracker, 2, "서윤", "The heir of the Baek house")
    _describe(tracker, 2, "서윤님", "Overwritten by the surviving name's own")
    stats = rt.sync_tracker_with_glossary(
        tracker, [{"type": "character", "raw_name": "서윤", "translated_name": "Seo-yun"},
                  {"type": "character", "raw_name": "도현", "translated_name": "Do-hyeon"}],
        strip_honorifics=lambda raw: raw.replace("님", ""))
    assert "서윤님" not in tracker["entries"] and stats["unlisted"] == 0
    assert [(d["chapter_num"], d["text"]) for d in rt.description_history(tracker, "서윤")] == [
        (1, "The heir, as first seen"), (2, "The heir of the Baek house")]


def test_descriptions_shown_by_a_diagram():
    tracker = rt.empty_tracker()
    glossary = [{"raw_name": "min", "description": "The glossary's  present\ndescription"},
                {"raw_name": "jin", "description": ""}]
    # A tracker from before descriptions were recorded: the glossary's own, undated.
    assert rt.descriptions_shown(tracker, "min", 5, glossary) == (
        {"text": "The glossary's present description", "chapter_num": None, "chapter_file": ""}, [], "glossary")
    assert rt.descriptions_shown(tracker, "jin", 5, glossary) == (None, [], "")
    assert rt.descriptions_shown(tracker, "min", 5) == (None, [], "")
    for chapter, text in ((3, "first"), (8, "second"), (12, "third")):
        _describe(tracker, chapter, "min", text)
    written, earlier, source = rt.descriptions_shown(tracker, "min", 9, glossary)
    assert (written["text"], written["chapter_num"], source) == ("second", 8, "chapter")
    assert [d["text"] for d in earlier] == ["first"]
    written, earlier, _source = rt.descriptions_shown(tracker, "min", None, glossary)
    assert written["text"] == "third" and [d["text"] for d in earlier] == ["second", "first"]
    # Recorded, but nothing yet at that chapter: not the glossary's later text.
    assert rt.descriptions_shown(tracker, "min", 2, glossary) == (None, [], "chapter")
