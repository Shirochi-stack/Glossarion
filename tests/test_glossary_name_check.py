"""Glossary Name Check (experimental): glossary_name_check core, translation hook, QA scanner."""
import json
import os
import sys

import pytest

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import glossary_name_check as g  # noqa: E402
import ner_names  # noqa: E402


@pytest.fixture(autouse=True)
def _no_ner_download(monkeypatch):
    """The checks below use the heuristic unless a test brings its own tagger (no 104 MB download)."""
    monkeypatch.setenv("GLOSSARY_NAME_CHECK_USE_NER", "0")

GLOSSARY_ROWS = [
    {"type": "character", "raw_name": "발터 아르겐트", "translated_name": "Valter Argent"},
    {"type": "character", "raw_name": "강시현", "translated_name": "Kang Si-hyeon"},
    {"type": "character", "raw_name": "유소정", "translated_name": "Yoo Sojeong"},
    {"type": "character", "raw_name": "흑철기사", "translated_name": "Black Iron Knight"},
    {"type": "character", "raw_name": "염라투룡", "translated_name": "Yeomra Turyong (Minho Namgung)"},
    {"type": "character", "raw_name": "밀라노-V", "translated_name": "Milano-V"},
    {"type": "character", "raw_name": "아델라", "translated_name": "Adela"},
    {"type": "character", "raw_name": "칼", "translated_name": "Carl"},
    {"type": "location", "raw_name": "아벨란 제국", "translated_name": "Abelan Empire"},
]


def entries(mode="characters", min_chars=3):
    return g.build_entries(GLOSSARY_ROWS, mode=mode, min_chars=min_chars)


def findings(source, output, mode="characters", min_chars=3):
    return g.check_chapter(source, output, entries(mode, min_chars)).findings


def test_variant_spelling_is_found_and_fixed_in_text_only():
    output = ('<p title="Walter Argent">Walter Argent smiled.</p>'
              "<p>It was Walter Argent's sword.</p>")
    result = g.check_chapter("발터 아르겐트가 웃었다.", output, entries())
    [finding] = result.findings
    assert (finding.kind, finding.variants, finding.expected) == ("variant", ("Walter Argent",), "Valter Argent")
    assert finding.fixable and finding.occurrences == 2
    fixed, count = g.apply_fixes(output, result.fixable)
    assert count == 2
    assert fixed == ('<p title="Walter Argent">Valter Argent smiled.</p>'
                     "<p>It was Valter Argent's sword.</p>")


@pytest.mark.parametrize("output", [
    "<p>The Black Iron Knights charged.</p>",       # plural
    "<p>Kang Si-hyeon-ssi nodded.</p>",              # honorific suffix
    "<p>Valter Argent nodded.</p>",                  # correct
    "<p>Yeomra Turyong laughed.</p>",                # glossary note in parentheses
])
def test_same_name_is_not_drift(output):
    source = "흑철기사 강시현 발터 아르겐트 염라투룡"
    assert [f for f in findings(source, output) if f.kind != g.KIND_MISSING] == []


def test_sentence_words_and_honorifics_are_stripped_from_the_variant():
    [finding] = findings("유소정", "<p>He looked at her. But Yoo So-jeong-ssi only smiled.</p>")
    assert finding.kind == "variant" and finding.variants == ("Yoo So-jeong",)


def test_numbered_units_cut_off_names_and_short_words_are_not_variants():
    source = "밀라노-V 아델라 발터 아르겐트"
    output = "<p>Milano-VI fired. Then Ade arrived. Sion and Walt left with Valter.</p>"
    assert [f.kind for f in findings(source, output)] == ["missing", "missing", "missing"]


def test_glued_and_cut_names_are_not_variants():
    rows = [{"type": "character", "raw_name": "사쿠라", "translated_name": "Sakura"},
            {"type": "character", "raw_name": "미하일", "translated_name": "Mihail"},
            {"type": "character", "raw_name": "싱리안 종", "translated_name": "Singlian Sect"}]
    ents = g.build_entries(rows)
    output = "<p>He met Asakura, then Mihailtal and the Lian Sect.</p>"
    assert [f.kind for f in g.check_chapter("사쿠라 미하일 싱리안 종", output, ents).findings] == ["missing"] * 3


@pytest.mark.parametrize("expected,candidate,plurals,tenses,same", [
    # plurals, word by word (owner 2026-10-28: "Heroes of the Sword", "Beastmen")
    ("Hero of the Sword", "Heroes of the Sword", True, False, True),
    ("Beastman", "Beastmen", True, False, True),
    ("Beastmen", "Beastman", True, False, True),
    ("Knight of the Round", "Knights of the Round", True, False, True),
    ("Child of Light", "Children of Light", True, False, True),
    ("Wolf", "Wolves", True, False, True), ("Dwarf", "Dwarves", True, False, True),
    ("Witch", "Witches", True, False, True), ("Fairy", "Fairies", True, False, True),
    ("Succubus", "Succubi", True, False, True), ("Seraph", "Seraphim", True, False, True),
    ("Swordsman", "Swordsmen", True, False, True),
    ("Hero of the Sword", "Heroes of the Sword", False, False, False),  # the toggle off
    ("Minho Moralis", "Minho Morales", True, False, False),  # no blanket Latin rule: a real misspelling
    ("Kang-hyun", "Kang Hyun", True, False, False),  # a spacing respelling stays one
    ("S Rank", "S-ranks", True, True, False),  # a hyphen respelling stays one, plural or not (corpus)
    ("Half-Elf", "Half-Elves", True, False, True), ("Cyclops", "Cyclopes", True, False, True),
    ("Fallen Angel", "Falling Angel", True, False, False),  # a verb form is not a plural
    # verb forms (off by default)
    ("Fallen Angel", "Falling Angel", False, True, True),
    ("Fallen Angel", "Fell Angel", False, True, True),
    ("Awakened One", "Awakening One", False, True, True),
    ("Stopped Time", "Stopping Time", False, True, True),
    ("The Hero Who Will Return", "The Hero Who Returned", False, True, True),
    ("The Girl Who Will Become the Saintess", "The Girl Who Became the Saintess", False, True, True),
    ("Will Return", "Returned", False, True, False),  # "Will" opening a name is a name
    ("Black Rose", "Black Rise", False, True, False),  # rose / saw / left... are left out of the verb list
    ("Fallen Dark Angel", "Falling Dark Angels", False, True, False),  # one word only
    ("Hunter", "Hunted", False, True, False),
])
def test_plural_and_tense_forms_are_the_same_name(expected, candidate, plurals, tenses, same):
    assert g._inflected_same(expected, candidate, plurals, tenses) is same


def test_tense_never_turns_a_word_into_another_glossary_name():
    names = {g._norm("Falling Angel")}
    assert g._inflected_same("Fallen Angel", "Falling Angel", False, True)
    assert not g._inflected_same("Fallen Angel", "Falling Angel", False, True, names={g._norm("Falling")})
    assert names  # the full names come from build_entries
    ents = g.build_entries([{"type": "term", "raw_name": "타락천사", "translated_name": "Fallen Angel"},
                            {"type": "term", "raw_name": "낙하", "translated_name": "Falling"}], mode="all")
    assert g._norm("Falling") in ents.names and g._norm("Fallen Angel") in ents.names



@pytest.mark.parametrize("expected,output,plurals,tenses,present", [
    ("Beastman", "The Beastmen ran.", True, False, True),
    ("Beastmen", "A lone Beastman.", True, False, True),
    ("Hero of the Sword", "Generations of Heroes of the Sword.", True, False, True),
    ("Fox Beastman", "The Fox Beastmen hid.", True, False, True),
    ("Wolf", "Wolves howled.", True, False, True),
    ("Succubus", "Two Succubi.", True, False, True),
    ("Hero of the Sword", "Generations of Heroes of the Sword.", False, False, False),  # the toggle off
    ("Kang-hyun", "Kang Hyun", True, False, False),
    ("Fallen Angel", "A Falling Angel.", True, False, False),
    ("Fallen Angel", "A Falling Angel.", False, True, True),
    ("The Hero Who Will Return", "The Hero Who Returned.", False, True, True),
    ("The Hero Who Returned", "The Hero Who Will Return.", False, True, True),
    ("Will Return", "Returned.", False, True, False),
    ("Black Rose", "Black Rise.", False, True, False),
    ("Minho Moralis", "Minho Morales.", True, True, False),
    # the 15,960-chapter corpus (2026-10-28): rank letters are no plurals, no "-ss" plural, the form as it is
    ("SSS Rank", "Her SS Rank trait.", True, False, False),
    ("S Rank", "Two SS-ranks.", True, False, False),
    ("Nebines Mansion", "The Nebiness Mansion.", True, False, False),
    ("Crown Prince", "Crown Princess Consort.", True, True, False),
    ("S Rank", "Hari's ranking.", False, True, False),
    ("Witch", "Two Witches.", True, False, True),
    ("Monarch", "The Monarchs.", True, False, True),
    ("Fox", "Foxes.", True, False, True),
])
def test_the_name_in_another_form_is_present_without_ner(expected, output, plurals, tenses, present):
    assert g._inflected_present(expected, output, plurals, tenses) is present


def test_a_plural_ner_did_not_tag_is_the_name_but_never_for_a_character(tmp_path):
    """2026-10-28: "Beastman missing" where the chapter said "Beastmen" (NER tags no common noun);
    "Luca" never makes the character "Lucas" present (the plural rule would read it as the singular)."""
    rows = [{"type": "terms", "raw_name": "수인", "translated_name": "Beastman"},
            {"type": "character", "raw_name": "루카스", "translated_name": "Lucas"}]
    output = "<p>The Beastmen ran. Luca smiled.</p>"
    tagger = _fake_tagger(tmp_path, g.html_to_text(output), {"Luca": 3})

    def kinds(**settings):
        ents = g.build_entries(rows, mode="all", **settings)
        return {f.expected: f.kind for f in g.check_chapter("수인들 루카스", output, ents, ner=tagger).findings}

    assert kinds(plural_exclusion=False) == {"Beastman": "missing", "Lucas": "missing"}
    assert kinds() == {"Lucas": "missing"}



def test_the_types_the_plural_and_tense_rules_skip_are_a_setting(tmp_path):
    """2026-10-28 (owner): "Gender types except titles, but make it fully configurable ... for user added
    custom entry types it is off by default"."""
    rows = [{"type": "nicknames", "raw_name": "늑대들", "translated_name": "Wolves of the North"},
            {"type": "skills", "raw_name": "송곳니", "translated_name": "Fang of the Wolf"},
            {"type": "titles", "raw_name": "검의 용사", "translated_name": "Hero of the Sword"},
            {"type": "character", "raw_name": "수인", "translated_name": "Beastman"}]
    output = "<p>The Wolf of the North. Fangs of the Wolf. Heroes of the Sword. The Beastmen ran.</p>"
    source = "늑대들 송곳니 검의 용사 수인"

    def reported(**settings):
        ents = g.build_entries(rows, mode="all", **settings)
        return sorted(f.expected for f in g.check_chapter(source, output, ents).findings)

    # default: character and nicknames keep their findings; titles and a custom type ("skills") do not
    assert reported() == ["Beastman", "Wolves of the North"]
    assert reported(inflection_skip_types=()) == []  # nothing skipped
    assert reported(inflection_skip_types=("skill", "title")) == ["Fang of the Wolf", "Hero of the Sword"]  # term / terms
    assert g.settings_from_env({})["inflection_skip_types"] == ("character", "nicknames")
    env = {"GLOSSARY_NAME_CHECK_INFLECTION_SKIP_TYPES": '["skills"]'}
    assert g.settings_from_env(env)["inflection_skip_types"] == ("skills",)
    assert g.settings_from_env({"GLOSSARY_NAME_CHECK_INFLECTION_SKIP_TYPES": "not json"})["inflection_skip_types"] \
        == ("character", "nicknames")
    config = {"glossary_name_check_inflection_skip_types": []}
    assert g.settings_from_config(config)["inflection_skip_types"] == ()
    book = tmp_path / "book"
    book.mkdir()
    (book / "glossary.csv").write_text("type,raw_name,translated_name\nnicknames,늑대들,Wolves of the North\n",
                                       encoding="utf-8")
    assert g.load_entries(str(book / "glossary.csv"), mode="all").inflection_skip == {
        "character", "characters", "nicknames", "nickname"}
    assert g.load_entries(str(book / "glossary.csv"), mode="all", inflection_skip_types=()).inflection_skip == set()


def test_plural_and_tense_exclusion_in_the_check(tmp_path):
    """2026-10-28 (owner): "Heroes of the Sword → Hero of the Sword" and "Beastmen → Beastman" were reported."""
    rows = [{"type": "titles", "raw_name": "검의 용사", "translated_name": "Hero of the Sword"},
            {"type": "terms", "raw_name": "수인", "translated_name": "Beastman"},
            {"type": "terms", "raw_name": "타락천사", "translated_name": "Fallen Angel"},
            {"type": "character", "raw_name": "아리아", "translated_name": "Fallen Aria"}]
    output = "<p>Weapons of generations of Heroes of the Sword. The Beastmen ran. A Falling Angel. Falling Aria.</p>"
    text = g.html_to_text(output)
    tagger = _fake_tagger(tmp_path, text, {"Heroes": 5, "of": 6, "the": 6, "Sword": 6, "Beastmen": 1, "Falling": 3,
                                           "Angel": 4, "Aria": 4})
    source = "역대 검의 용사들 수인들 타락천사 아리아"

    def kinds(**settings):
        ents = g.build_entries(rows, mode="all", **settings)
        return {f.expected: f.kind for f in g.check_chapter(source, output, ents, ner=tagger).findings}

    old = kinds(plural_exclusion=False)
    assert old["Hero of the Sword"] != "missing" and old["Beastman"] != "missing"  # reported before
    new = kinds()  # plural exclusion on (default), tense off
    assert "Hero of the Sword" not in new and "Beastman" not in new  # found: the plural is the name
    assert new["Fallen Angel"] != "missing"  # tense exclusion is off by default
    tense = kinds(tense_exclusion=True)
    assert "Fallen Angel" not in tense
    assert tense["Fallen Aria"] != "missing"  # never for a character name


def test_plural_and_tense_exclusion_only_for_english_output():
    assert g.english_output(None) and g.english_output("English") and g.english_output("english (US)")
    assert not g.english_output("French")
    on = {"GLOSSARY_NAME_CHECK_PLURAL_EXCLUSION": "1", "GLOSSARY_NAME_CHECK_TENSE_EXCLUSION": "1"}
    assert g.settings_from_env(on)["plural_exclusion"] and g.settings_from_env(on)["tense_exclusion"]
    scope = g.settings_from_env(dict(on, OUTPUT_LANGUAGE="Korean"))
    assert not scope["plural_exclusion"] and not scope["tense_exclusion"]
    assert not g.settings_from_config({"output_language": "Spanish"}, on)["plural_exclusion"]
    assert g.settings_from_env({})["plural_exclusion"] and not g.settings_from_env({})["tense_exclusion"]


def test_chapters_at_once_go_up_to_the_cores_and_share_them_out(monkeypatch):
    monkeypatch.setattr(os, "cpu_count", lambda: 16)
    assert g.max_chapter_workers() == 16 and g.chapter_workers(object(), chosen=99) == 16
    assert g.workers_from_config({"glossary_name_check_workers": 20}) == 20  # saved values up to 32
    assert [ner_names.ner_threads(n) for n in (1, 2, 4, 8, 16)] == [4, 4, 4, 2, 1]
    monkeypatch.setattr(os, "cpu_count", lambda: 64)
    assert g.max_chapter_workers() == g.MAX_CHAPTER_WORKERS == 32


def test_scan_folder_shares_the_cores_out_between_the_checks(tmp_path, monkeypatch):
    monkeypatch.setattr(os, "cpu_count", lambda: 16)
    book = _book(tmp_path, "<p>Walter Argent smiled.</p>")
    asked = []

    class Tagger:
        threads = 4

        def with_threads(self, n):
            asked.append(n)
            return self

        def entities(self, text):
            return []

    ents = g.load_entries(str(book / "glossary.csv"))
    g.scan_folder(str(book), ents, ner=Tagger(), workers=8)
    g.scan_folder(str(book), ents, ner=Tagger(), workers=1)
    assert asked == [2]  # 8 at once: 16 cores / 8; one at a time keeps the tagger as it is


def test_numbers_must_match_and_a_number_glued_to_a_word_is_present():
    """2026-10-27 (owner): "slain one of the72 Demon Kings" (a missing space) was reported as "Demon Kings → 72
    Demon Kings", whose fix would have written "the7272 Demon Kings"."""
    rows = [{"type": "term", "raw_name": "72마왕", "translated_name": "72 Demon Kings"}]
    ents = g.build_entries(rows, mode="all")
    source = "아카데미 시절부터 72마왕 중 하나의 목을 베고"
    assert g.check_chapter(source, "<p>slain one of the72 Demon Kings while still at the academy</p>", ents).findings == []
    # a candidate with another number, or none, is never a respelling (_norm drops digits)
    for output in ("<p>He slew one of the Demon Kings.</p>", "<p>He slew one of the 73 Demon Kings.</p>"):
        assert [f.kind for f in g.check_chapter(source, output, ents).findings] == ["missing"], output
    # "Names keep their numbers" off (glossary_name_check_match_numbers): the comparison ignores numbers
    loose = g.build_entries(rows, mode="all", match_numbers=False)
    [finding] = g.check_chapter(source, "<p>slain one of the72 Demon Kings</p>", loose).findings
    assert finding.variants == ("Demon Kings",)
    # the same number with other spacing still is one
    assert g._is_variant("72 Demon Kings", "demonkings", "72 Demon-Kings", "demonkings")
    assert not g._is_variant("72 Demon Kings", "demonkings", "Demon Kings", "demonkings")


def test_missing_names_are_not_reportable():
    result = g.check_chapter("발터 아르겐트", "<p>He nodded.</p>", entries())
    assert [f.kind for f in result.findings] == ["missing"]
    assert result.reportable == []


def test_two_spellings_are_ambiguous_and_not_auto_fixed():
    result = g.check_chapter("발터 아르겐트", "<p>Walter Argent and Valtor Argent.</p>", entries())
    [finding] = result.findings
    assert finding.kind == "ambiguous" and not finding.fixable
    assert g.apply_fixes("<p>Walter Argent</p>", result.findings) == ("<p>Walter Argent</p>", 0)
    assert g.apply_fixes("<p>Walter Argent</p>", result.findings, reviewed=True)[1] == 1


def test_scope_follows_emergency_glossary_compliance_types_and_min_length():
    assert "칼" not in [e.raw for e in entries().entries]                  # shorter than 3
    assert "칼" in [e.raw for e in entries(min_chars=0).entries]
    assert "아벨란 제국" not in [e.raw for e in entries().entries]          # characters only
    found = findings("아벨란 제국", "<p>The Abellan Empire fell.</p>", mode="all")
    assert found[0].kind == "variant" and not found[0].fixable              # report only
    assert g.entry_type_allowed("term", "custom", ["terms"])
    assert not g.entry_type_allowed("term", "characters")


def test_longer_raw_name_covers_its_parts():
    rows = GLOSSARY_ROWS + [{"type": "character", "raw_name": "아르겐트", "translated_name": "Argente"}]
    ents = g.build_entries(rows)
    result = g.check_chapter("발터 아르겐트", "<p>Valter Argent.</p>", ents)
    assert result.findings == []


def _book(tmp_path, output_html, *, glossary=True):
    (tmp_path / "chapters_full.json").write_text(json.dumps([
        {"content_hash": "h1", "original_basename": "chapter0001.xhtml",
         "body": "<p>발터 아르겐트가 웃었다.</p>"}]), encoding="utf-8")
    (tmp_path / "translation_progress.json").write_text(json.dumps({"chapters": {
        "1": {"status": "completed", "output_file": "response_chapter0001.html",
              "content_hash": "h1", "original_basename": "chapter0001.xhtml"}}}), encoding="utf-8")
    (tmp_path / "response_chapter0001.html").write_text(output_html, encoding="utf-8")
    if glossary:
        (tmp_path / "glossary.csv").write_text(
            "type,raw_name,translated_name\ncharacter,발터 아르겐트,Valter Argent\n", encoding="utf-8")
    return tmp_path


def test_scan_folder_and_fix_files_back_up_first(tmp_path):
    book = _book(tmp_path, "<p>Walter Argent smiled.</p>")
    ents = g.load_entries(str(book / "glossary.csv"))
    [chapter] = g.scan_folder(str(book), ents)
    assert chapter.output_file == "response_chapter0001.html"
    changed = g.fix_files(str(book), {chapter.output_file: chapter.reportable})
    assert changed == {"response_chapter0001.html": 1}
    assert (book / "response_chapter0001.html").read_text(encoding="utf-8") == "<p>Valter Argent smiled.</p>"
    backup = book / "glossary_name_check_backup" / "response_chapter0001.html"
    assert backup.read_text(encoding="utf-8") == "<p>Walter Argent smiled.</p>"


def test_settings_from_env_and_config():
    # entry types are Emergency Glossary Compliance's; the minimum length is the name check's own (default 2)
    assert g.settings_from_env({"EMERGENCY_GLOSSARY_COMPLIANCE_MIN_CHARS": "5"})["min_chars"] == 2
    env = {"EMERGENCY_GLOSSARY_COMPLIANCE_MODE": "custom",
           "EMERGENCY_GLOSSARY_COMPLIANCE_CUSTOM_TYPES": '["term"]',
           "EMERGENCY_GLOSSARY_COMPLIANCE_MIN_CHARS": "5",
           "GLOSSARY_NAME_CHECK_MIN_CHARS": "2"}
    assert g.settings_from_env(env) == {"mode": "custom", "custom_types": ("term",), "min_chars": 2,
                                        "autofix_level": 1, "match_numbers": True, "plural_exclusion": True,
                                        "tense_exclusion": False, "inflection_skip_types": ("character", "nicknames")}
    assert g.settings_from_config({"emergency_glossary_compliance_mode": "all",
                                   "emergency_glossary_compliance_min_chars": 5,
                                   "glossary_name_check_min_chars": 4,
                                   "glossary_name_check_autofix_level": 3}, env) == {
        "mode": "all", "custom_types": (), "min_chars": 4, "autofix_level": 3, "match_numbers": True,
        "plural_exclusion": True, "tense_exclusion": False, "inflection_skip_types": ("character", "nicknames")}
    assert g.settings_from_env({"GLOSSARY_NAME_CHECK_MATCH_NUMBERS": "0"})["match_numbers"] is False
    assert g.settings_from_config({"glossary_name_check_match_numbers": False}, {})["match_numbers"] is False
    assert g.settings_from_env({"GLOSSARY_NAME_CHECK_AUTOFIX_LEVEL": "9"})["autofix_level"] == 3


def test_short_raw_names_are_only_fixed_for_spacing_respellings():
    rows = [{"type": "character", "raw_name": "박", "translated_name": "Park"},
            {"type": "character", "raw_name": "소윤", "translated_name": "So-yoon"}]
    ents = g.build_entries(rows, min_chars=0)
    found = {f.expected: f for f in g.check_chapter("박 소윤", "<p>The Sparks flew and Soyoon smiled.</p>", ents).findings}
    assert found["Park"].kind == "variant" and found["Park"].variants == ("Sparks",)
    assert not found["Park"].fixable            # 1-character raw name, fuzzy match: report only
    assert found["So-yoon"].fixable             # 2 characters, but only the hyphen differs
    assert not g.enabled_from_env({}) and g.enabled_from_env({"GLOSSARY_NAME_CHECK": "1"})
    assert g.auto_fix_from_env({}) and not g.auto_fix_from_env({"GLOSSARY_NAME_CHECK_MODE": "review"})


@pytest.mark.parametrize("mode,expected", [("auto", "Valter"), ("review", "Walter")])
def test_translation_hook_fixes_or_only_logs(tmp_path, monkeypatch, capsys, mode, expected):
    import TransateKRtoEN as t
    book = _book(tmp_path, "<p>Walter Argent smiled.</p>")
    monkeypatch.setenv("GLOSSARY_NAME_CHECK", "1")
    monkeypatch.setenv("GLOSSARY_NAME_CHECK_MODE", mode)
    monkeypatch.setattr(t, "find_glossary_file", lambda out_dir: str(book / "glossary.csv"))
    t._run_glossary_name_check(str(book), "response_chapter0001.html",
                               {"body": "<p>발터 아르겐트가 웃었다.</p>"})
    assert (book / "response_chapter0001.html").read_text(encoding="utf-8") == f"<p>{expected} Argent smiled.</p>"
    assert "Walter Argent → Valter Argent" in capsys.readouterr().out


def test_translation_hook_is_off_by_default(tmp_path, monkeypatch):
    import TransateKRtoEN as t
    book = _book(tmp_path, "<p>Walter Argent smiled.</p>")
    monkeypatch.delenv("GLOSSARY_NAME_CHECK", raising=False)
    t._run_glossary_name_check(str(book), "response_chapter0001.html", {"body": "발터 아르겐트"})
    assert "Walter" in (book / "response_chapter0001.html").read_text(encoding="utf-8")


def test_qa_scan_logs_what_finds_the_names_and_its_progress(tmp_path):
    """2026-10-27 (owner): the QA scan's name check logged one line, then its result."""
    import scan_html_folder as s
    book = _book(tmp_path, "<p>Walter Argent smiled.</p>")
    lines = []
    s.run_glossary_name_check(str(book), [{"filename": "response_chapter0001.html", "issues": []}],
                              {"_glossary_name_scope": {"mode": "characters", "custom_types": (), "min_chars": 3}},
                              log=lines.append)
    assert "   🔎 Names found with the capitalization heuristic" in lines
    assert "   📖 Glossary name check: 1/1 chapters (100%)" in lines
    assert lines.index("   📖 Glossary name check: 1/1 chapters (100%)") < len(lines) - 1  # before the result


def test_ner_status_names_what_finds_the_names(monkeypatch):
    monkeypatch.setattr(ner_names, "runtime_available", lambda: True)
    monkeypatch.setattr(ner_names, "model_installed", lambda: True)
    assert g.ner_status(True) == "installed" and g.ner_status(False) == "off"
    monkeypatch.setattr(ner_names, "model_installed", lambda: False)
    assert g.ner_status(True) == "missing"
    monkeypatch.setattr(ner_names, "runtime_available", lambda: False)
    assert g.ner_status(True) == "unsupported"


def test_qa_scanner_adds_issues(tmp_path):
    import scan_html_folder as s
    book = _book(tmp_path, "<p>Walter Argent smiled.</p>")
    results = [{"filename": "response_chapter0001.html", "issues": []}]
    flagged = s.run_glossary_name_check(str(book), results, {"_glossary_name_scope": {
        "mode": "characters", "custom_types": (), "min_chars": 3}}, log=lambda *_: None)
    assert flagged == 1
    assert results[0]["issues"] == ["glossary_name_variant (Walter Argent → Valter Argent)"]
    assert results[0]["score"] == 1


def test_qa_setting_defaults_off_and_exports_env():
    import qa_scan_runtime as q
    settings = q.normalize_qa_scan_settings({})
    assert settings["check_glossary_names"] is False
    env = q.apply_qa_scan_env_from_settings({"check_glossary_names": True})
    try:
        assert os.environ["QA_CHECK_GLOSSARY_NAMES"] == "1"
    finally:
        q.restore_env(env)


# --------------------------------------------------------------------------------------------- NER
def test_basic_words_and_wordpiece(tmp_path):
    text = "Yoo So-jeong's sword, 林水!"
    assert [text[a:b] for a, b in ner_names.basic_words(text)] == \
        ["Yoo", "So", "-", "jeong", "'", "s", "sword", ",", "林", "水", "!"]
    vocab = tmp_path / "vocab.txt"
    vocab.write_text("\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "Wal", "##ter", "Ar", "##gent"]), encoding="utf-8")
    wp = ner_names.WordPiece(str(vocab))
    assert wp.pieces("Walter") == [4, 5] and wp.pieces("Argent") == [6, 7] and wp.pieces("Xyz") == [1]


class _FakeSession:
    """Labels each word by a fixed map (first sub-token), like the CoNLL model would."""

    def __init__(self, tags):
        self.tags = tags  # list of label ids, one per token position after [CLS]

    def run(self, _outputs, feed):
        import numpy as np
        ids = feed["input_ids"][0]
        logits = np.full((1, len(ids), 9), -5.0, dtype=np.float32)
        for position in range(len(ids)):
            label = self.tags[position - 1] if 0 < position <= len(self.tags) else 0
            logits[0, position, label] = 5.0
        return [logits]


def _fake_tagger(tmp_path, text, word_labels):
    import numpy as np  # noqa: F401  (NerTagger needs numpy)
    words = [text[a:b] for a, b in ner_names.basic_words(text)]
    vocab = tmp_path / "vocab.txt"
    vocab.write_text("\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]"] + sorted(set(words))), encoding="utf-8")
    tagger = ner_names.NerTagger.__new__(ner_names.NerTagger)
    tagger._np = np
    tagger.wordpiece = ner_names.WordPiece(str(vocab))
    tagger.labels = ["O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
    tagger.session = _FakeSession([word_labels.get(w, 0) for w in words])
    tagger.inputs = {"input_ids", "attention_mask", "token_type_ids"}
    import threading
    tagger._lock = threading.Lock()
    tagger._cache = {}
    return tagger


def test_tagger_merges_words_and_hyphenated_names(tmp_path):
    text = "Walter Argent met Yoo So-jeong in Lumina"
    tagger = _fake_tagger(tmp_path, text, {"Walter": 3, "Argent": 4, "Yoo": 3, "So": 4, "jeong": 4, "Lumina": 7})
    assert [(e.label, e.text) for e in tagger.entities(text)] == [
        ("PER", "Walter Argent"), ("PER", "Yoo So-jeong"), ("LOC", "Lumina")]


def test_ner_candidates_find_sentence_initial_names(tmp_path):
    # the heuristic skips a single capitalized word at a sentence start; NER does not
    output = "<p>He left.</p><p>Walther drew his sword.</p>"
    rows = [{"type": "character", "raw_name": "발터", "translated_name": "Walter"}]
    ents = g.build_entries(rows)
    assert [f.kind for f in g.check_chapter("발터", output, ents).findings] == ["missing"]
    text = g.html_to_text(output)
    tagger = _fake_tagger(tmp_path, text, {"Walther": 3})
    [finding] = g.check_chapter("발터", output, ents, ner=tagger).findings
    assert (finding.kind, finding.variants) == ("variant", ("Walther",))


def test_with_ner_its_names_are_the_only_candidates(tmp_path):
    """2026-10-27 (owner): with NER on, the check uses the model alone, never the capitalization heuristic."""
    output = "<p>Walter Argent smiled.</p>"
    assert [f.kind for f in findings("발터 아르겐트", output)] == ["variant"]  # the heuristic finds it
    text = g.html_to_text(output)
    untagged = _fake_tagger(tmp_path, text, {})
    assert [f.kind for f in g.check_chapter("발터 아르겐트", output, entries(), ner=untagged).findings] == ["missing"]
    tagged = _fake_tagger(tmp_path, text, {"Walter": 3, "Argent": 4})
    [finding] = g.check_chapter("발터 아르겐트", output, entries(), ner=tagged).findings
    assert (finding.kind, finding.variants) == ("variant", ("Walter Argent",))


def test_a_word_inside_an_ner_name_is_only_a_respelling(tmp_path):
    """A glossary entry is often a given name alone: "Min-ho" in "Lee Min-ho" is drift of "Minho", but a
    close spelling inside a longer name is another name ("Mikiyo" in "Sawayanagi Mikiyo" for "Miki")."""
    rows = [{"type": "character", "raw_name": "민호", "translated_name": "Minho"},
            {"type": "character", "raw_name": "미키", "translated_name": "Miki"}]
    ents = g.build_entries(rows, min_chars=0)
    output = "<p>Lee Min-ho met Sawayanagi Mikiyo.</p>"
    text = g.html_to_text(output)
    tagger = _fake_tagger(tmp_path, text, {"Lee": 3, "Min": 4, "ho": 4, "Sawayanagi": 3, "Mikiyo": 4})
    found = {f.expected: f for f in g.check_chapter("이민호 사와야나기 미키요", output, ents, ner=tagger).findings}
    assert (found["Minho"].kind, found["Minho"].variants) == ("variant", ("Min-ho",))
    assert found["Miki"].kind == "missing"
    # in a title every run of words is a full candidate (Title Case pulls words into the name)
    title = "Chapter 3: Sad Mikiyo"
    title_tagger = _fake_tagger(tmp_path, title, {"Sad": 3, "Mikiyo": 4})
    [finding] = g.check_chapter("미키", f"<p>{title}</p>", g.build_entries(rows[1:], min_chars=0),
                                ner=title_tagger, title=True).findings
    assert (finding.kind, finding.variants) == ("variant", ("Mikiyo",))


def test_tagger_keeps_the_part_after_a_hyphen_or_apostrophe(tmp_path):
    # the real model labels the joined part "O" or another type: "Yoo So", "Ju", "Tu Zhi", "Yuki"
    text = "Yoo So-jeong met Ju-in, Tu Zhi'an's son and the Yuki-onna at Walter's inn"
    tagger = _fake_tagger(tmp_path, text, {"Yoo": 3, "So": 4, "Ju": 3, "Tu": 3, "Zhi": 4, "Yuki": 1, "onna": 7,
                                           "Walter": 3})
    assert [(e.label, e.text) for e in tagger.entities(text)] == [
        ("PER", "Yoo So-jeong"), ("PER", "Ju-in"), ("PER", "Tu Zhi'an"), ("MISC", "Yuki-onna"), ("PER", "Walter")]


class _ProbSession(_FakeSession):
    """Per-word label probabilities ({label id: p}, the rest of the mass on "O") instead of sure labels."""

    def run(self, _outputs, feed):
        import numpy as np
        ids = feed["input_ids"][0]
        probs = np.full((1, len(ids), 9), 1e-6, dtype=np.float64)
        for position in range(len(ids)):
            word = self.tags[position - 1] if 0 < position <= len(self.tags) else {}
            for label, p in word.items():
                probs[0, position, label] = p
            probs[0, position, 0] = max(1e-6, 1.0 - sum(word.values()))
        return [np.log(probs).astype(np.float32)]


def test_tagger_scores_a_word_by_how_sure_it_is_a_name(tmp_path):
    """Corpus titles (2026-10-27): the model is often unsure of the type only ("Mirena Falls": place 0.58),
    keeps a name across a type switch ("World Tree El" misc + "Lamina" I-LOC), and a capitalized word it
    starts another entity with stays out of the name before it ("Lee Da-yun-Style")."""
    text = "Tasty Mirena met Lee Da-yun-Style near Mirena Falls by the World Tree El Lamina"
    probs = {"Tasty": {3: 0.79}, "Mirena": {4: 0.35, 3: 0.25},  # "Mirena": O 0.40 is the single best label
             "Lee": {3: 1.0}, "Da": {4: 1.0}, "yun": {4: 0.94}, "Style": {1: 0.88},
             "Falls": {8: 0.35, 2: 0.3, 4: 0.23}, "World": {1: 0.7}, "Tree": {2: 0.98},
             "El": {2: 0.64, 8: 0.2}, "Lamina": {8: 0.47, 2: 0.4}}
    tagger = _fake_tagger(tmp_path, text, {})
    words = [text[a:b] for a, b in ner_names.basic_words(text)]
    tagger.session = _ProbSession([probs.get(w, {}) for w in words])
    assert [(e.label, e.text) for e in tagger.entities(text)] == [
        ("PER", "Tasty Mirena"), ("PER", "Lee Da-yun"), ("MISC", "Style"), ("PER", "Mirena Falls"),
        ("MISC", "World Tree El Lamina")]


def test_resolve_ner_falls_back_when_unsupported(monkeypatch):
    monkeypatch.setattr(ner_names, "runtime_available", lambda: False)
    monkeypatch.setattr(g, "_ner_notes", set())
    logs = []
    assert g.resolve_ner(True, logs.append) is None and g.resolve_ner(True, logs.append) is None
    assert logs == ["ℹ️ Glossary name check: NER is not supported in this build; using the capitalization heuristic"]
    assert g.resolve_ner(False, logs.append) is None
    assert g.use_ner_from_env({}) is True and g.use_ner_from_env({"GLOSSARY_NAME_CHECK_USE_NER": "0"}) is False
    assert g.use_ner_from_config({"glossary_name_check_use_ner": False}, {}) is False


@pytest.mark.skipif(not (ner_names.runtime_available() and ner_names.model_installed()),
                    reason="the NER model is not downloaded here (ONNX_CACHE_DIR)")
def test_real_model_finds_names():
    tagger = ner_names.get_tagger()
    found = {(e.label, e.text) for e in tagger.entities("Walter Argent drew his sword while Gang Si-hyeon watched.")}
    assert {("PER", "Walter Argent"), ("PER", "Gang Si-hyeon")} <= found


def test_every_chapter_completion_runs_the_name_check():
    """2026-10-23 (Glossarion Mobile run fixed nothing): ProgressManager.update runs the check only when it
    gets chapter_obj; a completion without it (the batch / parallel chapter path) must call it itself."""
    import ast

    tree = ast.parse(open(os.path.join(SRC, "TransateKRtoEN.py"), encoding="utf-8-sig").read())
    missing = []
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call)]
        runs_check = any(getattr(c.func, "id", None) == "_run_glossary_name_check" for c in calls)
        for call in calls:
            if (getattr(call.func, "attr", None) or getattr(call.func, "id", None)) not in ("update", "update_progress_fn"):
                continue
            kws = {k.arg: k.value for k in call.keywords}
            status = kws.get("status")
            if isinstance(status, ast.Constant) and status.value == "completed" and "chapter_obj" not in kws \
                    and not runs_check:
                missing.append((fn.name, call.lineno))
    assert not missing, missing


# ---- chapter titles, TOC and book metadata ---------------------------------------------------------

HEADERS = ("Chapter Header Translations\r\n" + "=" * 50 + "\r\n\r\n"
           "Chapter 1:\r\n  Original:   발터 아르겐트의 검\r\n  Translated: Walter Argent's Sword\r\n"
           "  Output File: chapter0001\r\n" + "-" * 40 + "\r\n"
           "Chapter 2:\r\n  Original:   새로운 아침\r\n  Translated: A New Morning\r\n"
           "  Output File: chapter0002\r\n" + "-" * 40 + "\r\n")
TOC = ("TOC Translations\n" + "=" * 50 + "\n\n"
       "Chapter 1:\n  Original:   발터 아르겐트의 검\n  Translated: Walter Argent's Sword\n"
       "  Target URI: response_chapter0001.html\n" + "-" * 40 + "\n")


def _titled_book(tmp_path):
    book = _book(tmp_path, "<h1>Walter Argent's Sword</h1><p>Valter Argent smiled.</p>")
    (book / "translated_headers.txt").write_bytes(HEADERS.encode("utf-8"))
    (book / "TOC.txt").write_text(TOC, encoding="utf-8")
    (book / "metadata.json").write_text(json.dumps({
        "title": "The Knight Walter Argent", "original_title": "기사 발터 아르겐트",
        "description": "A story.", "original_description": "이야기."}, ensure_ascii=False), encoding="utf-8")
    return book, g.load_entries(str(book / "glossary.csv"))


def test_title_rows_parse_headers_toc_and_metadata():
    rows = g.title_rows(HEADERS)
    assert [(r.key, r.original, r.translated, r.output_ref) for r in rows] == [
        ("1", "발터 아르겐트의 검", "Walter Argent's Sword", "chapter0001"),
        ("2", "새로운 아침", "A New Morning", "chapter0002")]
    meta = g.metadata_rows({"title": "T", "original_title": "원", "creator": "X", "original_creator": "X"})
    assert [(r.key, r.original, r.translated) for r in meta] == [("title", "원", "T")]


def test_pipeline_fixes_titles_toc_and_the_chapter_header(tmp_path):
    book, ents = _titled_book(tmp_path)
    results = g.check_and_fix_titles(str(book), ents, auto_fix=True, names=(g.HEADERS_FILE, g.TOC_FILE))
    assert [(r.output_file, [f.describe() for f in r.reportable], sorted(fixed)) for r, fixed in results] == [
        ("translated_headers.txt", ["Walter Argent → Valter Argent"], ["1"]),
        ("TOC.txt", ["Walter Argent → Valter Argent"], ["1"])]
    # only the Translated line changes; CRLF and the source title stay as they were
    assert (book / "translated_headers.txt").read_bytes().decode("utf-8") == HEADERS.replace(
        "Translated: Walter Argent's Sword", "Translated: Valter Argent's Sword")
    assert "Translated: Valter Argent's Sword\n" in (book / "TOC.txt").read_text(encoding="utf-8")
    # the header translated_headers.txt was written into
    assert (book / "response_chapter0001.html").read_text(encoding="utf-8").startswith("<h1>Valter Argent's Sword")
    logs = []
    g.log_title_results(results, logs.append, auto_fix=True)
    assert logs[0] == "📖 Glossary name check: translated_headers.txt: Walter Argent → Valter Argent (fixed)"


def test_review_mode_only_reports_titles(tmp_path):
    book, ents = _titled_book(tmp_path)
    [(result, fixed)] = g.check_and_fix_titles(str(book), ents, auto_fix=False, names=(g.HEADERS_FILE,))
    assert result.reportable and fixed == {}
    assert (book / "translated_headers.txt").read_bytes().decode("utf-8") == HEADERS


def test_metadata_is_fixed_in_memory(tmp_path):
    _book_dir, ents = _titled_book(tmp_path)
    metadata = {"title": "The Knight Walter Argent", "original_title": "기사 발터 아르겐트"}
    result, changed = g.fix_metadata(metadata, ents, auto_fix=True)
    assert [f.describe() for f in result.reportable] == ["Walter Argent → Valter Argent"]
    assert list(changed) == ["title"] and metadata["title"] == "The Knight Valter Argent"
    assert metadata["original_title"] == "기사 발터 아르겐트"


def test_review_folder_lists_title_files_and_fix_files_applies_them(tmp_path):
    book, _ents = _titled_book(tmp_path)
    seen = []
    review = g.review_folder(str(book), {"mode": "characters", "custom_types": (), "min_chars": 0},
                             progress=lambda done, total: seen.append((done, total)))
    assert seen == [(0, 1)]
    assert [c.output_file for c in review.chapters] == ["translated_headers.txt", "TOC.txt", "metadata.json"]
    changed = g.fix_files(str(book), {c.output_file: c.reportable for c in review.chapters})
    assert changed == {"translated_headers.txt": 1, "TOC.txt": 1, "metadata.json": 1}
    assert json.loads((book / "metadata.json").read_text(encoding="utf-8"))["title"] == "The Knight Valter Argent"
    backups = book / "glossary_name_check_backup"
    assert sorted(os.listdir(backups)) == ["TOC.txt", "metadata.json", "response_chapter0001.html",
                                           "translated_headers.txt"]
    assert (backups / "translated_headers.txt").read_bytes().decode("utf-8") == HEADERS


def test_resolve_ner_passes_download_progress_and_cancel(monkeypatch):
    calls = []
    monkeypatch.setattr(ner_names, "runtime_available", lambda: True)
    monkeypatch.setattr(ner_names, "get_tagger", lambda log=None, cancel=None, progress=None:
                        calls.append((cancel, progress)) or "tagger")
    progress, cancel = object(), object()
    assert g.resolve_ner(True, progress=progress, cancel=cancel) == "tagger"
    assert calls == [(cancel, progress)]


def test_a_failed_model_load_is_not_retried_per_chapter(monkeypatch):
    """2026-10-23: without network every chapter started the 104 MB download again."""
    calls = []

    def failing(log=None, cancel=None, progress=None):
        calls.append(1)
        raise OSError("no network")

    monkeypatch.setattr(ner_names, "runtime_available", lambda: True)
    monkeypatch.setattr(ner_names, "get_tagger", failing)
    monkeypatch.setattr(g, "_ner_notes", set())
    logs = []
    assert g.resolve_ner(True, logs.append) is None and g.resolve_ner(True, logs.append) is None
    assert len(calls) == 1 and len(logs) == 1
    assert g.resolve_ner(True, logs.append, retry=True) is None and len(calls) == 2  # the dialogs try again
    monkeypatch.setattr(ner_names, "get_tagger", lambda log=None, cancel=None, progress=None: "tagger")
    assert g.resolve_ner(True, retry=True) == "tagger" and g.resolve_ner(True) == "tagger"


# ---- parallel chapters ---------------------------------------------------------------------------

@pytest.mark.parametrize("cpus,setting,expected", [
    (16, 0, 4),     # "All cores": 16 cores / 4 threads per NER call
    (16, 8, 2),     # "Half cores"
    (16, 1, 1),     # "Single thread"
    (2, 0, 1),      # a 2-core machine: one chapter at a time, NER on both cores
    (1, 0, 1),
    (16, 64, 4),    # never more than the machine has
])
def test_chapter_workers_follow_the_qa_performance_setting(monkeypatch, cpus, setting, expected):
    monkeypatch.setattr(os, "cpu_count", lambda: cpus)
    assert ner_names.ner_threads() == min(4, cpus)
    assert g.chapter_workers(object(), setting) == expected
    assert g.chapter_workers(None, setting) == 1  # the heuristic alone is pure Python
    # "Chapters checked at once" (glossary_name_check_workers): 0 = Auto (the rule above), else that count
    assert g.chapter_workers(object(), setting, chosen=0) == expected
    assert g.chapter_workers(object(), setting, chosen=3) == min(3, cpus)  # never more than the cores
    assert g.chapter_workers(None, setting, chosen=3) == 1


def test_chapters_at_once_setting_reads_and_clamps():
    assert g.workers_from_config({}) == 0 and g.workers_from_config(None) == 0
    assert g.workers_from_config({"glossary_name_check_workers": 3}) == 3
    assert g.workers_from_config({"glossary_name_check_workers": 99}) == g.MAX_CHAPTER_WORKERS
    assert g.workers_from_config({"glossary_name_check_workers": "x"}) == 0
    assert g.chapter_workers(object(), chosen=99) == g.max_chapter_workers()  # the cores


def test_parallel_scan_keeps_file_order_and_results(tmp_path):
    chapters, progress = [], {"chapters": {}}
    for i in range(1, 9):
        chapters.append({"content_hash": f"h{i}", "original_basename": f"c{i}.xhtml",
                         "body": "<p>발터 아르겐트가 웃었다.</p>"})
        progress["chapters"][str(i)] = {"status": "completed", "output_file": f"response_c{i:02}.html",
                                        "content_hash": f"h{i}"}
        (tmp_path / f"response_c{i:02}.html").write_text(
            "<p>Walter Argent smiled.</p>" if i % 2 else "<p>Valter Argent smiled.</p>", encoding="utf-8")
    (tmp_path / "chapters_full.json").write_text(json.dumps(chapters), encoding="utf-8")
    (tmp_path / "translation_progress.json").write_text(json.dumps(progress), encoding="utf-8")
    ents = g.build_entries(GLOSSARY_ROWS)
    seen = []
    one = g.scan_folder(str(tmp_path), ents, workers=1)
    many = g.scan_folder(str(tmp_path), ents, workers=4, progress=lambda done, total: seen.append(total))
    assert [r.output_file for r in many] == [r.output_file for r in one] == [
        "response_c01.html", "response_c03.html", "response_c05.html", "response_c07.html"]
    assert [[f.describe() for f in r.reportable] for r in many] == [["Walter Argent → Valter Argent"]] * 4
    assert seen == [8] * 8


# ---- which glossaries: the book's, then the enabled extension / unified glossary -----------------

def test_names_come_from_the_enabled_glossaries_in_request_priority(tmp_path, monkeypatch):
    """2026-10-23 (owner): only the glossaries a request uses, per novel > extension > unified."""
    import unified_glossary

    book = tmp_path / "glossary.csv"
    book.write_text("type,raw_name,translated_name\ncharacter,발터 아르겐트,Valter Argent\n", encoding="utf-8")
    (tmp_path / "glossary_extension.csv").write_text(
        "type,raw_name,translated_name\ncharacter,발터 아르겐트,Walter Argent\ncharacter,강시현,Kang Si-hyeon\n",
        encoding="utf-8")
    unified = tmp_path / "glossary_unified.csv"
    unified.write_text("type,raw_name,translated_name\ncharacter,강시현,Gang Si-hyeon\n"
                       "character,유소정,Yoo Sojeong\n", encoding="utf-8")
    monkeypatch.setattr(unified_glossary, "resolve_prompt_glossary_path", lambda path, settings=None: str(unified))

    def names(add, unify):
        ents = g.load_entries(str(book), settings={"ADD_ADDITIONAL_GLOSSARY": add, "ENABLE_UNIFIED_GLOSSARY": unify})
        return {(e.raw, e.translated, e.origin) for e in ents.entries}

    assert names("0", "0") == {("발터 아르겐트", "Valter Argent", "")}
    assert names("1", "0") == {("발터 아르겐트", "Valter Argent", ""),             # the book wins
                               ("강시현", "Kang Si-hyeon", g.LAYER_EXTENSION)}
    assert names("0", "1") == {("발터 아르겐트", "Valter Argent", ""),
                               ("강시현", "Gang Si-hyeon", g.LAYER_UNIFIED), ("유소정", "Yoo Sojeong", g.LAYER_UNIFIED)}
    assert names("1", "1") == {("발터 아르겐트", "Valter Argent", ""),
                               ("강시현", "Kang Si-hyeon", g.LAYER_EXTENSION),  # the extension beats unified
                               ("유소정", "Yoo Sojeong", g.LAYER_UNIFIED)}
    [finding] = g.check_chapter("유소정이 웃었다.", "<p>Yoo So-jung smiled.</p>",
                                g.load_entries(str(book), settings={"ENABLE_UNIFIED_GLOSSARY": "1"})).reportable
    assert finding.origin == g.LAYER_UNIFIED
    assert g.layer_settings_from_config({"enable_unified_glossary": True})["ENABLE_UNIFIED_GLOSSARY"] == "1"
    assert g.layer_settings_from_config({})["ADD_ADDITIONAL_GLOSSARY"] == "0"


def test_preview_snippets_show_each_occurrence_in_context():
    finding = g.Finding("발터 아르겐트", "Valter Argent", "character", g.KIND_VARIANT, ("Walter Argent",), 2)
    markup = "<p>First, Walter Argent smiled.</p><p title='Walter Argent'>Then Walter Argent's sword fell.</p>"
    assert g.finding_snippets(markup, finding, width=10) == [
        ("First, ", "Walter Argent", " smiled.…"), ("…ed. Then ", "Walter Argent", "'s sword f…")]  # not the attribute
    assert g.source_snippets("그때 발터 아르겐트가 웃었다.", finding, width=3) == [("그때 ", "발터 아르겐트", "가 웃…")]


# ---- Auto-fix strength and the book report -----------------------------------------------------

@pytest.mark.parametrize("level,expected", [
    (1, {"character long"}),
    (2, {"character long", "location long"}),
    (3, {"character long", "location long", "character short", "character two"}),
])
def test_autofix_strength_levels(level, expected):
    findings = {
        "character long": g.Finding("발터 아르겐트", "Valter Argent", "character", g.KIND_VARIANT, ("Walter Argent",)),
        "location long": g.Finding("아벨란 제국", "Abelan Empire", "location", g.KIND_VARIANT, ("Avelan Empire",)),
        "character short": g.Finding("칼", "Carl", "character", g.KIND_VARIANT, ("Karl",)),
        "character two": g.Finding("유소정", "Yoo Sojeong", "character", g.KIND_AMBIGUOUS, ("Yoo Sojung", "Yu Sojeong")),
        "missing": g.Finding("강시현", "Kang Si-hyeon", "character", g.KIND_MISSING),
    }
    for finding in findings.values():
        finding.autofix_level = level
    assert {name for name, f in findings.items() if f.fixable} == expected
    # the level travels from the settings to the findings
    ents = g.build_entries(GLOSSARY_ROWS, autofix_level=level)
    [found] = g.check_chapter("아벨란 제국이 무너졌다.", "<p>The Avelan Empire fell.</p>",
                              g.build_entries(GLOSSARY_ROWS, mode="all", autofix_level=level)).reportable
    assert found.fixable == (level >= 2) and ents.autofix_level == level


def test_translation_records_findings_and_writes_the_report(tmp_path):
    book = _book(tmp_path, "<p>Walter Argent smiled.</p>")
    ents = g.load_entries(str(book / "glossary.csv"))
    path = str(book / "response_chapter0001.html")
    g.check_and_fix_file(path, "발터 아르겐트가 웃었다.", ents, auto_fix=False, record=True)
    report = (book / "glossary_name_check" / "report.html").read_text(encoding="utf-8")
    assert "Walter Argent</b> → <b>Valter Argent" in report and "to review" in report
    assert "<mark>Walter Argent</mark>" in report
    line = g.summary_line(str(book))
    assert line.startswith("📖 Glossary name check: 1 name(s) in 1 file(s) differ from the glossary: 0 fixed, 1 to review")
    # fixing it from the review marks it fixed
    [chapter] = g.scan_folder(str(book), ents)
    g.fix_files(str(book), {chapter.output_file: chapter.reportable})
    assert g.report_counts(str(book)) == {"files": 1, "names": 1, "fixed": 1, "review": 0}
    assert "fixed (reviewed)" in (book / "glossary_name_check" / "report.html").read_text(encoding="utf-8")
    # a clean re-check removes the file's record
    g.check_and_fix_file(path, "발터 아르겐트가 웃었다.", ents, auto_fix=False, record=True)
    assert g.report_counts(str(book))["names"] == 0
    assert g.summary_line(str(book)).startswith("📖 Glossary name check: every glossary name matches")


def test_no_record_no_report(tmp_path):
    book = _book(tmp_path, "<p>Valter Argent smiled.</p>")
    g.check_and_fix_file(str(book / "response_chapter0001.html"), "발터 아르겐트가 웃었다.",
                         g.load_entries(str(book / "glossary.csv")), auto_fix=True, record=True)
    assert not (book / "glossary_name_check").exists() and g.summary_line(str(book)) is None


# ------------------------------------------------------------------- entry types (Other Settings + QA)
_ENTRY_TYPES_SYNC = r'''
import json, sys
sys.path.insert(0, sys.argv[1])
import shiboken6
from PySide6.QtWidgets import QApplication, QComboBox, QDialog, QPushButton, QVBoxLayout, QWidget
app = QApplication.instance() or QApplication([])
import other_settings

class Owner(QWidget):
    def __init__(self):
        super().__init__()
        self.config = {"emergency_glossary_compliance_mode": "all"}

owner = Owner()
dialogs = []
for _ in range(2):  # Other Settings and the QA Scanner settings
    dlg = QDialog()
    QVBoxLayout(dlg).addWidget(other_settings.glossary_entry_types_row(owner, lambda d=dlg: d))
    dlg.show()
    dialogs.append(dlg)
combos = [d.findChild(QComboBox) for d in dialogs]
buttons = [next(b for b in d.findChildren(QPushButton) if b.text() == "Configure…") for d in dialogs]
out = {"start": [c.currentText() for c in combos], "configure_start": [b.isVisible() for b in buttons]}
combos[1].setCurrentIndex(2)  # Custom in the QA Scanner settings
out["custom"] = [c.currentText() for c in combos] + [owner.emergency_glossary_compliance_mode_var]
out["configure"] = [b.isVisible() for b in buttons]
shiboken6.delete(dialogs[1])  # the QA Scanner settings closed
combos[0].setCurrentIndex(0)
out["after_close"] = [combos[0].currentText(), owner.emergency_glossary_compliance_mode_var, buttons[0].isVisible()]
from PySide6.QtWidgets import QLabel, QSlider
owner.glossary_name_check_autofix_level_var = 2
rows = [other_settings.glossary_name_autofix_row(owner) for _ in range(2)]  # Other Settings, QA Scanner
sliders = [r.findChild(QSlider) for r in rows]
values = [r.findChildren(QLabel)[-1] for r in rows]
out["autofix_start"] = [s.value() for s in sliders] + [v.text() for v in values]
sliders[1].setValue(3)
out["autofix_moved"] = [s.value() for s in sliders] + [v.text() for v in values] + [owner.glossary_name_check_autofix_level_var]
shiboken6.delete(rows[0])
sliders[1].setValue(1)
out["autofix_after_close"] = [sliders[1].value(), values[1].text(), owner.glossary_name_check_autofix_level_var]
owner.config["glossary_name_check_workers"] = 2  # saved: the rows start from it
rows = [other_settings.glossary_name_workers_row(owner) for _ in range(2)]  # Other Settings, QA Scanner
sliders = [r.findChild(QSlider) for r in rows]
values = [r.findChildren(QLabel)[-1] for r in rows]
out["workers_start"] = [s.value() for s in sliders] + [v.text() for v in values]
sliders[0].setValue(0)
out["workers_auto"] = [s.value() for s in sliders] + [v.text().split(" (")[0] for v in values]
out["workers_auto"] += [owner.glossary_name_check_workers_var]
sliders[1].setValue(1)
out["workers_one"] = [s.value() for s in sliders] + [v.text() for v in values]
from PySide6.QtWidgets import QCheckBox
owner._create_styled_checkbox = lambda text: QCheckBox(text)
rows = [other_settings.glossary_name_numbers_row(owner) for _ in range(2)]  # Other Settings, QA Scanner
boxes = [r.findChild(QCheckBox) for r in rows]
out["numbers_start"] = [b.isChecked() for b in boxes] + [b.text() for b in boxes]
boxes[1].setChecked(False)
out["numbers_off"] = [b.isChecked() for b in boxes] + [owner.glossary_name_check_match_numbers_var]
shiboken6.delete(rows[1])
boxes[0].setChecked(True)
out["numbers_after_close"] = [boxes[0].isChecked(), owner.glossary_name_check_match_numbers_var]
seen = []
pairs = {name: [getattr(other_settings, name)(owner) for _ in range(2)]
         for name in ("glossary_name_plural_row", "glossary_name_tense_row")}
pairs["glossary_name_ner_row"] = [other_settings.glossary_name_ner_row(owner, on_change=seen.append)
                                  for _ in range(2)]
out["toggles_start"] = {n: [r.findChild(QCheckBox).isChecked() for r in rs] for n, rs in pairs.items()}
out["toggles_text"] = {n: rs[0].findChild(QCheckBox).text() for n, rs in pairs.items()}
for rs in pairs.values():
    box = rs[1].findChild(QCheckBox)
    box.setChecked(not box.isChecked())
out["toggles_flipped"] = {n: [r.findChild(QCheckBox).isChecked() for r in rs] for n, rs in pairs.items()}
out["toggles_vars"] = [owner.glossary_name_check_plural_exclusion_var, owner.glossary_name_check_tense_exclusion_var,
                       owner.glossary_name_check_use_ner_var]
out["ner_seen"] = seen
listed = []
real_exec = QDialog.exec
QDialog.exec = lambda dlg: listed.extend(cb.text() for cb in dlg.findChildren(QCheckBox)) or 0
out["dialog_listed"] = [other_settings.open_glossary_entry_types_dialog(
    owner, None, "t", "p", ["nicknames"], extra_types=["character", "nicknames", "skills"]), listed]
QDialog.exec = real_exec
asked = []
def fake_dialog(self_, parent, title, prompt, current, extra_types=()):
    asked.append([list(current), list(extra_types)])
    return ["character", "skills"]
other_settings.open_glossary_entry_types_dialog = fake_dialog
owner.custom_entry_types = {"character": {}, "titles": {}, "nicknames": {}, "skills": {}}
skip_rows = [other_settings.glossary_name_inflection_types_row(owner, lambda: None) for _ in range(2)]
summary = lambda row: row.findChildren(QLabel)[1].text()
out["skip_start"] = [summary(r) for r in skip_rows]
next(b for b in skip_rows[1].findChildren(QPushButton) if b.text() == "Configure…").click()
out["skip_after"] = [summary(r) for r in skip_rows] + [owner.glossary_name_check_inflection_skip_types_var]
out["skip_asked"] = asked
print("SYNC=" + json.dumps(out))
'''


def test_entry_types_dropdown_is_one_setting_in_other_settings_and_the_qa_scanner(tmp_path):
    """2026-10-27 (owner): the QA Scanner settings show Other Settings' entry types dropdown below their
    glossary name check toggle, and the two stay in sync."""
    pytest.importorskip("PySide6")
    import subprocess
    script = tmp_path / "entry_types_sync.py"
    script.write_text(_ENTRY_TYPES_SYNC, encoding="utf-8")
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen", GLOSSARION_HTTP_LOG="0", PYTHONIOENCODING="utf-8")
    result = subprocess.run([sys.executable, str(script), SRC], cwd=str(tmp_path), env=env, capture_output=True,
                            text=True, encoding="utf-8", errors="replace", timeout=300)
    line = next((ln for ln in result.stdout.splitlines() if ln.startswith("SYNC=")), None)
    assert line is not None, (result.returncode, result.stdout[-2000:], result.stderr[-2000:])
    out = json.loads(line[len("SYNC="):])
    assert out["start"] == ["All", "All"] and out["configure_start"] == [False, False]
    assert out["custom"] == ["Custom", "Custom", "custom"] and out["configure"] == [True, True]
    assert out["after_close"] == ["Characters", "characters", False]
    assert out["autofix_start"] == [2, 2, "2 Balanced", "2 Balanced"]
    assert out["autofix_moved"] == [3, 3, "3 Aggressive", "3 Aggressive", 3]
    assert out["autofix_after_close"] == [1, "1 Conservative", 1]
    assert out["workers_start"] == [2, 2, "2 chapters", "2 chapters"]
    assert out["workers_auto"] == [0, 0, "Auto", "Auto", 0]  # "Auto (N)": N is what Auto checks here
    assert out["workers_one"] == [1, 1, "1 chapter", "1 chapter"]
    assert out["numbers_start"] == [True, True, "Names keep their numbers", "Names keep their numbers"]
    assert out["numbers_off"] == [False, False, False]
    assert out["numbers_after_close"] == [True, True]
    # plural exclusion on, tense exclusion off ("not recommended"), Use NER on: one setting each in both rows
    assert out["toggles_start"] == {"glossary_name_plural_row": [True, True], "glossary_name_tense_row": [False, False],
                                    "glossary_name_ner_row": [True, True]}
    assert out["toggles_text"]["glossary_name_tense_row"] == "NER tense exclusion (not recommended)"
    assert out["toggles_text"]["glossary_name_plural_row"] == "NER plural exclusion"
    assert out["toggles_flipped"] == {"glossary_name_plural_row": [False, False],
                                      "glossary_name_tense_row": [True, True], "glossary_name_ner_row": [False, False]}
    assert out["toggles_vars"] == [False, True, False]
    assert out["ner_seen"] == [False, False]  # each row's own change: the QA label follows
    # "Skip plural / tense exclusion for": one setting in both rows, every configured type in its dialog
    assert out["skip_start"] == ["Character, Nicknames", "Character, Nicknames"]
    assert out["skip_after"] == ["Character, Skills", "Character, Skills", ["character", "skills"]]
    current, extra = out["skip_asked"][0]
    assert current == ["character", "nicknames"] and {"character", "nicknames", "titles", "skills"} <= set(extra)
    selected, listed = out["dialog_listed"]
    assert selected is None and listed.count("Character") == 1 and {"Nicknames", "Skills"} <= set(listed)
    # both dialogs build it from the one helper; the QA row sits right below the name check toggle
    with open(os.path.join(SRC, "other_settings.py"), encoding="utf-8") as f:
        other = f.read()
    with open(os.path.join(SRC, "QA_Scanner_GUI.py"), encoding="utf-8") as f:
        qa = f.read()
    section = other[other.index("def _create_processing_options_section"):]
    body = section[:section.index("\ndef ")]
    assert "glossary_entry_types_row(\n        self, " in body and "glossary_name_autofix_row(self)" in body
    assert "glossary_name_workers_row(self)" in body and "glossary_name_numbers_row(self)" in body
    # the toggle (with its ✓ when NER is installed), then the entry types and Auto-fix strength, then the button
    assert (qa.index("detection_layout.addWidget(check_glossary_names_row)")
            < qa.index("glossary_entry_types_row(self, lambda: dialog)")
            < qa.index("glossary_name_autofix_row(self)")
            < qa.index("glossary_name_workers_row(self)")
            < qa.index("glossary_name_numbers_row(self)")
            < qa.index("glossary_name_plural_row(self)")
            < qa.index("glossary_name_tense_row(self)")
            < qa.index("glossary_name_inflection_types_row(self, lambda: dialog)")
            < qa.index("glossary_name_ner_row(self, on_change=_refresh_ner_label)")
            < qa.index("detection_layout.addWidget(glossary_names_fix_row)"))
    for builder in ("glossary_name_plural_row(self)", "glossary_name_tense_row(self)", "glossary_name_ner_row(self)",
                    "glossary_name_inflection_types_row(\n        self, "):
        assert builder in body, builder
