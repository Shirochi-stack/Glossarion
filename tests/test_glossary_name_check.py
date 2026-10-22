"""Glossary Name Check (experimental): glossary_name_check core, translation hook, QA scanner."""
import json
import os
import sys

import pytest

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import glossary_name_check as g  # noqa: E402

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
    # entry types are Emergency Glossary Compliance's; the minimum length is the name check's own (default 0)
    assert g.settings_from_env({"EMERGENCY_GLOSSARY_COMPLIANCE_MIN_CHARS": "5"})["min_chars"] == 0
    env = {"EMERGENCY_GLOSSARY_COMPLIANCE_MODE": "custom",
           "EMERGENCY_GLOSSARY_COMPLIANCE_CUSTOM_TYPES": '["term"]',
           "EMERGENCY_GLOSSARY_COMPLIANCE_MIN_CHARS": "5",
           "GLOSSARY_NAME_CHECK_MIN_CHARS": "2"}
    assert g.settings_from_env(env) == {"mode": "custom", "custom_types": ("term",), "min_chars": 2}
    assert g.settings_from_config({"emergency_glossary_compliance_mode": "all",
                                   "emergency_glossary_compliance_min_chars": 5,
                                   "glossary_name_check_min_chars": 4}, env) == {
        "mode": "all", "custom_types": (), "min_chars": 4}


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
