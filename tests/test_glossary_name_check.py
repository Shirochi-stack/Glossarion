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
