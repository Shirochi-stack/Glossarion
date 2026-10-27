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
    # entry types are Emergency Glossary Compliance's; the minimum length is the name check's own (default 2)
    assert g.settings_from_env({"EMERGENCY_GLOSSARY_COMPLIANCE_MIN_CHARS": "5"})["min_chars"] == 2
    env = {"EMERGENCY_GLOSSARY_COMPLIANCE_MODE": "custom",
           "EMERGENCY_GLOSSARY_COMPLIANCE_CUSTOM_TYPES": '["term"]',
           "EMERGENCY_GLOSSARY_COMPLIANCE_MIN_CHARS": "5",
           "GLOSSARY_NAME_CHECK_MIN_CHARS": "2"}
    assert g.settings_from_env(env) == {"mode": "custom", "custom_types": ("term",), "min_chars": 2,
                                        "autofix_level": 1}
    assert g.settings_from_config({"emergency_glossary_compliance_mode": "all",
                                   "emergency_glossary_compliance_min_chars": 5,
                                   "glossary_name_check_min_chars": 4,
                                   "glossary_name_check_autofix_level": 3}, env) == {
        "mode": "all", "custom_types": (), "min_chars": 4, "autofix_level": 3}
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
    # both dialogs build it from the one helper; the QA row sits right below the name check toggle
    with open(os.path.join(SRC, "other_settings.py"), encoding="utf-8") as f:
        other = f.read()
    with open(os.path.join(SRC, "QA_Scanner_GUI.py"), encoding="utf-8") as f:
        qa = f.read()
    section = other[other.index("def _create_processing_options_section"):]
    assert "glossary_entry_types_row(\n        self, " in section[:section.index("\ndef ")]
    assert (qa.index("detection_layout.addWidget(check_glossary_names_checkbox)")
            < qa.index("glossary_entry_types_row(self, lambda: dialog)")
            < qa.index("detection_layout.addWidget(glossary_names_fix_row)"))
