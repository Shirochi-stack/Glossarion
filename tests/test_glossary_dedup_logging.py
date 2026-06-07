from extract_glossary_from_epub import skip_duplicate_entries


def _pass1_summary(output):
    return next(line for line in output.splitlines() if "PASS 1 complete" in line)


def _dedup_env(monkeypatch):
    monkeypatch.setenv("GLOSSARY_USE_ADVANCED_DETECTION", "0")
    monkeypatch.setenv("GLOSSARY_FUZZY_THRESHOLD", "0.90")
    monkeypatch.setenv("GLOSSARY_DEDUPE_TRANSLATIONS", "0")


def test_dedup_pass1_log_reports_normal_duplicates(capsys, monkeypatch):
    _dedup_env(monkeypatch)
    entries = [
        {"type": "term", "raw_name": "Alpha", "translated_name": "Alpha EN"},
        {"type": "term", "raw_name": "Alpha", "translated_name": "Alpha Alt"},
        {"type": "term", "raw_name": "Beta", "translated_name": "Beta EN"},
    ]

    result = skip_duplicate_entries(entries)

    assert len(result) == 2
    summary = _pass1_summary(capsys.readouterr().out)
    assert "1 duplicates removed (2 remaining)" in summary
    assert "skipped due to missing raw_name" not in summary


def test_dedup_pass1_log_reports_all_missing_raw_names(capsys, monkeypatch):
    _dedup_env(monkeypatch)
    entries = [
        {"type": "term", "translated_name": "Alpha EN"},
        {"type": "term", "translated_name": "Beta EN"},
        {"type": "term", "translated_name": "Gamma EN"},
    ]

    result = skip_duplicate_entries(entries)

    assert result == []
    summary = _pass1_summary(capsys.readouterr().out)
    assert "0 duplicates removed, 3 entries skipped due to missing raw_name (0 remaining)" in summary


def test_dedup_pass1_log_reports_mixed_duplicates_and_missing_raw_names(capsys, monkeypatch):
    _dedup_env(monkeypatch)
    entries = [
        {"type": "term", "raw_name": "Alpha", "translated_name": "Alpha EN"},
        {"type": "term", "raw_name": "Alpha", "translated_name": "Alpha Alt"},
        {"type": "term", "translated_name": "Missing Raw"},
        {"type": "term", "raw_name": "Beta", "translated_name": "Beta EN"},
    ]

    result = skip_duplicate_entries(entries)

    assert len(result) == 2
    summary = _pass1_summary(capsys.readouterr().out)
    assert "1 duplicates removed, 1 entry skipped due to missing raw_name (2 remaining)" in summary
