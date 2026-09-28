"""Glossary QA failures can be retried without repeating exhausted API errors."""

import json

from glossary_translation_gate import retryable_glossary_qa_failures


def test_retryable_glossary_qa_failures_uses_saved_progress(tmp_path):
    progress_path = tmp_path / "book_glossary_progress.json"
    progress_path.write_text(json.dumps({
        "failed": [0, 1, 2],
        "qa_issues_found": {
            "0": ["TRUNCATED"],
            "1": ["SPLIT_FAILED"],
            "2": ["API_ERROR"],
        },
        "chapters": {
            "3": {"chapter_index": 3, "status": "qa_failed", "qa_issues_found": ["EMPTY_OUTPUT"]},
            "4": {"chapter_index": 4, "status": "completed"},
        },
    }), encoding="utf-8")

    assert retryable_glossary_qa_failures(str(progress_path)) == {0, 1, 3}


def test_missing_or_unreadable_progress_does_not_retry(tmp_path):
    progress_path = tmp_path / "book_glossary_progress.json"
    assert retryable_glossary_qa_failures(str(progress_path)) == set()
    progress_path.write_text("{broken", encoding="utf-8")
    assert retryable_glossary_qa_failures(str(progress_path)) == set()
