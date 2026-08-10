import json
import zipfile

import pytest

from Chapter_Extractor import _extract_epub_metadata
from epub_metadata_utils import restore_truncated_repeatable_metadata
from metadata_batch_translator import (
    MetadataBatchTranslatorUI,
    MetadataTranslationCancelled,
    MetadataTranslator,
)
from unified_api_client import UnifiedClientError


SUBJECTS = [
    "현대",
    "아포칼립스",
    "TS",
    "천재",
    "마법사",
    "시스템",
    "성장",
    "갤러리",
    "커뮤니티",
    "노맨스",
]


def _opf_bytes():
    subject_xml = "\n".join(
        f"      <dc:subject>{subject}</dc:subject>" for subject in SUBJECTS
    )
    return f"""<?xml version="1.0" encoding="UTF-8"?>
    <package xmlns="http://www.idpf.org/2007/opf"
             xmlns:dc="http://purl.org/dc/elements/1.1/" version="2.0">
      <metadata>
        <dc:title>아포칼립스 속 방구석 마탑주</dc:title>
{subject_xml}
      </metadata>
    </package>""".encode("utf-8")


def _write_epub(path):
    with zipfile.ZipFile(path, "w") as epub:
        epub.writestr("OEBPS/content.opf", _opf_bytes())


def test_chapter_extractor_preserves_all_subjects_in_order(tmp_path):
    epub_path = tmp_path / "book.epub"
    _write_epub(epub_path)

    with zipfile.ZipFile(epub_path) as epub:
        metadata = _extract_epub_metadata(epub)

    assert metadata["subject"] == SUBJECTS


def test_metadata_configuration_reader_sees_all_subjects(tmp_path):
    epub_path = tmp_path / "book.epub"
    _write_epub(epub_path)
    ui = MetadataBatchTranslatorUI.__new__(MetadataBatchTranslatorUI)

    metadata = ui._detect_all_metadata_fields_for_epub(str(epub_path))

    assert metadata["subject"] == SUBJECTS


def test_cached_single_subject_is_restored_and_marked_for_retranslation():
    existing = {
        "subject": "Modern",
        "original_subject": "현대",
        "subject_translated": True,
    }

    restored = restore_truncated_repeatable_metadata(
        existing, {"subject": SUBJECTS}
    )

    assert restored == {"subject"}
    assert existing["subject"] == SUBJECTS
    assert existing["original_subject"] == SUBJECTS
    assert "subject_translated" not in existing


def test_complete_translated_subject_list_is_not_reset():
    translated_subjects = [
        "Modern",
        "Apocalypse",
        "TS",
        "Genius",
        "Mage",
        "System",
        "Growth",
        "Gallery",
        "Community",
        "No Romance",
    ]
    existing = {
        "subject": translated_subjects,
        "original_subject": SUBJECTS,
        "subject_translated": True,
    }

    restored = restore_truncated_repeatable_metadata(
        existing, {"subject": SUBJECTS}
    )

    assert restored == set()
    assert existing["subject"] == translated_subjects
    assert existing["subject_translated"] is True


def test_together_translation_preserves_subject_array_shape():
    translated_subjects = [
        "Modern",
        "Apocalypse",
        "TS",
        "Genius",
        "Mage",
        "System",
        "Growth",
        "Gallery",
        "Community",
        "No Romance",
    ]
    translator = MetadataTranslator(object(), {"output_language": "English"})
    translator._send_with_retry = lambda **kwargs: json.dumps(
        {"subject": translated_subjects}, ensure_ascii=False
    )

    result = translator.translate_metadata(
        {"subject": SUBJECTS}, {"subject": True}, mode="together"
    )

    assert result["subject"] == translated_subjects
    assert isinstance(result["subject"], list)


def test_parallel_translation_preserves_subject_array_shape():
    translated_subjects = [
        "Modern",
        "Apocalypse",
        "TS",
        "Genius",
        "Mage",
        "System",
        "Growth",
        "Gallery",
        "Community",
        "No Romance",
    ]
    translator = MetadataTranslator(object(), {"output_language": "English"})
    translator._send_with_retry = lambda **kwargs: json.dumps(
        translated_subjects, ensure_ascii=False
    )

    result = translator.translate_metadata(
        {"subject": SUBJECTS}, {"subject": True}, mode="parallel"
    )

    assert result["subject"] == translated_subjects


def test_english_pdf_title_is_really_sent_when_target_is_arabic(monkeypatch):
    monkeypatch.setenv("OUTPUT_LANGUAGE", "Arabic")
    translator = MetadataTranslator(
        object(),
        {
            "output_language": "Arabic",
            "metadata_batch_prompt": (
                "Translate the following metadata fields to {target_lang}. "
                "Return only JSON."
            ),
        },
    )
    calls = []

    def capture_request(**kwargs):
        calls.append(kwargs)
        return json.dumps({"title": "\u0645\u0627 \u0647\u0648 \u062f\u0641\u0627\u0639 \u0627\u0644\u0627\u0633\u062a\u0642\u0631\u0627\u0631 (1)"})

    translator._send_with_retry = capture_request
    source_title = "What is rest defence (1)"
    result = translator.translate_metadata(
        {"title": source_title}, {"title": True}, mode="together"
    )

    assert len(calls) == 1
    assert calls[0]["context"] == "metadata"
    assert source_title in calls[0]["messages"][1]["content"]
    assert result["title"] == "\u0645\u0627 \u0647\u0648 \u062f\u0641\u0627\u0639 \u0627\u0644\u0627\u0633\u062a\u0642\u0631\u0627\u0631 (1)"
    assert translator.last_completed_fields == {"title"}
    assert translator.last_requested_fields == {"title"}
    assert translator.last_no_request_fields == set()


def test_english_title_can_skip_only_when_target_is_english(monkeypatch):
    monkeypatch.setenv("OUTPUT_LANGUAGE", "English")
    translator = MetadataTranslator(object(), {"output_language": "English"})
    translator._send_with_retry = lambda **_kwargs: pytest.fail(
        "English target should not send an already-English title"
    )

    result = translator.translate_metadata(
        {"title": "What is rest defence (1)"},
        {"title": True},
        mode="together",
    )

    assert result["title"] == "What is rest defence (1)"
    assert translator.last_completed_fields == {"title"}
    assert translator.last_requested_fields == set()
    assert translator.last_no_request_fields == {"title"}


def test_incomplete_subject_array_response_is_rejected():
    translator = MetadataTranslator(object(), {"output_language": "English"})
    response = json.dumps({"subject": ["Modern"]})

    parsed = translator._parse_metadata_response(
        response, {"subject": SUBJECTS}
    )

    assert parsed == {}


def test_metadata_send_normalizes_user_cancellation_without_exception_chain(monkeypatch):
    import TransateKRtoEN

    def cancelled_send(**_kwargs):
        raise UnifiedClientError("Translation stopped by user", error_type="cancelled")

    monkeypatch.setattr(TransateKRtoEN, "send_with_interrupt", cancelled_send)
    translator = MetadataTranslator(object(), {"output_language": "English"})

    with pytest.raises(MetadataTranslationCancelled) as exc_info:
        translator._send_with_retry([], 0.3, 100, context="metadata")

    assert exc_info.value.__cause__ is None
    assert exc_info.value.__suppress_context__ is True
