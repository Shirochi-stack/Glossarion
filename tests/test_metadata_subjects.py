import json
import threading
import zipfile

import pytest

from Chapter_Extractor import _extract_epub_metadata
from epub_metadata_utils import restore_truncated_repeatable_metadata
from metadata_batch_translator import (
    BatchHeaderTranslator,
    MetadataBatchTranslatorUI,
    MetadataTranslationCancelled,
    MetadataTranslator,
)
from gender_tracking import tracker_path_for_glossary
from metadata_translation_worker import _translate_title as _translate_worker_title
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


def _artifact_glossary_config(
    tmp_path, *, decision="auto", compress=True, legacy_pair=False
):
    output_root = tmp_path / "Output"
    output_dir = output_root / "book"
    glossary_dir = output_root / "Glossary" / "book"
    output_dir.mkdir(parents=True)
    glossary_dir.mkdir(parents=True)
    source_path = tmp_path / "book.epub"
    source_path.write_bytes(b"not-needed-by-request-tests")
    glossary_path = glossary_dir / "book_glossary.csv"
    glossary_text = (
        "type,raw_name,translated_name,gender,description\n"
        "character,루나,Luna,male,Hero\n"
        "term,마탑,Magic Tower,,Place\n"
    )
    if legacy_pair:
        glossary_text += "character,루나,Luna,female,Heroine\n"
    glossary_path.write_text(glossary_text, encoding="utf-8")
    tracker_path = tracker_path_for_glossary(str(glossary_path))
    with open(tracker_path, "w", encoding="utf-8") as tracker_file:
        json.dump(
            {
                "version": 2,
                "entries": {
                    "루나": {
                        "raw_name": "루나",
                        "translated_name": "Luna",
                        "decision": decision,
                        "occurrences": [
                            {
                                "gender": "male",
                                "chapter_num": 1,
                                "chapter_file": "chapter1.xhtml",
                            },
                            {
                                "gender": "female",
                                "chapter_num": 2,
                                "chapter_file": "chapter2.xhtml",
                            },
                        ],
                        "changes": [],
                    }
                },
            },
            tracker_file,
            ensure_ascii=False,
        )
    settings = {
        "APPEND_GLOSSARY": "1",
        "APPEND_GLOSSARY_PROMPT": "GLOSSARY_REFERENCE",
        "COMPRESS_GLOSSARY_PROMPT": "1" if compress else "0",
        "AUTO_GLOSSARY_MODE": "full",
        "OUTPUT_DIRECTORY": str(output_root),
        "GLOSSARY_SOURCE_PATH": str(source_path),
        "MANUAL_GLOSSARY": "",
        "EMERGENCY_GLOSSARY_COMPLIANCE": "1",
        "EMERGENCY_GLOSSARY_COMPLIANCE_MODE": "all",
        "GLOSSARY_SKIP_GENDER_TRACKING": "0",
        "GLOSSARY_GENDER_NOISE_THRESHOLD": "0",
        "GLOSSARY_GENDER_TRACKING_BIAS": "none",
        "OUTPUT_LANGUAGE": "English",
        "MODEL": "test-model",
    }
    return {
        "_prefer_explicit_config": True,
        "_glossary_settings": settings,
        "output_dir": str(output_dir),
        "source_path": str(source_path),
        "output_language": "English",
        "headers_per_batch": 1,
        "metadata_batch_prompt": "Translate metadata to {target_lang}; return JSON.",
        "metadata_field_prompts": {
            "description": "Translate to {target_lang}.",
        },
    }


@pytest.mark.parametrize("batch_enabled", ["0", "1"])
@pytest.mark.parametrize("translation_type", ["header", "toc"])
def test_header_and_toc_requests_use_compressed_tracked_glossary_in_all_batch_modes(
    tmp_path, monkeypatch, batch_enabled, translation_type
):
    config = _artifact_glossary_config(tmp_path)
    monkeypatch.setenv("BATCH_TRANSLATION", batch_enabled)
    monkeypatch.setenv("EXTRACTION_WORKERS", "2")

    class Client:
        model = "test-model"
        output_dir = config["output_dir"]

    translator = BatchHeaderTranslator(Client(), config)
    captured = {}
    capture_lock = threading.Lock()

    def capture_request(**kwargs):
        messages = kwargs["messages"]
        batch = json.loads(
            messages[-1]["content"].rsplit("Titles to translate:\n", 1)[1]
        )
        key = next(iter(batch))
        with capture_lock:
            captured[int(key)] = messages
        return json.dumps({key: f"Translated {key}"})

    translator._send_with_retry = capture_request
    translated = translator.translate_headers_batch(
        {1: "루나 1", 2: "루나 2"},
        batch_size=1,
        translation_type=translation_type,
    )

    assert translated == {1: "Translated 1", 2: "Translated 2"}
    assert set(captured) == {1, 2}
    assert "GLOSSARY_REFERENCE" in captured[1][0]["content"]
    assert "character,루나,Luna,male,Hero" in captured[1][0]["content"]
    assert "character,루나,Luna,female,Hero" in captured[2][0]["content"]
    assert "루나" not in captured[1][1]["content"]
    assert "Luna 1" in captured[1][1]["content"]


def test_header_request_appends_full_glossary_when_compression_is_disabled(
    tmp_path, monkeypatch
):
    config = _artifact_glossary_config(tmp_path, compress=False)
    monkeypatch.setenv("BATCH_TRANSLATION", "0")

    class Client:
        model = "test-model"
        output_dir = config["output_dir"]

    translator = BatchHeaderTranslator(Client(), config)
    captured = []

    def capture_request(**kwargs):
        captured.append(kwargs["messages"])
        return '{"1": "Translated"}'

    translator._send_with_retry = capture_request
    translator.translate_headers_batch({1: "루나"}, batch_size=1)

    system = captured[0][0]["content"]
    assert "GLOSSARY_REFERENCE" in system
    assert "마탑,Magic Tower" in system


@pytest.mark.parametrize("mode", ["together", "parallel"])
def test_metadata_requests_use_compliance_and_manual_gender_decision(
    tmp_path, mode
):
    config = _artifact_glossary_config(tmp_path, decision="female")

    class Client:
        client_type = "openai"
        output_dir = config["output_dir"]

    translator = MetadataTranslator(Client(), config)
    captured = []

    def capture_request(**kwargs):
        captured.append(kwargs["messages"])
        if mode == "together":
            return json.dumps({"description": "Translated description"})
        return "Translated description"

    translator._send_with_retry = capture_request
    result = translator.translate_metadata(
        {"description": "루나의 이야기"},
        {"description": True},
        mode=mode,
    )

    assert result["description"] == "Translated description"
    assert len(captured) == 1
    system, user = captured[0]
    assert "GLOSSARY_REFERENCE" in system["content"]
    assert "character,루나,Luna,female,Hero" in system["content"]
    assert "루나" not in user["content"]
    assert "Luna의 이야기" in user["content"]


@pytest.mark.parametrize("legacy_pair", [False, True])
def test_metadata_auto_gender_uses_the_stable_stored_winner(
    tmp_path, legacy_pair
):
    config = _artifact_glossary_config(
        tmp_path, decision="auto", legacy_pair=legacy_pair
    )

    class Client:
        client_type = "openai"
        output_dir = config["output_dir"]

    translator = MetadataTranslator(Client(), config)
    captured = []

    def capture_request(**kwargs):
        captured.append(kwargs["messages"])
        return json.dumps({"description": "Translated description"})

    translator._send_with_retry = capture_request
    translator.translate_metadata(
        {"description": "루나의 이야기"},
        {"description": True},
        mode="together",
    )

    system = captured[0][0]["content"]
    assert "character,루나,Luna,male,Hero" in system
    assert "character,루나,Luna,female,Hero" not in system


def test_metadata_worker_title_request_uses_same_glossary_pipeline(
    tmp_path, monkeypatch
):
    import TransateKRtoEN

    config = _artifact_glossary_config(tmp_path, decision="female")
    settings = dict(config["_glossary_settings"])
    settings.update({
        "BOOK_TITLE_SYSTEM_PROMPT": "Translate title to {target_lang}.",
        "BOOK_TITLE_PROMPT": "",
        "MAX_OUTPUT_TOKENS": "100",
        "TRANSLATION_TEMPERATURE": "0.1",
    })
    captured = []

    class Client:
        client_type = "openai"
        output_dir = config["output_dir"]

    def capture_send(**kwargs):
        captured.append(kwargs["messages"])
        return "Luna's Story"

    monkeypatch.setattr(TransateKRtoEN, "send_with_interrupt", capture_send)
    translated, succeeded = _translate_worker_title(
        "루나의 이야기", Client(), settings, lambda: False
    )

    assert succeeded is True
    assert translated == "Luna's Story"
    system, user = captured[0]
    assert "GLOSSARY_REFERENCE" in system["content"]
    assert "character,루나,Luna,female,Hero" in system["content"]
    assert "루나" not in user["content"]
    assert "Luna의 이야기" in user["content"]


def test_primary_book_title_request_uses_same_glossary_pipeline(
    tmp_path, monkeypatch
):
    import TransateKRtoEN

    config = _artifact_glossary_config(tmp_path, decision="female")
    settings = dict(config["_glossary_settings"])
    settings.update({
        "TRANSLATE_BOOK_TITLE": "1",
        "BOOK_TITLE_SYSTEM_PROMPT": "Translate title to {target_lang}.",
        "BOOK_TITLE_PROMPT": "",
        "MAX_OUTPUT_TOKENS": "100",
    })
    captured = []

    class Client:
        client_type = "openai"
        output_dir = config["output_dir"]

    def capture_send(**kwargs):
        captured.append(kwargs["messages"])
        return "Luna's Story"

    monkeypatch.setattr(TransateKRtoEN, "send_with_interrupt", capture_send)
    translated, succeeded = TransateKRtoEN.translate_title(
        "루나의 이야기",
        Client(),
        None,
        None,
        return_status=True,
        output_dir=config["output_dir"],
        source_path=config["source_path"],
        settings=settings,
    )

    assert succeeded is True
    assert translated == "Luna's Story"
    system, user = captured[0]
    assert "GLOSSARY_REFERENCE" in system["content"]
    assert "character,루나,Luna,female,Hero" in system["content"]
    assert "Luna의 이야기" in user["content"]
