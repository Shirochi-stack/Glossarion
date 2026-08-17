import base64
import json
import os
import re
import subprocess
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from lxml import etree
from bs4 import BeautifulSoup
from PySide6.QtWidgets import QFrame, QLabel, QListWidgetItem, QPlainTextEdit

from sdlxliff_converter import convert_sdlxliff
from sdlxliff_extractor import extract_sdlxliff_to_chapters
from sdlxliff_sidecar_writer import (
    _is_manual_editing_sdlxliff,
    _is_manual_untranslated_sdlxliff,
    _write_html_sdlxliff_sidecar as _shared_write_html_sdlxliff_sidecar,
)
from TransateKRtoEN import (
    _original_markup_for_copy,
    _refinement_raw_source_message,
    _write_html_sdlxliff_sidecar,
    should_skip_configured_special_file_for_translation,
)
from Retranslation_GUI import RetranslationMixin, SDLXLIFFReviewDialog, _sdlxliff_machine_translation_path
from qa_scan_runtime import default_qa_scan_settings
from Chapter_Extractor import prepare_epub_image_assets
from scan_html_folder import (
    _count_beautifulsoup_review_tags,
    _extract_paragraphs,
    _missing_ending_quotation_paragraphs,
    _missing_beautifulsoup_tags_issue,
    _sdlxliff_review_tag_counts,
    check_html_structure_issues,
    process_html_file_batch,
)


def test_sdlxliff_image_asset_preparation_runs_without_chapter_extraction(tmp_path):
    epub_path = tmp_path / "source.epub"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    source_html = (
        '<html><body><p><img src="../Images/1.png" alt="one"></p></body></html>'
    )
    with zipfile.ZipFile(epub_path, "w") as source_zip:
        source_zip.writestr(
            "OEBPS/Text/chapter_notice0002.xhtml", source_html
        )
        source_zip.writestr(
            "OEBPS/Images/1.png",
            base64.b64decode(
                "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
            ),
        )
    output_html = workspace / "response_chapter_notice0002.html"
    output_html.write_text(source_html, encoding="utf-8")

    progress_messages = []
    result = prepare_epub_image_assets(
        str(epub_path),
        str(workspace),
        progress_callback=progress_messages.append,
    )

    assert result["ready"] is True
    assert result["prepared"] is True
    assert result["source_images"] == 1
    assert result["extracted"] == 1
    rename_map = json.loads(
        (workspace / "image_rename_map.json").read_text(encoding="utf-8")
    )
    assert rename_map == {"1.png": "chapter_notice0002_img_1.png"}
    assert (workspace / "images" / "chapter_notice0002_img_1.png").is_file()
    assert "../Images/chapter_notice0002_img_1.png" in output_html.read_text(
        encoding="utf-8"
    )
    assert not (workspace / "metadata.json").exists()
    assert not (workspace / "chapters_info.json").exists()
    assert progress_messages[0] == "🖼️ Preparing EPUB image assets for SDLXLIFF review..."
    assert any(message.startswith("📥 Extracted 1 of 1") for message in progress_messages)
    assert any("Applying chapter image rename map" in message for message in progress_messages)
    assert progress_messages[-1] == "✅ Prepared 1 EPUB image asset(s) for SDLXLIFF review"

    second_result = prepare_epub_image_assets(str(epub_path), str(workspace))
    assert second_result["ready"] is True
    assert second_result["prepared"] is False


def test_sdlxliff_image_asset_status_is_forwarded_to_translator_log(tmp_path, monkeypatch):
    epub_path = tmp_path / "source.epub"
    epub_path.write_bytes(b"epub")
    logged = []

    def fake_prepare(_epub_path, _output_dir, progress_callback=None):
        progress_callback("🖼️ Preparing EPUB image assets for SDLXLIFF review...")
        progress_callback("✅ Prepared 3 EPUB image asset(s) for SDLXLIFF review")
        return {"ready": True, "prepared": True, "renamed": 3}

    monkeypatch.setattr("Chapter_Extractor.prepare_epub_image_assets", fake_prepare)
    dialog = SDLXLIFFReviewDialog.__new__(SDLXLIFFReviewDialog)
    dialog.output_dir = str(tmp_path)
    dialog._context_parent = SimpleNamespace(
        append_log=lambda message, source_thread=None: logged.append(message)
    )
    dialog._review_source_epub_for_image_assets = lambda: str(epub_path)

    result = dialog._ensure_review_image_assets()

    assert result["ready"] is True
    assert logged == [
        "🖼️ Preparing EPUB image assets for SDLXLIFF review...",
        "✅ Prepared 3 EPUB image asset(s) for SDLXLIFF review",
    ]


def test_html_sdlxliff_writer_is_shared_between_translation_and_review_paths():
    transate_source = (SRC / "TransateKRtoEN.py").read_text(encoding="utf-8")
    review_source = (SRC / "Retranslation_GUI.py").read_text(encoding="utf-8")

    assert _write_html_sdlxliff_sidecar is _shared_write_html_sdlxliff_sidecar
    assert "from sdlxliff_sidecar_writer import" in transate_source
    assert "from sdlxliff_sidecar_writer import _write_html_sdlxliff_sidecar" in review_source
    generator_body = review_source[
        review_source.index("def _generate_sdlxliff_sidecars_from_completed_entries"):
        review_source.index("def _open_or_reuse_sdlxliff_review", review_source.index("def _generate_sdlxliff_sidecars_from_completed_entries"))
    ]
    assert "from TransateKRtoEN import _write_html_sdlxliff_sidecar" not in generator_body


def test_html_sdlxliff_writer_is_packaged_in_app_specs():
    for spec_name in (
        "translator.spec",
        "translator_Heavy.spec",
        "translator_lite.spec",
        "translator_lite_linux.spec",
        "translator_lite_mac.spec",
        "translator_lite_mac_intel.spec",
        "translator_lite_mac_intel_NoCuda.spec",
        "translator_lite_mac_NoCuda.spec",
        "translator_NoCuda.spec",
        "translator_TurboLite.spec",
    ):
        spec_source = (SRC / spec_name).read_text(encoding="utf-8")
        assert "('sdlxliff_sidecar_writer.py', '.')" in spec_source
        assert "'sdlxliff_sidecar_writer'" in spec_source


def test_numbered_special_html_skip_predicate_keeps_refinement_scope(monkeypatch):
    monkeypatch.setenv("TRANSLATE_SPECIAL_FILES", "0")
    monkeypatch.setenv("TRANSLATE_ALL_NUMBERED_HTML", "1")
    monkeypatch.setenv("SPECIAL_FILE_KEYWORDS", "notice, info")

    assert should_skip_configured_special_file_for_translation("chapter_notice0004.xhtml") is False
    assert should_skip_configured_special_file_for_translation("response_chapter_notice0004.html") is False
    assert should_skip_configured_special_file_for_translation("info.xhtml") is True
    assert should_skip_configured_special_file_for_translation("chapter0004.xhtml") is False

    assert should_skip_configured_special_file_for_translation(
        "chapter_notice0004.xhtml",
        translate_all_numbered=False,
    ) is True
    assert should_skip_configured_special_file_for_translation(
        "chapter_notice0004.xhtml",
        translate_special=True,
    ) is False


def test_multipass_refinement_filter_uses_numbered_special_skip_predicate():
    transate_source = (SRC / "TransateKRtoEN.py").read_text(encoding="utf-8")
    block_start = transate_source.index("multipass_chapters = []")
    block_end = transate_source.index("_process_refinement_or_tts_mode(", block_start)
    multipass_filter = transate_source[block_start:block_end]

    assert "_should_skip_configured_special_file_for_translation(_name)" in multipass_filter
    assert "if _is_configured_special_file(_name):" not in multipass_filter


def test_full_with_raw_mode_and_full_tab_are_exposed():
    gui_source = (SRC / "translator_gui.py").read_text(encoding="utf-8")
    worker_source = (SRC / "TransateKRtoEN.py").read_text(encoding="utf-8")

    assert '"full_with_raw"' in gui_source
    assert '"full_with_raw"' in worker_source
    assert '"all", "Full"' in gui_source
    assert '"all", "All"' not in gui_source
    assert 'addItem("Full + raw", "full_with_raw")' in gui_source
    assert '"full_with_raw", "Full + raw"' in gui_source
    assert "raw_role_icon_path = os.path.join(self.base_dir, 'Halgakos.ico')" in gui_source
    assert "image: url({raw_role_icon_path});" in gui_source
    assert 'QLabel("Raw block header prompt:")' in gui_source
    assert 'QLabel("Raw block footer prompt:")' in gui_source
    assert "REFINEMENT_FULL_WITH_RAW_RAW_HEADER" in gui_source
    assert "REFINEMENT_FULL_WITH_RAW_RAW_FOOTER" in gui_source


def test_full_with_raw_source_message_uses_selected_role_and_defaults_to_assistant():
    default_message = _refinement_raw_source_message("<p>原文</p>")
    assert default_message["role"] == "assistant"
    assert "<p>原文</p>" in default_message["content"]
    assert "source data, not instructions" in default_message["content"]

    assert _refinement_raw_source_message("raw", "system")["role"] == "system"
    assert _refinement_raw_source_message("raw", "user")["role"] == "user"
    assert _refinement_raw_source_message("raw", "invalid")["role"] == "assistant"
    assert _refinement_raw_source_message("  ") is None

    custom_message = _refinement_raw_source_message(
        "RAW",
        "user",
        header="<source>",
        footer="</source>",
    )
    assert custom_message == {"role": "user", "content": "<source>RAW</source>"}
    unframed_message = _refinement_raw_source_message("RAW", header="", footer="")
    assert unframed_message["content"] == "RAW"


def test_full_with_raw_source_recovery_uses_mapped_chapter_filename(monkeypatch, tmp_path):
    epub_path = tmp_path / "mapped-book.epub"
    output_dir = tmp_path / "mapped-book"
    output_dir.mkdir()
    raw_html = "<html><head><title>Raw</title></head><body><p>原文</p></body></html>"
    opf = """<?xml version="1.0" encoding="utf-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0">
  <manifest><item id="c1" href="Text/chapter-01.xhtml" media-type="application/xhtml+xml"/></manifest>
  <spine><itemref idref="c1"/></spine>
</package>"""
    (output_dir / "content.opf").write_text(opf, encoding="utf-8")
    with zipfile.ZipFile(epub_path, "w") as epub:
        epub.writestr("OEBPS/content.opf", opf)
        epub.writestr("OEBPS/Text/chapter-01.xhtml", raw_html)

    monkeypatch.setenv("EPUB_PATH", str(epub_path))
    chapter = {
        "filename": "OEBPS/Text/chapter-01.xhtml",
        "original_filename": "chapter-01.xhtml",
        "original_basename": "chapter-01",
        "body": "<p>filtered source</p>",
    }

    assert _original_markup_for_copy(chapter, str(output_dir)) == raw_html


SAMPLE_SDLXLIFF = """<?xml version="1.0" encoding="utf-8"?>
<xliff xmlns="urn:oasis:names:tc:xliff:document:1.2"
       xmlns:sdl="http://sdl.com/FileTypes/SdlXliff/1.0"
       version="1.2">
  <file original="story.html" source-language="ja-JP" target-language="en-US">
    <body>
      <trans-unit id="u1">
        <source>Alpha <x id="1"/> beta</source>
        <seg-source><mrk mtype="seg" mid="1">Alpha <x id="1"/> beta</mrk></seg-source>
        <target><mrk mtype="seg" mid="1"></mrk></target>
        <sdl:seg-defs><sdl:seg id="1" conf="Draft"/></sdl:seg-defs>
      </trans-unit>
      <trans-unit id="u2" translate="no">
        <source>Do not translate</source>
        <seg-source><mrk mtype="seg" mid="1">Do not translate</mrk></seg-source>
        <target><mrk mtype="seg" mid="1">Existing no-translate target</mrk></target>
      </trans-unit>
      <trans-unit id="u3">
        <source>Locked segment</source>
        <seg-source><mrk mtype="seg" mid="1">Locked segment</mrk></seg-source>
        <target><mrk mtype="seg" mid="1"></mrk></target>
        <sdl:seg-defs><sdl:seg id="1" conf="Draft" locked="true"/></sdl:seg-defs>
      </trans-unit>
      <trans-unit id="u4">
        <source>Approved segment</source>
        <seg-source><mrk mtype="seg" mid="1">Approved segment</mrk></seg-source>
        <target><mrk mtype="seg" mid="1">Already approved</mrk></target>
        <sdl:seg-defs><sdl:seg id="1" conf="ApprovedTranslation"/></sdl:seg-defs>
      </trans-unit>
      <trans-unit id="u5">
        <source>No segmentation fallback</source>
        <target></target>
      </trans-unit>
      <trans-unit id="u6">
        <source>Missing target fallback</source>
      </trans-unit>
    </body>
  </file>
</xliff>
"""


def _write_sample(tmp_path, content=SAMPLE_SDLXLIFF):
    source = tmp_path / "sample.sdlxliff"
    source.write_text(content, encoding="utf-8")
    return source


def _visible_text(elem):
    return "".join(elem.itertext())


def _records_from_first_batch(out):
    chapters = json.loads((out / "chapters_full.json").read_text(encoding="utf-8"))
    return chapters, json.loads(chapters[0]["body"])


def _write_batch_response(out, records):
    (out / "response_section_1.txt").write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def test_sdlxliff_extraction_filters_and_protects_inline_tags(tmp_path, monkeypatch):
    monkeypatch.setenv("EXTRACTION_WORKERS", "4")
    monkeypatch.setenv("SDLXLIFF_AVAILABLE_TOKENS", "100000")
    source = _write_sample(tmp_path)
    out = tmp_path / "out"

    result = extract_sdlxliff_to_chapters(str(source), str(out))

    chapters = json.loads((out / "chapters_full.json").read_text(encoding="utf-8"))
    records = json.loads(chapters[0]["body"])
    manifest = json.loads((out / "sdlxliff_manifest.json").read_text(encoding="utf-8"))
    metadata = json.loads((out / "metadata.json").read_text(encoding="utf-8"))

    assert result["success"] is True
    assert len(chapters) == 1
    assert manifest["segment_count"] == 3
    assert manifest["batch_count"] == 1
    assert manifest["type"] == "sdlxliff_json_batches"
    assert metadata["type"] == "sdlxliff"
    assert records[0] == {"id": "1", "source": "Alpha [[XLIFF_TAG_000001_0000]] beta"}
    assert [record["id"] for record in records] == ["1", "2", "3"]
    assert chapters[0]["sdlxliff_batch"] is True
    assert chapters[0]["sdlxliff_placeholder_only"] is False
    tag_info = manifest["segments"][0]["tag_map"]["[[XLIFF_TAG_000001_0000]]"]
    assert tag_info["kind"] == "empty"
    assert etree.QName(tag_info["tag"]).localname == "x"
    assert tag_info["attrib"]["id"] == "1"
    assert [segment["unit_id"] for segment in manifest["segments"]] == ["u1", "u5", "u6"]


def test_sdlxliff_placeholder_only_segment_is_marked_and_round_trips(tmp_path):
    source = _write_sample(
        tmp_path,
        """<?xml version="1.0" encoding="utf-8"?>
<xliff xmlns="urn:oasis:names:tc:xliff:document:1.2"
       xmlns:sdl="http://sdl.com/FileTypes/SdlXliff/1.0"
       version="1.2">
  <file original="story.html" source-language="ja-JP" target-language="en-US">
    <body>
      <trans-unit id="tag-only">
        <source><x id="1"/></source>
        <seg-source><mrk mtype="seg" mid="1"><x id="1"/></mrk></seg-source>
        <target><mrk mtype="seg" mid="1"></mrk></target>
        <sdl:seg-defs><sdl:seg id="1" conf="Draft"/></sdl:seg-defs>
      </trans-unit>
    </body>
  </file>
</xliff>
""",
    )
    out = tmp_path / "out"

    extract_sdlxliff_to_chapters(str(source), str(out))
    chapters = json.loads((out / "chapters_full.json").read_text(encoding="utf-8"))
    manifest = json.loads((out / "sdlxliff_manifest.json").read_text(encoding="utf-8"))

    assert chapters == []
    assert manifest["segment_count"] == 1
    assert manifest["translatable_segment_count"] == 0
    assert manifest["auto_insert_segment_count"] == 1
    assert manifest["segments"][0]["auto_insert"] is True
    assert manifest["segments"][0]["auto_target_text"] == "[[XLIFF_TAG_000001_0000]]"

    result = convert_sdlxliff(str(out))

    assert result["updated"] == 1
    tree = etree.parse(result["output_path"])
    target = tree.xpath("//*[local-name()='trans-unit'][@id='tag-only']/*[local-name()='target']/*[local-name()='mrk']")[0]
    assert _visible_text(target) == ""
    assert etree.QName(target[0]).localname == "x"
    assert target[0].get("id") == "1"


def test_sdlxliff_converter_updates_only_eligible_targets(tmp_path):
    source = _write_sample(tmp_path)
    out = tmp_path / "out"
    extract_sdlxliff_to_chapters(str(source), str(out))

    _, records = _records_from_first_batch(out)
    placeholder = re.search(r"\[\[XLIFF_TAG_\d{6}_\d{4}\]\]", records[0]["source"]).group(0)
    _write_batch_response(
        out,
        [
            {"id": records[0]["id"], "target": f"First {placeholder} target"},
            {"id": records[1]["id"], "target": "Fallback target"},
            {"id": records[2]["id"], "target": "Created target"},
        ],
    )

    result = convert_sdlxliff(str(out))

    assert result["updated"] == 3
    assert result["skipped"] == 0
    assert result["missing"] == 0

    tree = etree.parse(result["output_path"])
    units = {unit.get("id"): unit for unit in tree.xpath("//*[local-name()='trans-unit']")}
    u1_target = units["u1"].xpath("./*[local-name()='target']//*[local-name()='mrk'][@mid='1']")[0]
    assert _visible_text(u1_target) == "First  target"
    assert etree.QName(u1_target[0]).localname == "x"
    assert u1_target[0].tail == " target"
    assert _visible_text(units["u2"].xpath("./*[local-name()='target']")[0]) == "Existing no-translate target"
    assert _visible_text(units["u3"].xpath("./*[local-name()='target']")[0]) == ""
    assert _visible_text(units["u4"].xpath("./*[local-name()='target']")[0]) == "Already approved"
    assert _visible_text(units["u5"].xpath("./*[local-name()='target']")[0]) == "Fallback target"
    assert _visible_text(units["u6"].xpath("./*[local-name()='target']")[0]) == "Created target"


def test_sdlxliff_converter_skips_placeholder_mismatch(tmp_path):
    source = _write_sample(tmp_path)
    out = tmp_path / "out"
    extract_sdlxliff_to_chapters(str(source), str(out))

    _, records = _records_from_first_batch(out)
    _write_batch_response(
        out,
        [
            {"id": records[0]["id"], "target": "Missing protected placeholder"},
            {"id": records[1]["id"], "target": "Fallback target"},
            {"id": records[2]["id"], "target": "Created target"},
        ],
    )

    result = convert_sdlxliff(str(out))

    assert result["updated"] == 2
    assert result["skipped"] == 1
    assert result["missing"] == 0
    tree = etree.parse(result["output_path"])
    u1_target = tree.xpath("//*[local-name()='trans-unit'][@id='u1']/*[local-name()='target']")[0]
    assert _visible_text(u1_target) == ""


def test_sdlxliff_converter_updates_target_language_from_output_language(tmp_path, monkeypatch):
    monkeypatch.setenv("SDLXLIFF_AVAILABLE_TOKENS", "100000")
    monkeypatch.setenv("OUTPUT_LANGUAGE", "Japanese")
    source = _write_sample(
        tmp_path,
        SAMPLE_SDLXLIFF.replace(
            'source-language="ja-JP" target-language="en-US"',
            'source-language="en-US" target-language="de-DE"',
        ),
    )
    out = tmp_path / "out"
    extract_sdlxliff_to_chapters(str(source), str(out))

    _, records = _records_from_first_batch(out)
    placeholder = re.search(r"\[\[XLIFF_TAG_\d{6}_\d{4}\]\]", records[0]["source"]).group(0)
    _write_batch_response(
        out,
        [
            {"id": records[0]["id"], "target": f"最初 {placeholder} 対象"},
            {"id": records[1]["id"], "target": "フォールバック対象"},
            {"id": records[2]["id"], "target": "作成された対象"},
        ],
    )

    manifest = json.loads((out / "sdlxliff_manifest.json").read_text(encoding="utf-8"))
    assert manifest["target_language"] == "Japanese"
    assert manifest["target_language_code"] == "ja-JP"

    result = convert_sdlxliff(str(out))

    assert result["target_language_code"] == "ja-JP"
    tree = etree.parse(result["output_path"])
    file_elem = tree.xpath("//*[local-name()='file']")[0]
    assert file_elem.get("source-language") == "en-US"
    assert file_elem.get("target-language") == "ja-JP"


def test_sdlxliff_translated_existing_target_is_retranslated(tmp_path, monkeypatch):
    monkeypatch.setenv("SDLXLIFF_AVAILABLE_TOKENS", "100000")
    source = _write_sample(
        tmp_path,
        """<?xml version="1.0" encoding="utf-8"?>
<xliff xmlns="urn:oasis:names:tc:xliff:document:1.2"
       xmlns:sdl="http://sdl.com/FileTypes/SdlXliff/1.0"
       version="1.2">
  <file original="story.html" source-language="en-US" target-language="de-DE">
    <body>
      <trans-unit id="translated-stale">
        <source>Getting Started</source>
        <seg-source><mrk mtype="seg" mid="1">Getting Started</mrk></seg-source>
        <target><mrk mtype="seg" mid="1">Erste Schritte</mrk></target>
        <sdl:seg-defs><sdl:seg id="1" conf="Translated"/></sdl:seg-defs>
      </trans-unit>
    </body>
  </file>
</xliff>
""",
    )
    out = tmp_path / "out"

    extract_sdlxliff_to_chapters(str(source), str(out))
    chapters, records = _records_from_first_batch(out)

    assert len(chapters) == 1
    assert records == [{"id": "1", "source": "Getting Started"}]

    _write_batch_response(out, [{"id": "1", "target": "はじめに"}])
    result = convert_sdlxliff(str(out))

    assert result["updated"] == 1
    tree = etree.parse(result["output_path"])
    target = tree.xpath("//*[local-name()='trans-unit'][@id='translated-stale']/*[local-name()='target']/*[local-name()='mrk']")[0]
    assert _visible_text(target) == "はじめに"


def test_sdlxliff_worker_smoke_writes_manifest_and_chapters(tmp_path):
    source = _write_sample(tmp_path)
    out = tmp_path / "worker_out"
    worker = SRC / "sdlxliff_extraction_worker.py"

    completed = subprocess.run(
        [sys.executable, str(worker), str(source), str(out)],
        text=True,
        encoding="utf-8",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=20,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "[RESULT]" in completed.stdout
    assert (out / "chapters_full.json").exists()
    assert (out / "metadata.json").exists()
    assert (out / "sdlxliff_manifest.json").exists()


def test_html2text_output_writes_sdlxliff_sidecar(tmp_path, monkeypatch):
    monkeypatch.setenv("OUTPUT_SDLXLIFF", "1")
    monkeypatch.setenv("SOURCE_LANGUAGE", "Japanese")
    monkeypatch.setenv("OUTPUT_LANGUAGE", "English")
    source_html = "<html><body><h1>Source title</h1><p>Source body</p></body></html>"
    target_html = "<html><body><h1>Target title</h1><p>Target body</p></body></html>"
    chapter = {
        "enhanced_extraction": True,
        "original_html": source_html,
        "original_filename": "chapter001.xhtml",
    }

    sidecar_path = _write_html_sdlxliff_sidecar(
        str(tmp_path),
        "response_chapter001.html",
        chapter,
        "",
        target_html,
    )

    assert sidecar_path == str(tmp_path / "SDLXLIFF" / "response_chapter001.html.sdlxliff")
    tree = etree.parse(sidecar_path)
    file_elem = tree.xpath("//*[local-name()='file']")[0]
    source_elem = tree.xpath("//*[local-name()='source']")[0]
    target_elem = tree.xpath("//*[local-name()='target']")[0]
    assert file_elem.get("original") == "chapter001.xhtml"
    assert file_elem.get("source-language") == "ja-JP"
    assert file_elem.get("target-language") == "en-US"
    assert _visible_text(source_elem) == source_html
    assert _visible_text(target_elem) == target_html


def test_shared_sdlxliff_writer_records_live_translation_freshness(tmp_path, monkeypatch):
    monkeypatch.setenv("OUTPUT_SDLXLIFF", "1")
    output_name = "response_chapter001.html"
    output_path = tmp_path / output_name
    source_html = "<html><body><p>Source</p></body></html>"
    target_html = "<html><body><p>Target</p></body></html>"
    output_path.write_text(target_html, encoding="utf-8")

    sidecar_path = _shared_write_html_sdlxliff_sidecar(
        str(tmp_path),
        output_name,
        {"original_filename": "chapter001.xhtml"},
        source_html,
        target_html,
    )
    manifest_path = tmp_path / "SDLXLIFF" / "sdlxliff_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = manifest["entries"]["chapter001"]

    assert Path(sidecar_path).is_file()
    assert manifest["type"] == "html_sdlxliff_sidecar_freshness"
    assert len(record["output_sha256"]) == 64
    assert len(record["source_sha256"]) == 64

    sidecar = Path(sidecar_path)
    newer_ns = max(sidecar.stat().st_mtime_ns, output_path.stat().st_mtime_ns) + 5_000_000_000
    os.utime(output_path, ns=(newer_ns, newer_ns))
    mixin = RetranslationMixin.__new__(RetranslationMixin)
    assert mixin._sdlxliff_sidecar_current_for_output(
        str(sidecar),
        str(output_path),
    ) is True


def test_html_sdlxliff_sidecar_respects_output_toggle(tmp_path, monkeypatch):
    monkeypatch.setenv("OUTPUT_SDLXLIFF", "0")

    sidecar_path = _write_html_sdlxliff_sidecar(
        str(tmp_path),
        "response_chapter001.html",
        {"enhanced_extraction": True, "original_html": "<p>Source</p>"},
        "<p>Fallback</p>",
        "<p>Target</p>",
    )

    assert sidecar_path is None
    assert not (tmp_path / "SDLXLIFF").exists()


def test_sdlxliff_prompt_profile_is_bootstrapped_and_mirrored():
    gui_source = (SRC / "translator_gui.py").read_text(encoding="utf-8")
    app_source = (SRC / "app.py").read_text(encoding="utf-8")
    discord_source = (SRC / "discord_bot.py").read_text(encoding="utf-8")

    assert '"SDLXLIFF Editing v2"' in gui_source
    assert '"SDLXLIFF Editing":' not in gui_source
    assert re.search(r"protected = \{[\s\S]*?SDLXLIFF Editing v2[\s\S]*?\}", gui_source)
    assert re.search(r"always_include_profiles = \[[\s\S]*?SDLXLIFF Editing v2[\s\S]*?\]", gui_source)
    assert 'prompt_profiles["SDLXLIFF Editing"]' not in gui_source
    assert "You are editing SDLXLIFF JSON batch records" in gui_source
    assert "No markdown fences" in gui_source
    assert '"SDLXLIFF Editing v2"' in app_source
    assert '"SDLXLIFF Editing":' not in app_source
    assert 'profiles["SDLXLIFF Editing"]' not in app_source
    assert "You are editing SDLXLIFF JSON batch records" in app_source
    assert "No markdown fences" in app_source
    assert '"SDLXLIFF Editing v2"' in discord_source
    assert '"SDLXLIFF Editing":' not in discord_source
    assert "You are editing SDLXLIFF JSON batch records" in discord_source
    assert "No markdown fences" in discord_source


def test_sdlxliff_and_empty_attribute_settings_are_single_global_toggles():
    settings_source = (SRC / "other_settings.py").read_text(encoding="utf-8")
    gui_source = (SRC / "translator_gui.py").read_text(encoding="utf-8")

    assert "Fix Empty Attribute Tags (BeautifulSoup) - LLM Token Fix" not in settings_source
    assert settings_source.count("Fix Empty Attribute Tags (Extraction) - LLM Token Fix") == 1
    assert settings_source.index("Fix Empty Attribute Tags (EPUB) - LLM Token Fix") < settings_source.index("Fix Empty Attribute Tags (Extraction) - LLM Token Fix")
    assert settings_source.index("Number Spacing Tokenization Fix") < settings_source.index("Output SDLXLIFF")
    assert settings_source.index("Output SDLXLIFF") < settings_source.index("Skip Thinking for Lightweight Tasks")
    assert "fix_empty_attr_tags_bs_var = self.fix_empty_attr_tags_extract_var" in gui_source


def test_translation_chunk_prompt_parts_keep_chunk_html_in_user_message():
    from TransateKRtoEN import _build_translation_chunk_prompt_parts

    cfg = SimpleNamespace(
        ENABLE_TRANSLATION_CHUNK_PROMPT=False,
        INCLUDE_PREVIOUS_CHUNK=False,
        PREVIOUS_CHUNK_CONTEXT_LIMIT=3,
        TRANSLATION_CHUNK_PROMPT_ROLE="assistant",
        TRANSLATION_CHUNK_PROMPT="Part {chunk_idx}/{total_chunks} {chunk_html}",
    )
    system, prompt_msgs, user_prompt = _build_translation_chunk_prompt_parts(
        "system base",
        "<p>Chunk HTML</p>",
        2,
        5,
        cfg,
    )
    assert system == "system base"
    assert prompt_msgs == []
    assert user_prompt == "<p>Chunk HTML</p>"

    cfg.ENABLE_TRANSLATION_CHUNK_PROMPT = True
    system, prompt_msgs, user_prompt = _build_translation_chunk_prompt_parts(
        "system base",
        "<p>Chunk HTML</p>",
        2,
        5,
        cfg,
    )
    assert system == "system base"
    assert prompt_msgs == [{"role": "assistant", "content": "Part 2/5"}]
    assert user_prompt == "<p>Chunk HTML</p>"

    cfg.TRANSLATION_CHUNK_PROMPT_ROLE = "system"
    cfg.TRANSLATION_CHUNK_PROMPT = "Part {chunk_idx}/{total_chunks}"
    system, prompt_msgs, user_prompt = _build_translation_chunk_prompt_parts(
        "system base",
        "<p>Chunk HTML</p>",
        2,
        5,
        cfg,
    )
    assert system == "system base\n\nPart 2/5"
    assert prompt_msgs == []
    assert user_prompt == "<p>Chunk HTML</p>"

    cfg.TRANSLATION_CHUNK_PROMPT_ROLE = "user"
    system, prompt_msgs, user_prompt = _build_translation_chunk_prompt_parts(
        "system base",
        "<p>Chunk HTML</p>",
        2,
        5,
        cfg,
    )
    assert system == "system base"
    assert prompt_msgs == []
    assert user_prompt == "Part 2/5\n<p>Chunk HTML</p>"


def test_translation_chunk_prompt_can_include_previous_chunk_memory():
    from TransateKRtoEN import _build_translation_chunk_prompt_parts

    cfg = SimpleNamespace(
        ENABLE_TRANSLATION_CHUNK_PROMPT=False,
        INCLUDE_PREVIOUS_CHUNK=True,
        PREVIOUS_CHUNK_CONTEXT_LIMIT=3,
        TRANSLATION_CHUNK_PROMPT_ROLE="assistant",
        TRANSLATION_CHUNK_PROMPT="Part {chunk_idx}/{total_chunks}",
    )
    previous = "<p>One</p><p>Two</p><p>Three</p><p>Four</p>"
    system, prompt_msgs, user_prompt = _build_translation_chunk_prompt_parts(
        "system base",
        "<p>Current</p>",
        2,
        5,
        cfg,
        previous_chunk_html=previous,
    )
    assert system == "system base"
    assert user_prompt == "<p>Current</p>"
    assert len(prompt_msgs) == 1
    assert prompt_msgs[0]["role"] == "assistant"
    assert "[MEMORY - PREVIOUS CHUNK CONTEXT]" in prompt_msgs[0]["content"]
    assert "<p>One</p>" not in prompt_msgs[0]["content"]
    assert "<p>Two</p>" in prompt_msgs[0]["content"]
    assert "<p>Three</p>" in prompt_msgs[0]["content"]
    assert "<p>Four</p>" in prompt_msgs[0]["content"]

    cfg.TRANSLATION_CHUNK_PROMPT_ROLE = "user"
    cfg.ENABLE_TRANSLATION_CHUNK_PROMPT = True
    system, prompt_msgs, user_prompt = _build_translation_chunk_prompt_parts(
        "system base",
        "<p>Current</p>",
        2,
        5,
        cfg,
        previous_chunk_html=previous,
    )
    assert system == "system base"
    assert prompt_msgs == []
    assert user_prompt.startswith("[MEMORY - PREVIOUS CHUNK CONTEXT]")
    assert "[END MEMORY - PREVIOUS CHUNK CONTEXT]\nPart 2/5\n<p>Current</p>" in user_prompt

    cfg.PREVIOUS_CHUNK_CONTEXT_LIMIT = -1
    cfg.ENABLE_TRANSLATION_CHUNK_PROMPT = False
    system, prompt_msgs, user_prompt = _build_translation_chunk_prompt_parts(
        "system base",
        "<p>Current</p>",
        2,
        5,
        cfg,
        previous_chunk_html=previous,
    )
    assert previous in user_prompt

    cfg.PREVIOUS_CHUNK_CONTEXT_LIMIT = 2
    plain_previous = "line one\nline two\nline three"
    system, prompt_msgs, user_prompt = _build_translation_chunk_prompt_parts(
        "system base",
        "<p>Current</p>",
        2,
        5,
        cfg,
        previous_chunk_html=plain_previous,
    )
    assert "line one" not in user_prompt
    assert "line two\nline three" in user_prompt


def test_translation_chunk_prompt_ui_and_paths_use_new_toggle_contract():
    transate_source = (SRC / "TransateKRtoEN.py").read_text(encoding="utf-8")
    settings_source = (SRC / "other_settings.py").read_text(encoding="utf-8")
    dialog_source = settings_source[
        settings_source.index("def configure_translation_chunk_prompt"):
        settings_source.index("def configure_image_chunk_prompt")
    ]

    assert transate_source.count("_build_translation_chunk_prompt_parts(") >= 3
    assert "chunk_prompt_template =" not in transate_source
    assert "Enable chunk prompt" in dialog_source
    assert "Include previous chunk" in dialog_source
    assert "HTML tags or lines" in dialog_source
    assert "previous_chunk_context_limit" in dialog_source
    assert "PREVIOUS_CHUNK_CONTEXT_LIMIT" in transate_source
    assert "translation_chunk_prompt_role" in dialog_source
    assert '"{chunk_html}"' not in dialog_source


def test_sdlxliff_review_button_is_not_extraction_mode_gated():
    source = (SRC / "Retranslation_GUI.py").read_text(encoding="utf-8")

    assert "Review source -> output" in source
    assert "text_analysis_btn.setVisible(True)" in source
    assert "text_analysis_btn.setEnabled(True)" in source
    assert "No BeautifulSoup SDLXLIFF sidecars" not in source
    assert "Text Analysis is available for BeautifulSoup outputs" not in source
    assert "_text_analysis_is_beautifulsoup_mode" not in source
    assert "_text_analysis_profile_allowed" not in source


def test_sdlxliff_review_ignores_empty_source_paragraphs_for_alignment(tmp_path):
    sidecar = tmp_path / "response_chapter_notice0004.html.sdlxliff"
    source_html = (
        "<html><body>"
        "<h1>Notice: Cover completed in source language!</h1>"
        "<p></p>"
        "<p>Child version cover complete source text.</p>"
        "<p>Adult version cover next-time source text.</p>"
        "</body></html>"
    )
    target_html = (
        "<html><body>"
        "<h1>Notice: Cover completed!</h1>"
        "<p>Cover completed for the child version of the three slaves!</p>"
        "<p>Next time, I will return with the adult version cover.</p>"
        "</body></html>"
    )
    sidecar.write_text(
        f"""<?xml version="1.0" encoding="utf-8"?>
<xliff xmlns="urn:oasis:names:tc:xliff:document:1.2" version="1.2">
  <file original="chapter_notice0004.xhtml" source-language="ko-KR" target-language="en-US">
    <body>
      <trans-unit id="html">
        <source><![CDATA[{source_html}]]></source>
        <target><![CDATA[{target_html}]]></target>
      </trans-unit>
    </body>
  </file>
</xliff>
""",
        encoding="utf-8",
    )
    dialog = SDLXLIFFReviewDialog.__new__(SDLXLIFFReviewDialog)

    piece = dialog._build_piece(str(sidecar), 0, {"output_name": "response_chapter_notice0004.html"})

    assert piece["source_count"] == 3
    assert piece["target_count"] == 3
    assert piece["mismatch"] is False
    assert piece["red_count"] == 0
    assert [row["source"] for row in piece["rows"]] == [
        "Notice: Cover completed in source language!",
        "Child version cover complete source text.",
        "Adult version cover next-time source text.",
    ]
    assert [row["target"] for row in piece["rows"]] == [
        "Notice: Cover completed!",
        "Cover completed for the child version of the three slaves!",
        "Next time, I will return with the adult version cover.",
    ]
    assert all(row["source"] for row in piece["rows"])


def test_sdlxliff_review_user_text_in_empty_dom_slot_is_added_without_offset(tmp_path):
    output_name = "response_chapter_notice0003.html"
    source_html = (
        "<html><body>"
        "<h1>Notice heading</h1>"
        "<p>First source paragraph.</p>"
        "<p><br/></p>"
        "<p>Second source paragraph.</p>"
        "<p>Third source paragraph.</p>"
        "</body></html>"
    )
    target_html = (
        "<html><body>"
        "<h1>Notice heading</h1>"
        "<p>First source paragraph.</p>"
        "<p>ss</p>"
        "<p>Second source paragraph.</p>"
        "<p>Third source paragraph.</p>"
        "</body></html>"
    )
    sidecar = _shared_write_html_sdlxliff_sidecar(
        str(tmp_path),
        output_name,
        {"original_basename": "chapter_notice0003.xhtml"},
        source_html,
        target_html,
        raise_errors=True,
    )
    dialog = SDLXLIFFReviewDialog.__new__(SDLXLIFFReviewDialog)

    piece = dialog._build_piece(sidecar, 0, {"output_name": output_name})

    assert piece["source_count"] == 4
    assert piece["target_count"] == 4
    assert [(row["source"], row["target"]) for row in piece["rows"]] == [
        ("Notice heading", "Notice heading"),
        ("First source paragraph.", "First source paragraph."),
        ("", "ss"),
        ("Second source paragraph.", "Second source paragraph."),
        ("Third source paragraph.", "Third source paragraph."),
    ]
    added = piece["rows"][2]
    assert added["source_tag"] == ""
    assert added["target_tag"] == ""
    assert added["target_dom_tag"] == "p"
    assert added["source_tag_label"] == "TN(1)"
    assert added["target_tag_label"] == "TN(1)"
    assert added["source_missing"] is False
    assert added["target_missing"] is False
    assert added["translator_note"] is True
    assert added["translator_note_ordinal"] == 1
    assert added["status"] == "green"
    assert added["reason"] == "translator note"
    assert [row["source_tag_label"] for row in piece["rows"]] == [
        "h1", "p", "TN(1)", "p(2)", "p(3)",
    ]
    assert [row["target_tag_label"] for row in piece["rows"]] == [
        "h1", "p", "TN(1)", "p(2)", "p(3)",
    ]
    row_snapshot = dialog._review_row_snapshot(added)
    row_model = dialog._build_review_piece_render_model_from_rows(
        [row_snapshot],
        1200,
    )["rows"][0]
    assert row_model["source_missing"] is False
    assert row_model["target_missing"] is False
    assert row_model["translator_note"] is True
    assert dialog._tag_label_text("", "", "TN(1)", "TN(1)") == "TN(1)"


def test_sdlxliff_review_inserted_target_node_uses_text_anchor_without_offset(tmp_path):
    output_name = "response_chapter_inserted.html"
    source_html = (
        "<html><body><p>One.</p><p>Two.</p><p>Three.</p></body></html>"
    )
    target_html = (
        "<html><body><p>One.</p><p>User addition.</p><p>Two.</p><p>Three.</p></body></html>"
    )
    sidecar = _shared_write_html_sdlxliff_sidecar(
        str(tmp_path),
        output_name,
        {"original_basename": "chapter_inserted.xhtml"},
        source_html,
        target_html,
        raise_errors=True,
    )
    dialog = SDLXLIFFReviewDialog.__new__(SDLXLIFFReviewDialog)

    piece = dialog._build_piece(sidecar, 0, {"output_name": output_name})

    assert [(row["source"], row["target"]) for row in piece["rows"]] == [
        ("One.", "One."),
        ("", "User addition."),
        ("Two.", "Two."),
        ("Three.", "Three."),
    ]


def test_sdlxliff_manual_inserted_paragraph_is_translator_note_and_does_not_shift_ordinals(tmp_path):
    output_name = "response_chapter_manual_inserted.html"
    source_html = (
        "<html><body><p>One.</p><p>Two.</p><p>Three.</p></body></html>"
    )
    target_html = (
        "<html><body><p>One.</p><p>User addition one.</p>"
        "<p>User addition two.</p><p>Two.</p><p>Three.</p></body></html>"
    )
    sidecar = _shared_write_html_sdlxliff_sidecar(
        str(tmp_path),
        output_name,
        {"original_basename": "chapter_manual_inserted.xhtml"},
        source_html,
        target_html,
        raise_errors=True,
    )
    tree = etree.parse(sidecar)
    file_element = tree.xpath("//*[local-name()='file']")[0]
    file_element.set("{urn:glossarion:sdlxliff}manual-editing", "true")
    tree.write(sidecar, encoding="utf-8", xml_declaration=True)

    dialog = SDLXLIFFReviewDialog.__new__(SDLXLIFFReviewDialog)
    piece = dialog._build_piece(sidecar, 0, {"output_name": output_name})

    assert [(row["source"], row["target"]) for row in piece["rows"]] == [
        ("One.", "One."),
        ("", "User addition one."),
        ("", "User addition two."),
        ("Two.", "Two."),
        ("Three.", "Three."),
    ]
    first_note, second_note = piece["rows"][1:3]
    assert first_note["translator_note"] is True
    assert first_note["translator_note_ordinal"] == 1
    assert first_note["source_tag"] == ""
    assert first_note["target_tag"] == ""
    assert first_note["target_dom_tag"] == "p"
    assert first_note["source_tag_label"] == "TN(1)"
    assert first_note["target_tag_label"] == "TN(1)"
    assert first_note["source_missing"] is False
    assert first_note["status"] == "green"
    assert second_note["translator_note"] is True
    assert second_note["translator_note_ordinal"] == 2
    assert second_note["source_tag_label"] == "TN(2)"
    assert second_note["target_tag_label"] == "TN(2)"
    assert piece["source_count"] == 3
    assert piece["target_count"] == 3
    assert [row["source_tag_label"] for row in piece["rows"]] == [
        "p", "TN(1)", "TN(2)", "p(2)", "p(3)",
    ]
    assert [row["target_tag_label"] for row in piece["rows"]] == [
        "p", "TN(1)", "TN(2)", "p(2)", "p(3)",
    ]


def test_sdlxliff_notepad_added_row_does_not_offset_original_edit_history(tmp_path):
    output_name = "response_chapter_history.html"
    source_html = (
        "<html><body><p>One.</p><p><br/></p><p>Two.</p><p>Three.</p></body></html>"
    )
    initial_target = (
        "<html><body><p>One.</p><p><br/></p><p>Two.</p><p>Three.</p></body></html>"
    )
    edited_target = (
        "<html><body><p>One.</p><p>User addition.</p><p>Two.</p><p>Three.</p></body></html>"
    )
    sidecar = _shared_write_html_sdlxliff_sidecar(
        str(tmp_path),
        output_name,
        {"original_basename": "chapter_history.xhtml"},
        source_html,
        initial_target,
        raise_errors=True,
    )
    dialog = SDLXLIFFReviewDialog.__new__(SDLXLIFFReviewDialog)
    dialog.output_dir = str(tmp_path)
    piece = dialog._build_piece(sidecar, 0, {"output_name": output_name})
    for row in piece["rows"]:
        row["target_original"] = f"original:{row['source']}"

    dialog.pieces = [piece]
    dialog._piece_pages = {}
    dialog._refresh_piece_list_item = lambda _index: None
    dialog._refresh_piece_header = lambda _index: None
    dialog._current_review_signature = lambda: ()

    assert dialog._apply_notepad_document_edit(0, edited_target) is True

    rebuilt_rows = dialog.pieces[0]["rows"]
    added = next(row for row in rebuilt_rows if row["target"] == "User addition.")
    second = next(row for row in rebuilt_rows if row["source"] == "Two.")
    third = next(row for row in rebuilt_rows if row["source"] == "Three.")
    assert added["source"] == ""
    assert added["target_original"] == "User addition."
    assert second["target_original"] == "original:Two."
    assert third["target_original"] == "original:Three."


def test_sdlxliff_review_ignores_invisible_empty_html_tags(tmp_path):
    sidecar = tmp_path / "response_chapter_empty_tags.html.sdlxliff"
    source_html = (
        "<html><body>"
        "<p></p>"
        "<p>&nbsp;</p>"
        "<p>\u200b</p>"
        "<p>Real source text.</p>"
        "</body></html>"
    )
    target_html = (
        "<html><body>"
        "<p> </p>"
        "<p>&#8203;</p>"
        "<p>Real target text.</p>"
        "</body></html>"
    )
    sidecar.write_text(
        f"""<?xml version="1.0" encoding="utf-8"?>
<xliff xmlns="urn:oasis:names:tc:xliff:document:1.2" version="1.2">
  <file original="chapter_empty_tags.xhtml" source-language="ko-KR" target-language="en-US">
    <body>
      <trans-unit id="html">
        <source><![CDATA[{source_html}]]></source>
        <target><![CDATA[{target_html}]]></target>
      </trans-unit>
    </body>
  </file>
</xliff>
""",
        encoding="utf-8",
    )
    dialog = SDLXLIFFReviewDialog.__new__(SDLXLIFFReviewDialog)

    piece = dialog._build_piece(str(sidecar), 0, {"output_name": "response_chapter_empty_tags.html"})

    assert piece["source_count"] == 1
    assert piece["target_count"] == 1
    assert len(piece["rows"]) == 1
    assert piece["rows"][0]["source_tag_label"] == "p"
    assert piece["rows"][0]["target_tag_label"] == "p"
    assert piece["rows"][0]["source"] == "Real source text."
    assert piece["rows"][0]["target"] == "Real target text."


def test_sdlxliff_review_treats_list_items_as_rows_not_list_containers(tmp_path):
    sidecar = tmp_path / "response_chapter_list_items.html.sdlxliff"
    source_html = (
        "<html><body><ul>"
        "<li>Source list item one.</li>"
        "<li>Source list item two.</li>"
        "</ul></body></html>"
    )
    target_html = (
        "<html><body><ul>"
        "<li>Target list item one.</li>"
        "<li>Target list item two.</li>"
        "</ul></body></html>"
    )
    sidecar.write_text(
        f"""<?xml version="1.0" encoding="utf-8"?>
<xliff xmlns="urn:oasis:names:tc:xliff:document:1.2" version="1.2">
  <file original="chapter_list_items.xhtml" source-language="ko-KR" target-language="en-US">
    <body><trans-unit id="html">
      <source><![CDATA[{source_html}]]></source>
      <target><![CDATA[{target_html}]]></target>
    </trans-unit></body>
  </file>
</xliff>
""",
        encoding="utf-8",
    )
    dialog = SDLXLIFFReviewDialog.__new__(SDLXLIFFReviewDialog)

    piece = dialog._build_piece(
        str(sidecar), 0, {"output_name": "response_chapter_list_items.html"}
    )

    assert piece["source_count"] == 2
    assert piece["target_count"] == 2
    assert piece["mismatch"] is False
    assert [row["source_tag"] for row in piece["rows"]] == ["li", "li"]
    assert [row["target_tag"] for row in piece["rows"]] == ["li", "li"]
    assert dialog._tooltip_batch_tag_name("li") == "li"

    edited_html = dialog._target_html_with_edit(
        piece, piece["rows"][1], "Edited target list item two."
    )
    edited_soup = BeautifulSoup(edited_html, "html.parser")
    assert len(edited_soup.find_all("ul")) == 1
    assert [item.get_text(" ", strip=True) for item in edited_soup.find_all("li")] == [
        "Target list item one.",
        "Edited target list item two.",
    ]


def test_sdlxliff_review_treats_paragraph_and_list_item_as_equivalent_text_units(tmp_path):
    sidecar = tmp_path / "response_chapter_p_to_li.html.sdlxliff"
    source_html = "<html><body><p>Source sentence.</p></body></html>"
    target_html = "<html><body><ul><li>Translated sentence.</li></ul></body></html>"
    sidecar.write_text(
        f"""<?xml version="1.0" encoding="utf-8"?>
<xliff xmlns="urn:oasis:names:tc:xliff:document:1.2" version="1.2">
  <file original="chapter_p_to_li.xhtml" source-language="ko-KR" target-language="en-US">
    <body><trans-unit id="html">
      <source><![CDATA[{source_html}]]></source>
      <target><![CDATA[{target_html}]]></target>
    </trans-unit></body>
  </file>
</xliff>
""",
        encoding="utf-8",
    )
    dialog = SDLXLIFFReviewDialog.__new__(SDLXLIFFReviewDialog)

    piece = dialog._build_piece(
        str(sidecar), 0, {"output_name": "response_chapter_p_to_li.html"}
    )

    assert piece["source_count"] == 1
    assert piece["target_count"] == 1
    assert piece["mismatch"] is False
    assert piece["red_count"] == 0
    assert piece["rows"][0]["source_tag"] == "p"
    assert piece["rows"][0]["target_tag"] == "li"
    assert piece["rows"][0]["status"] == "green"


def test_sdlxliff_review_counts_hr_as_asterisk_paragraph_unit(tmp_path):
    sidecar = tmp_path / "response_chapter_hr.html.sdlxliff"
    source_html = (
        "<html><body>"
        "<p>Source before.</p><p>*****</p><p>Source after.</p>"
        "</body></html>"
    )
    target_html = (
        "<html><body>"
        "<p>Target before.</p><hr/><p>Target after.</p>"
        "</body></html>"
    )
    sidecar.write_text(
        f"""<?xml version="1.0" encoding="utf-8"?>
<xliff xmlns="urn:oasis:names:tc:xliff:document:1.2" version="1.2">
  <file original="chapter_hr.xhtml" source-language="ko-KR" target-language="en-US">
    <body><trans-unit id="html">
      <source><![CDATA[{source_html}]]></source>
      <target><![CDATA[{target_html}]]></target>
    </trans-unit></body>
  </file>
</xliff>
""",
        encoding="utf-8",
    )
    dialog = SDLXLIFFReviewDialog.__new__(SDLXLIFFReviewDialog)

    piece = dialog._build_piece(
        str(sidecar), 0, {"output_name": "response_chapter_hr.html"}
    )

    assert piece["source_count"] == 3
    assert piece["target_count"] == 3
    assert piece["mismatch"] is False
    assert [row["target"] for row in piece["rows"]] == [
        "Target before.", "*****", "Target after."
    ]
    assert [row["target_tag_label"] for row in piece["rows"]] == [
        "p", "p(2)", "p(3)"
    ]

    edited_html = dialog._target_html_with_edit(
        piece, piece["rows"][1], "Edited separator."
    )
    edited_soup = BeautifulSoup(edited_html, "html.parser")
    assert edited_soup.find("hr") is None
    assert [tag.get_text(" ", strip=True) for tag in edited_soup.find_all("p")] == [
        "Target before.", "Edited separator.", "Target after."
    ]


def test_qa_counts_and_text_checks_include_list_items():
    list_html = "<ul><li>First list unit.</li><li>Second list unit.</li></ul>"

    assert _count_beautifulsoup_review_tags(list_html) == {"li": 2}
    assert _missing_beautifulsoup_tags_issue(
        {"li": 20},
        {"li": 19},
        min_source_paragraph_tags=20,
    ) == "missing_tags: 20/19 (-1)"
    assert _extract_paragraphs(list_html) == [
        "First list unit.",
        "Second list unit.",
    ]
    assert len(_missing_ending_quotation_paragraphs(
        '<ul><li>"Missing closing quotation.</li></ul>'
    )) == 1


def test_qa_structure_checks_list_tag_balance(tmp_path):
    valid_path = tmp_path / "valid_list.html"
    valid_path.write_text(
        "<html><body><ul><li>One</li><li>Two</li></ul></body></html>",
        encoding="utf-8",
    )
    invalid_path = tmp_path / "invalid_list.html"
    invalid_path.write_text(
        "<html><body><ul><li>One</li><li>Two</ul></body></html>",
        encoding="utf-8",
    )

    valid_has_issues, valid_issues = check_html_structure_issues(
        str(valid_path), lambda _message: None, check_header_tags=False
    )
    invalid_has_issues, invalid_issues = check_html_structure_issues(
        str(invalid_path), lambda _message: None, check_header_tags=False
    )

    assert valid_has_issues is False
    assert "unclosed_html_tags" not in valid_issues
    assert invalid_has_issues is True
    assert "unclosed_html_tags" in invalid_issues


def test_sdlxliff_review_heading_level_change_is_yellow(tmp_path):
    sidecar = tmp_path / "response_chapter_heading.html.sdlxliff"
    source_html = "<html><body><h1>Source heading</h1><p>Source body</p></body></html>"
    target_html = "<html><body><h2>Translated heading</h2><p>Translated body</p></body></html>"
    sidecar.write_text(
        f"""<?xml version="1.0" encoding="utf-8"?>
<xliff xmlns="urn:oasis:names:tc:xliff:document:1.2" version="1.2">
  <file original="chapter_heading.xhtml" source-language="ko-KR" target-language="en-US">
    <body>
      <trans-unit id="html">
        <source><![CDATA[{source_html}]]></source>
        <target><![CDATA[{target_html}]]></target>
      </trans-unit>
    </body>
  </file>
</xliff>
""",
        encoding="utf-8",
    )
    dialog = SDLXLIFFReviewDialog.__new__(SDLXLIFFReviewDialog)

    piece = dialog._build_piece(str(sidecar), 0, {"output_name": "response_chapter_heading.html"})

    assert piece["mismatch"] is False
    assert piece["red_count"] == 0
    assert piece["yellow_count"] == 1
    assert piece["rows"][0]["source_tag"] == "h1"
    assert piece["rows"][0]["target_tag"] == "h2"
    assert piece["rows"][0]["status"] == "yellow"
    assert piece["rows"][0]["reason"] == "heading level changed"


def test_sdlxliff_review_missing_initial_heading_does_not_offset_paragraphs(tmp_path):
    sidecar = tmp_path / "response_chapter_missing_heading.html.sdlxliff"
    source_html = (
        "<html><body>"
        "<h1>Source heading</h1>"
        "<p>Source paragraph one</p>"
        "<p>Source paragraph two</p>"
        "</body></html>"
    )
    target_html = (
        "<html><body>"
        "<p>Translated paragraph one</p>"
        "<p>Translated paragraph two</p>"
        "</body></html>"
    )
    sidecar.write_text(
        f"""<?xml version="1.0" encoding="utf-8"?>
<xliff xmlns="urn:oasis:names:tc:xliff:document:1.2" version="1.2">
  <file original="chapter_heading.xhtml" source-language="ko-KR" target-language="en-US">
    <body>
      <trans-unit id="html">
        <source><![CDATA[{source_html}]]></source>
        <target><![CDATA[{target_html}]]></target>
      </trans-unit>
    </body>
  </file>
</xliff>
""",
        encoding="utf-8",
    )
    dialog = SDLXLIFFReviewDialog.__new__(SDLXLIFFReviewDialog)

    piece = dialog._build_piece(str(sidecar), 0, {"output_name": "response_chapter_missing_heading.html"})

    assert piece["source_count"] == 3
    assert piece["target_count"] == 2
    assert piece["rows"][0]["source_tag"] == "h1"
    assert piece["rows"][0]["target_tag"] == ""
    assert piece["rows"][0]["status"] == "red"
    assert piece["rows"][1]["source_tag"] == "p"
    assert piece["rows"][1]["target_tag"] == "p"
    assert piece["rows"][1]["source"] == "Source paragraph one"
    assert piece["rows"][1]["target"] == "Translated paragraph one"
    assert piece["rows"][2]["source"] == "Source paragraph two"
    assert piece["rows"][2]["target"] == "Translated paragraph two"


def test_sdlxliff_review_extra_initial_heading_does_not_offset_paragraphs(tmp_path):
    sidecar = tmp_path / "response_chapter_extra_heading.html.sdlxliff"
    source_html = (
        "<html><body>"
        "<p>Source paragraph one</p>"
        "<p>Source paragraph two</p>"
        "</body></html>"
    )
    target_html = (
        "<html><body>"
        "<h2>Added translated heading</h2>"
        "<p>Translated paragraph one</p>"
        "<p>Translated paragraph two</p>"
        "</body></html>"
    )
    sidecar.write_text(
        f"""<?xml version="1.0" encoding="utf-8"?>
<xliff xmlns="urn:oasis:names:tc:xliff:document:1.2" version="1.2">
  <file original="chapter_heading.xhtml" source-language="ko-KR" target-language="en-US">
    <body>
      <trans-unit id="html">
        <source><![CDATA[{source_html}]]></source>
        <target><![CDATA[{target_html}]]></target>
      </trans-unit>
    </body>
  </file>
</xliff>
""",
        encoding="utf-8",
    )
    dialog = SDLXLIFFReviewDialog.__new__(SDLXLIFFReviewDialog)

    piece = dialog._build_piece(str(sidecar), 0, {"output_name": "response_chapter_extra_heading.html"})

    assert piece["source_count"] == 2
    assert piece["target_count"] == 3
    assert piece["rows"][0]["source_tag"] == ""
    assert piece["rows"][0]["target_tag"] == "h2"
    assert piece["rows"][0]["status"] == "red"
    assert piece["rows"][1]["source_tag"] == "p"
    assert piece["rows"][1]["target_tag"] == "p"
    assert piece["rows"][1]["source"] == "Source paragraph one"
    assert piece["rows"][1]["target"] == "Translated paragraph one"
    assert piece["rows"][2]["source"] == "Source paragraph two"
    assert piece["rows"][2]["target"] == "Translated paragraph two"


def test_sdlxliff_review_count_mismatch_promotes_top_skewed_row_to_yellow(tmp_path):
    sidecar = tmp_path / "response_chapter_skewed_extra.html.sdlxliff"
    source_html = (
        "<html><body>"
        "<p>그들은 경고를 지켰다.</p>"
        "<p>결국 그들은 경고를 지켰다.</p>"
        "<p>마지막 문장입니다.</p>"
        "</body></html>"
    )
    target_html = (
        "<html><body>"
        "<p>They kept the warning.</p>"
        "<p>In the end, they acted as if keeping their warning. Instead of her mother's eyes, "
        "they plucked out her own eyeball and made her swallow it, and her ear was torn off.</p>"
        "<p>This is the closing line.</p>"
        "<p>This is an added target row.</p>"
        "</body></html>"
    )
    sidecar.write_text(
        f"""<?xml version="1.0" encoding="utf-8"?>
<xliff xmlns="urn:oasis:names:tc:xliff:document:1.2" version="1.2">
  <file original="chapter_skewed.xhtml" source-language="ko-KR" target-language="en-US">
    <body>
      <trans-unit id="html">
        <source><![CDATA[{source_html}]]></source>
        <target><![CDATA[{target_html}]]></target>
      </trans-unit>
    </body>
  </file>
</xliff>
""",
        encoding="utf-8",
    )
    dialog = SDLXLIFFReviewDialog.__new__(SDLXLIFFReviewDialog)

    piece = dialog._build_piece(str(sidecar), 0, {"output_name": "response_chapter_skewed_extra.html"})

    assert piece["source_count"] == 3
    assert piece["target_count"] == 4
    assert piece["mismatch"] is True
    assert piece["red_count"] == 1
    assert piece["yellow_count"] == 1
    promoted = [row for row in piece["rows"] if row.get("reason", "").startswith("top translated-column skew")]
    assert len(promoted) == 1
    assert promoted[0]["source"] == "결국 그들은 경고를 지켰다."
    assert promoted[0]["status"] == "yellow"
    assert piece["rows"][-1]["status"] == "red"


def test_sdlxliff_review_count_mismatch_always_promotes_highest_ratio_row(tmp_path):
    sidecar = tmp_path / "response_chapter_one_missing.html.sdlxliff"
    source_html = (
        "<html><body>"
        "<p>Source first row.</p>"
        "<p>Source second row.</p>"
        "</body></html>"
    )
    target_html = (
        "<html><body>"
        "<p>Target first row.</p>"
        "</body></html>"
    )
    sidecar.write_text(
        f"""<?xml version="1.0" encoding="utf-8"?>
<xliff xmlns="urn:oasis:names:tc:xliff:document:1.2" version="1.2">
  <file original="chapter_one_missing.xhtml" source-language="ko-KR" target-language="en-US">
    <body>
      <trans-unit id="html">
        <source><![CDATA[{source_html}]]></source>
        <target><![CDATA[{target_html}]]></target>
      </trans-unit>
    </body>
  </file>
</xliff>
""",
        encoding="utf-8",
    )
    dialog = SDLXLIFFReviewDialog.__new__(SDLXLIFFReviewDialog)

    piece = dialog._build_piece(str(sidecar), 0, {"output_name": "response_chapter_one_missing.html"})

    assert piece["source_count"] == 2
    assert piece["target_count"] == 1
    assert piece["red_count"] == 1
    assert piece["yellow_count"] == 1
    assert piece["rows"][0]["status"] == "yellow"
    assert piece["rows"][0]["reason"].startswith("top translated-column skew")
    assert piece["rows"][1]["status"] == "red"


def test_sdlxliff_review_count_mismatch_prefers_translated_column_outlier():
    dialog = SDLXLIFFReviewDialog.__new__(SDLXLIFFReviewDialog)
    rows = [
        {
            "source_tag": "p",
            "source": "In the end, they acted as if they had followed their warning.",
            "tooltip_translation": "In the end, they acted as if they had followed their warning.",
            "target_tag": "p",
            "target": (
                "In the end, they acted as if keeping their warning. Instead of her mother's eyes, "
                "they plucked out her own eyeball and made her swallow it, and her ear was torn off."
            ),
            "status": "green",
            "reason": "ok",
        },
        {
            "source_tag": "p",
            "source": (
                "They say beastmen meat gets more tender the more you feed them their own kind, right? "
                "They even use it as medicine. The client specially requested it. The people handling "
                "the order explained the whole process in uncomfortable detail, including why the meat "
                "had to be prepared softly and why the client insisted on a young one."
            ),
            "tooltip_translation": (
                "They say beastmen meat gets more tender the more you feed them their own kind, right? "
                "They even use it as medicine. The client specially requested it. The people handling "
                "the order explained the whole process in uncomfortable detail, including why the meat "
                "had to be prepared softly and why the client insisted on a young one."
            ),
            "target_tag": "p",
            "target": (
                "They say beastmen meat gets more tender the more you feed them their own kind, right? "
                "They even use it as medicine. The client specially requested it. Asked us to make a "
                "plump and tender little one. The handlers described the request in detail and repeated "
                "that the client wanted the finished product soft, young, and carefully prepared."
            ),
            "status": "green",
            "reason": "ok",
        },
        {
            "source_tag": "p",
            "source": "Missing source row.",
            "target_tag": "",
            "target": "",
            "status": "red",
            "reason": "dropped/added",
        },
    ]

    promoted = dialog._promote_top_skewed_row_for_count_mismatch(rows, 3, 2)

    assert promoted is True
    assert rows[0]["status"] == "yellow"
    assert rows[0]["reason"].startswith("top translated-column skew")
    assert rows[1]["status"] == "green"
    assert len(rows[1]["target"]) > len(rows[0]["target"])
    assert SDLXLIFFReviewDialog._row_length_ratio(rows[0]) > SDLXLIFFReviewDialog._row_length_ratio(rows[1])


def test_sdlxliff_review_count_mismatch_marks_expanded_row_not_downstream_shift():
    dialog = SDLXLIFFReviewDialog.__new__(SDLXLIFFReviewDialog)
    rows = [
        {
            "source_tag": "p",
            "source": "결국, 그들은 경고를 지키듯 행동했다.",
            "tooltip_translation": "In the end, they acted as if they had followed their warning.",
            "target_tag": "p",
            "target": (
                "In the end, they acted as if keeping their warning. Instead of her mother's eyes, "
                "they plucked out her own eyeball and made her swallow it, and her ear was torn off."
            ),
            "status": "green",
            "reason": "ok",
        },
        {
            "source_tag": "p",
            "source": "“...따뜻해.”",
            "tooltip_translation": "...It's warm.",
            "target_tag": "p",
            "target": "It was the oldest warmth of 'home,' the one Piel thought she had forgotten.",
            "status": "green",
            "reason": "ok",
        },
        {
            "source_tag": "p",
            "source": "피엘이 잊었다고 생각했던, 가장 오래된 집의 온기였다.",
            "tooltip_translation": "It was the oldest warmth of 'home,' the one Piel thought she had forgotten.",
            "target_tag": "",
            "target": "",
            "status": "red",
            "reason": "dropped/added",
        },
    ]

    promoted = dialog._promote_top_skewed_row_for_count_mismatch(rows, 3, 2)

    assert promoted is True
    assert rows[0]["status"] == "yellow"
    assert rows[0]["reason"].startswith("top translated-column skew")
    assert rows[1]["status"] == "green"
    assert SDLXLIFFReviewDialog._row_length_ratio(rows[1]) > SDLXLIFFReviewDialog._row_length_ratio(rows[0])


def test_sdlxliff_review_manual_machine_accuracy_marks_all_inaccurate_rows_purple():
    dialog = SDLXLIFFReviewDialog.__new__(SDLXLIFFReviewDialog)
    piece = {
        "rows": [
            {
                "source_tag": "p",
                "source": "synthetic source alpha",
                "tooltip_translation": "alpha beta gamma delta epsilon zeta eta theta iota kappa",
                "target_tag": "p",
                "target": "red blue green yellow orange purple silver bronze copper iron",
                "status": "green",
                "reason": "ok",
            },
            {
                "source_tag": "p",
                "source": "synthetic source bravo",
                "tooltip_translation": "north south east west center inner outer upper lower",
                "target_tag": "p",
                "target": "circle square triangle diamond spiral ribbon anchor window",
                "status": "green",
                "reason": "ok",
            },
            {
                "source_tag": "p",
                "source": "synthetic source charlie",
                "tooltip_translation": "stable matching comparison text",
                "target_tag": "p",
                "target": "stable matching comparison text",
                "status": "green",
                "reason": "ok",
            },
            {
                "source_tag": "p",
                "source": "synthetic source delta",
                "tooltip_translation": "missing row comparison text",
                "target_tag": "",
                "target": "",
                "status": "red",
                "reason": "dropped/added",
            },
        ],
    }

    promoted_indices = dialog._promote_inaccurate_machine_translation_rows(piece)

    assert promoted_indices == [0, 1]
    assert piece["_machine_accuracy_review_active"] is True
    assert piece["rows"][0]["status"] == "purple"
    assert piece["rows"][0]["reason"].startswith("machine translation inaccurate")
    assert piece["rows"][1]["status"] == "purple"
    assert piece["rows"][2]["status"] == "green"


def test_sdlxliff_review_machine_accuracy_ignores_whitespace_only_differences():
    dialog = SDLXLIFFReviewDialog.__new__(SDLXLIFFReviewDialog)
    piece = {
        "rows": [
            {
                "source_tag": "p",
                "source": "synthetic source whitespace",
                "tooltip_translation": "alpha,\u00a0beta\ngamma.",
                "target_tag": "p",
                "target": "alpha, beta   gamma.",
                "status": "green",
                "reason": "ok",
            }
        ],
    }

    score = dialog._machine_translation_accuracy_score(piece["rows"], 0)
    promoted_indices = dialog._promote_inaccurate_machine_translation_rows(piece)

    assert score == 0.0
    assert promoted_indices == []
    assert piece["rows"][0]["status"] == "green"
    assert piece["rows"][0]["reason"] == "ok"


def test_sdlxliff_review_machine_accuracy_ignores_short_entries():
    dialog = SDLXLIFFReviewDialog.__new__(SDLXLIFFReviewDialog)
    piece = {
        "rows": [
            {
                "source_tag": "p",
                "source": "synthetic source short",
                "tooltip_translation": "Brief.",
                "target_tag": "p",
                "target": "Tiny.",
                "status": "green",
                "reason": "ok",
            }
        ],
    }

    score = dialog._machine_translation_accuracy_score(piece["rows"], 0)
    promoted_indices = dialog._promote_inaccurate_machine_translation_rows(piece)

    assert score == 0.0
    assert promoted_indices == []
    assert piece["rows"][0]["status"] == "green"
    assert piece["rows"][0]["reason"] == "ok"


def test_sdlxliff_review_machine_accuracy_scores_one_word_against_phrase():
    dialog = SDLXLIFFReviewDialog.__new__(SDLXLIFFReviewDialog)
    piece = {
        "rows": [
            {
                "source_tag": "p",
                "source": "synthetic source asymmetric short",
                "tooltip_translation": "Marker.",
                "target_tag": "p",
                "target": "alpha beta gamma delta.",
                "status": "green",
                "reason": "ok",
            }
        ],
    }

    score = dialog._machine_translation_accuracy_score(piece["rows"], 0)
    promoted_indices = dialog._promote_inaccurate_machine_translation_rows(piece)

    assert score >= dialog.MACHINE_TRANSLATION_INACCURACY_THRESHOLD
    assert promoted_indices == [0]
    assert piece["rows"][0]["status"] == "purple"
    assert piece["rows"][0]["reason"].startswith("machine translation inaccurate")


def test_sdlxliff_review_machine_accuracy_ignores_punctuation_only_short_entries():
    dialog = SDLXLIFFReviewDialog.__new__(SDLXLIFFReviewDialog)
    piece = {
        "rows": [
            {
                "source_tag": "p",
                "source": "synthetic source punctuation",
                "tooltip_translation": "'...alpha.'",
                "target_tag": "p",
                "target": "\"... alpha.\"",
                "status": "purple",
                "reason": "machine translation inaccurate (old)",
            }
        ],
    }

    score = dialog._machine_translation_accuracy_score(piece["rows"], 0)
    promoted_indices = dialog._promote_inaccurate_machine_translation_rows(piece)

    assert score == 0.0
    assert promoted_indices == []
    assert piece["rows"][0]["status"] == "green"
    assert piece["rows"][0]["reason"] == "ok"


def test_sdlxliff_review_machine_accuracy_flags_low_content_overlap():
    dialog = SDLXLIFFReviewDialog.__new__(SDLXLIFFReviewDialog)
    piece = {
        "rows": [
            {
                "source_tag": "p",
                "source": "synthetic source low overlap",
                "tooltip_translation": (
                    "alpha beta gamma delta epsilon zeta eta theta iota kappa "
                    "lambda mu nu xi omicron"
                ),
                "target_tag": "p",
                "target": (
                    "red blue green yellow orange purple silver bronze copper iron "
                    "north south east west center"
                ),
                "status": "green",
                "reason": "ok",
            }
        ],
    }

    score = dialog._machine_translation_accuracy_score(piece["rows"], 0)
    promoted_indices = dialog._promote_inaccurate_machine_translation_rows(piece)

    assert score >= dialog.MACHINE_TRANSLATION_INACCURACY_THRESHOLD
    assert promoted_indices == [0]
    assert piece["rows"][0]["status"] == "purple"
    assert piece["rows"][0]["reason"].startswith("machine translation inaccurate")


def test_sdlxliff_review_machine_accuracy_uses_saved_threshold():
    dialog = SDLXLIFFReviewDialog.__new__(SDLXLIFFReviewDialog)
    piece = {
        "rows": [
            {
                "source_tag": "p",
                "source": "synthetic source threshold",
                "tooltip_translation": (
                    "alpha beta gamma delta epsilon zeta eta theta iota kappa "
                    "lambda mu nu xi omicron"
                ),
                "target_tag": "p",
                "target": (
                    "red blue green yellow orange purple silver bronze copper iron "
                    "north south east west center"
                ),
                "status": "green",
                "reason": "ok",
            }
        ],
    }

    dialog._config = {dialog.MACHINE_TRANSLATION_THRESHOLD_CONFIG_KEY: 1000}
    assert dialog._promote_inaccurate_machine_translation_rows(piece) == []
    assert piece["rows"][0]["status"] == "green"

    dialog._config[dialog.MACHINE_TRANSLATION_THRESHOLD_CONFIG_KEY] = 25
    assert dialog._promote_inaccurate_machine_translation_rows(piece) == [0]
    assert piece["rows"][0]["status"] == "purple"


def test_sdlxliff_review_machine_accuracy_threshold_saves_to_parent_config():
    class Parent:
        def __init__(self):
            self.config = {}
            self.save_calls = 0

        def save_config(self, show_message=True):
            self.save_calls += 1

    parent = Parent()
    dialog = SDLXLIFFReviewDialog.__new__(SDLXLIFFReviewDialog)
    dialog._config = {}
    dialog._context_parent = parent

    threshold, saved = dialog._set_machine_translation_inaccuracy_threshold(175.4)

    assert saved is True
    assert threshold == 175.4
    assert dialog._config[dialog.MACHINE_TRANSLATION_THRESHOLD_CONFIG_KEY] == 175.4
    assert parent.config[dialog.MACHINE_TRANSLATION_THRESHOLD_CONFIG_KEY] == 175.4
    assert parent.save_calls == 1


def test_sdlxliff_review_two_column_layout_toggle_saves_to_parent_config():
    class Parent:
        def __init__(self):
            self.config = {}
            self.save_calls = 0

        def save_config(self, show_message=True):
            self.save_calls += 1

    parent = Parent()
    dialog = SDLXLIFFReviewDialog.__new__(SDLXLIFFReviewDialog)
    dialog._config = {}
    dialog._context_parent = parent
    dialog.two_column_layout_btn = None
    dialog.pieces = []
    dialog._piece_pages = {}
    dialog._piece_render_complete = set()
    dialog._review_data_preload_token = 0
    dialog.piece_list = None

    dialog._set_review_two_column_layout(True)

    assert dialog._two_column_layout_enabled is True
    assert dialog._config[dialog.TWO_COLUMN_LAYOUT_CONFIG_KEY] is True
    assert parent.config[dialog.TWO_COLUMN_LAYOUT_CONFIG_KEY] is True
    assert parent.save_calls == 1


def test_sdlxliff_review_two_column_layout_defaults_on_and_reads_legacy_config():
    dialog = SDLXLIFFReviewDialog.__new__(SDLXLIFFReviewDialog)
    dialog._config = {}
    assert dialog._review_two_column_layout_enabled() is True

    dialog._config = {dialog.LEGACY_ONE_ROW_LAYOUT_CONFIG_KEY: False}
    assert dialog._review_two_column_layout_enabled() is False

    dialog._config = {
        dialog.LEGACY_ONE_ROW_LAYOUT_CONFIG_KEY: False,
        dialog.LEGACY_ONE_COLUMN_LAYOUT_CONFIG_KEY: True,
    }
    assert dialog._review_two_column_layout_enabled() is True

    dialog._config = {
        dialog.LEGACY_ONE_COLUMN_LAYOUT_CONFIG_KEY: False,
        dialog.TWO_COLUMN_LAYOUT_CONFIG_KEY: True,
    }
    assert dialog._review_two_column_layout_enabled() is True


def test_sdlxliff_review_build_uses_machine_translation_for_top_skew(tmp_path):
    sidecar_dir = tmp_path / "SDLXLIFF"
    sidecar_dir.mkdir()
    output_name = "response_chapter_mt_skew.html"
    sidecar = sidecar_dir / f"{output_name}.sdlxliff"
    source_html = (
        "<html><body>"
        "<p>Synthetic source alpha.</p>"
        "<p>Synthetic source beta.</p>"
        "<p>Synthetic source gamma.</p>"
        "</body></html>"
    )
    target_html = (
        "<html><body>"
        "<p>Synthetic alpha output with many additional words that make this translated column expanded.</p>"
        "<p>Synthetic gamma output.</p>"
        "</body></html>"
    )
    sidecar.write_text(
        f"""<?xml version="1.0" encoding="utf-8"?>
<xliff xmlns="urn:oasis:names:tc:xliff:document:1.2" version="1.2">
  <file original="chapter_mt_skew.xhtml" source-language="ko-KR" target-language="en-US">
    <body>
      <trans-unit id="html">
        <source><![CDATA[{source_html}]]></source>
        <target><![CDATA[{target_html}]]></target>
      </trans-unit>
    </body>
  </file>
</xliff>
""",
        encoding="utf-8",
    )
    dialog = SDLXLIFFReviewDialog.__new__(SDLXLIFFReviewDialog)
    dialog.output_dir = str(tmp_path)
    dialog._config = {"output_language": "English"}

    first_piece = dialog._build_piece(str(sidecar), 0, {"output_name": output_name})
    dialog._write_machine_translation_entries(
        first_piece,
        [
            (first_piece["rows"][0], "Synthetic alpha output."),
            (first_piece["rows"][1], "Synthetic beta output."),
            (
                first_piece["rows"][2],
                "Synthetic gamma output.",
            ),
        ],
    )

    piece = dialog._build_piece(str(sidecar), 0, {"output_name": output_name})

    promoted = [row for row in piece["rows"] if row.get("reason", "").startswith("top translated-column skew")]
    assert len(promoted) == 1
    assert promoted[0]["source"] == "Synthetic source alpha."
    assert piece["rows"][1]["status"] == "green"
    assert piece["rows"][2]["status"] == "red"


def test_sdlxliff_review_heading_to_paragraph_mismatch_is_yellow(tmp_path):
    sidecar = tmp_path / "response_chapter_heading_to_p.html.sdlxliff"
    source_html = "<html><body><h1>Source heading</h1></body></html>"
    target_html = "<html><body><p>Translated heading</p></body></html>"
    sidecar.write_text(
        f"""<?xml version="1.0" encoding="utf-8"?>
<xliff xmlns="urn:oasis:names:tc:xliff:document:1.2" version="1.2">
  <file original="chapter_heading.xhtml" source-language="ko-KR" target-language="en-US">
    <body>
      <trans-unit id="html">
        <source><![CDATA[{source_html}]]></source>
        <target><![CDATA[{target_html}]]></target>
      </trans-unit>
    </body>
  </file>
</xliff>
""",
        encoding="utf-8",
    )
    dialog = SDLXLIFFReviewDialog.__new__(SDLXLIFFReviewDialog)

    piece = dialog._build_piece(str(sidecar), 0, {"output_name": "response_chapter_heading_to_p.html"})

    assert piece["mismatch"] is False
    assert piece["red_count"] == 0
    assert piece["yellow_count"] == 1
    assert piece["rows"][0]["source_tag"] == "h1"
    assert piece["rows"][0]["target_tag"] == "p"
    assert piece["rows"][0]["status"] == "yellow"
    assert piece["rows"][0]["reason"] == "heading/paragraph tag changed"


def test_sdlxliff_review_heading_to_paragraph_does_not_offset_following_paragraphs(tmp_path):
    sidecar = tmp_path / "response_chapter_heading_to_p_sequence.html.sdlxliff"
    source_html = (
        "<html><body>"
        "<h2>Source heading</h2>"
        "<p>Source paragraph one</p>"
        "<p>Source paragraph two</p>"
        "</body></html>"
    )
    target_html = (
        "<html><body>"
        "<p>Translated heading</p>"
        "<p>Translated paragraph one</p>"
        "<p>Translated paragraph two</p>"
        "</body></html>"
    )
    sidecar.write_text(
        f"""<?xml version="1.0" encoding="utf-8"?>
<xliff xmlns="urn:oasis:names:tc:xliff:document:1.2" version="1.2">
  <file original="chapter_heading.xhtml" source-language="ko-KR" target-language="en-US">
    <body>
      <trans-unit id="html">
        <source><![CDATA[{source_html}]]></source>
        <target><![CDATA[{target_html}]]></target>
      </trans-unit>
    </body>
  </file>
</xliff>
""",
        encoding="utf-8",
    )
    dialog = SDLXLIFFReviewDialog.__new__(SDLXLIFFReviewDialog)

    piece = dialog._build_piece(str(sidecar), 0, {"output_name": "response_chapter_heading_to_p_sequence.html"})

    assert piece["source_count"] == 3
    assert piece["target_count"] == 3
    assert piece["red_count"] == 0
    assert piece["yellow_count"] == 1
    assert piece["rows"][0]["source_tag"] == "h2"
    assert piece["rows"][0]["target_tag"] == "p"
    assert piece["rows"][0]["status"] == "yellow"
    assert piece["rows"][0]["reason"] == "heading/paragraph tag changed"
    assert piece["rows"][1]["source_tag"] == "p"
    assert piece["rows"][1]["target_tag"] == "p"
    assert piece["rows"][1]["source"] == "Source paragraph one"
    assert piece["rows"][1]["target"] == "Translated paragraph one"
    assert piece["rows"][2]["source"] == "Source paragraph two"
    assert piece["rows"][2]["target"] == "Translated paragraph two"


def test_sdlxliff_review_translate_tooltips_uses_machine_translation_provider():
    source = (SRC / "Retranslation_GUI.py").read_text(encoding="utf-8")

    assert "🌐 Generate Machine Translation Preview" in source
    assert "MACHINE_TRANSLATION_PROVIDER_CONFIG_KEY" in source
    assert "MACHINE_TRANSLATION_PROVIDER_LABELS" in source
    assert '"deepl": "DeepL"' in source
    assert '"bing": "Bing"' in source
    assert '"yandex": "Yandex"' in source
    assert "customContextMenuRequested.connect(self._show_machine_translation_provider_menu)" in source
    assert "_set_machine_translation_provider" in source
    assert "_prompt_machine_translation_credentials" in source
    assert "_machine_translation_translator" in source
    assert "MACHINE_TRANSLATION_API_KEY_CONFIG_KEYS" in source
    assert "_encrypt_machine_translation_api_key" in source
    assert "_decrypt_machine_translation_api_key" in source
    assert "_machine_translation_config_value" in source
    assert "honor_global_stop=False" in source
    assert "QLineEdit.Password" in source
    assert "from google_free_translate import GoogleFreeTranslateNew" in source
    assert 'name="sdlxliff-machine-translation-preview"' in source
    assert "batch_html = self._tooltip_batch_html(work)" in source
    assert "result = translator.translate(batch_html)" in source
    assert 'data-sdl-tip="' in source
    assert "self._review_loading_minimum_ms = 10" in source
    assert "_start_tooltip_translation" in source
    assert "_translate_single_row_tooltip" in source
    assert "_machine_translation_result_note" in source
    assert "_append_machine_translation_note" in source
    assert "fallback_note" in source
    assert "fallback_failed_endpoints" in source
    assert "_tooltip_translation_status = Signal(int, object, str)" in source
    assert "_tooltip_translation_status.connect(self._apply_tooltip_translation_status)" in source
    assert "endpoint_status_callback=status_callback" in source
    assert "_apply_tooltip_translation_status" in source
    assert "_compact_machine_translation_error" in source
    assert "setSelectionMode(QAbstractItemView.ExtendedSelection)" in source
    assert "setContextMenuPolicy(Qt.NoContextMenu)" in source
    assert "self.piece_list.viewport().installEventFilter(self)" in source
    assert "customContextMenuRequested.connect(self._translate_piece_list_context_selection)" not in source
    assert "event_type == QEvent.MouseButtonPress and event.button() == Qt.RightButton" in source
    assert "QMenu(self)" in source
    assert "QMenu::item { padding: 6px 18px 6px 12px; }" in source
    assert "Generate Machine Translation Preview ({entry_count} entries)" in source
    assert "menu.popup(self.piece_list.viewport().mapToGlobal(pos))" in source
    assert "QKeySequence, QShortcut" in source
    assert "MANUAL_REFRESH_BUTTON_TEXT" in source
    assert "FLAG_ACCURACY_BUTTON_TEXT" in source
    assert "MACHINE_TRANSLATION_THRESHOLD_CONFIG_KEY" in source
    assert "TWO_COLUMN_LAYOUT_BUTTON_TEXT" in source
    assert "TWO_COLUMN_LAYOUT_CONFIG_KEY" in source
    assert 'TWO_COLUMN_LAYOUT_BUTTON_TEXT = "Compact"' in source
    assert "LEGACY_ONE_COLUMN_LAYOUT_CONFIG_KEY" in source
    assert "LEGACY_ONE_ROW_LAYOUT_CONFIG_KEY" in source
    assert "self.flag_accuracy_btn = QPushButton(self.FLAG_ACCURACY_BUTTON_TEXT)" in source
    assert "self.flag_accuracy_btn.setContextMenuPolicy(Qt.CustomContextMenu)" in source
    assert "self.flag_accuracy_btn.customContextMenuRequested.connect(self._show_flag_accuracy_context_menu)" in source
    assert "self.flag_accuracy_btn.clicked.connect(self._flag_current_piece_inaccurate_translations)" in source
    assert "self.two_column_layout_btn = QPushButton(self.TWO_COLUMN_LAYOUT_BUTTON_TEXT)" in source
    assert "self.two_column_layout_btn.setCheckable(True)" in source
    assert "self.two_column_layout_btn.toggled.connect(self._set_review_two_column_layout)" in source
    assert 'legend_row.addWidget(self.two_column_layout_btn, 0, Qt.AlignVCenter)' in source
    assert "SdlReviewTwoColumnText" in source
    assert "SdlReviewTwoColumnControls" in source
    assert "SdlReviewTwoColumnMetrics" in source
    assert "Undo All Edits" in source
    assert "Generate Preview" in source
    assert "Inject Machine Translation" in source
    assert "return True" in source[source.index("def _review_two_column_layout_enabled"):source.index("def _set_review_two_column_layout")]
    assert "_promote_inaccurate_machine_translation_rows" in source
    assert "_show_flag_accuracy_context_menu" in source
    assert "_prompt_machine_translation_threshold" in source
    assert "_set_machine_translation_inaccuracy_threshold" in source
    assert "_machine_translation_inaccuracy_threshold()" in source
    assert "QInputDialog.getDouble" in source
    assert "config.json" in source
    assert "MACHINE_TRANSLATION_INACCURACY_THRESHOLD" in source
    assert "purple MT inaccurate" in source
    assert "left = source   right = output   bar width ~= length" not in source
    assert "self.refresh_review_btn = QPushButton(self.MANUAL_REFRESH_BUTTON_TEXT)" in source
    assert "self.refresh_review_btn.clicked.connect(self._manual_review_refresh)" in source
    assert "_start_refresh_button_animation" in source
    assert "_tick_refresh_button_animation" in source
    assert "_stop_refresh_button_animation" in source
    assert 'self.refresh_review_btn.setText(f"{frames[self._refresh_button_frame]} Refreshing")' in source
    assert "_start_flag_accuracy_button_animation" in source
    assert "_tick_flag_accuracy_button_animation" in source
    assert "_stop_flag_accuracy_button_animation" in source
    assert "_queue_stop_flag_accuracy_button_animation" in source
    assert 'self.flag_accuracy_btn.setText(f"{frames[self._flag_accuracy_button_frame]} Flagging")' in source


def test_sdlxliff_machine_translation_api_keys_are_encrypted_and_decrypted():
    from api_key_encryption import get_handler

    source = (SRC / "Retranslation_GUI.py").read_text(encoding="utf-8")
    handler = get_handler()
    key_fields = [
        SDLXLIFFReviewDialog.MACHINE_TRANSLATION_DEEPL_API_KEY_CONFIG_KEY,
        SDLXLIFFReviewDialog.MACHINE_TRANSLATION_BING_API_KEY_CONFIG_KEY,
        SDLXLIFFReviewDialog.MACHINE_TRANSLATION_YANDEX_API_KEY_CONFIG_KEY,
    ]

    assert set(key_fields).issubset(set(getattr(handler, "api_key_fields", [])))

    raw_config = {field: f"{field}-secret" for field in key_fields}
    encrypted_config = handler.encrypt_config(raw_config)
    for field in key_fields:
        assert encrypted_config[field].startswith("ENC:")

    assert handler.decrypt_config(encrypted_config) == raw_config

    dialog = SDLXLIFFReviewDialog.__new__(SDLXLIFFReviewDialog)
    dialog._config = {
        SDLXLIFFReviewDialog.MACHINE_TRANSLATION_DEEPL_API_KEY_CONFIG_KEY: encrypted_config[
            SDLXLIFFReviewDialog.MACHINE_TRANSLATION_DEEPL_API_KEY_CONFIG_KEY
        ],
        SDLXLIFFReviewDialog.MACHINE_TRANSLATION_BING_API_KEY_CONFIG_KEY: encrypted_config[
            SDLXLIFFReviewDialog.MACHINE_TRANSLATION_BING_API_KEY_CONFIG_KEY
        ],
        SDLXLIFFReviewDialog.MACHINE_TRANSLATION_BING_REGION_CONFIG_KEY: "eastus",
        SDLXLIFFReviewDialog.MACHINE_TRANSLATION_YANDEX_API_KEY_CONFIG_KEY: encrypted_config[
            SDLXLIFFReviewDialog.MACHINE_TRANSLATION_YANDEX_API_KEY_CONFIG_KEY
        ],
        SDLXLIFFReviewDialog.MACHINE_TRANSLATION_YANDEX_FOLDER_ID_CONFIG_KEY: "folder-123",
    }

    options = dialog._machine_translation_api_options()
    assert options["deepl"]["api_key"] == raw_config[SDLXLIFFReviewDialog.MACHINE_TRANSLATION_DEEPL_API_KEY_CONFIG_KEY]
    assert options["bing"]["api_key"] == raw_config[SDLXLIFFReviewDialog.MACHINE_TRANSLATION_BING_API_KEY_CONFIG_KEY]
    assert options["bing"]["region"] == "eastus"
    assert options["yandex"]["api_key"] == raw_config[SDLXLIFFReviewDialog.MACHINE_TRANSLATION_YANDEX_API_KEY_CONFIG_KEY]
    assert options["yandex"]["folder_id"] == "folder-123"

    parent_config = {}
    dialog._config = {}
    dialog._context_parent = SimpleNamespace(config=parent_config, save_config=lambda show_message=False: True)
    assert dialog._persist_review_config_value(
        SDLXLIFFReviewDialog.MACHINE_TRANSLATION_DEEPL_API_KEY_CONFIG_KEY,
        "fresh-deepl-key",
    )
    stored = dialog._config[SDLXLIFFReviewDialog.MACHINE_TRANSLATION_DEEPL_API_KEY_CONFIG_KEY]
    assert stored.startswith("ENC:")
    assert parent_config[SDLXLIFFReviewDialog.MACHINE_TRANSLATION_DEEPL_API_KEY_CONFIG_KEY] == stored
    assert handler.decrypt_value(stored) == "fresh-deepl-key"
    assert 'QShortcut(QKeySequence("F5"), self)' in source
    assert "self._manual_refresh_shortcut.activated.connect(self._manual_review_refresh)" in source
    constructor_body = source[source.index("def __init__"):source.index("self.setWindowTitle", source.index("def __init__"))]
    assert "self._last_review_signature = self._current_review_signature()" not in constructor_body
    assert "self._last_machine_translation_signature = self._current_machine_translation_signature()" not in constructor_body
    manual_refresh_body = source[
        source.index("def _manual_review_refresh"):
        source.index("def _silent_review_refresh", source.index("def _manual_review_refresh"))
    ]
    assert "self._queue_review_refresh_scan(" in manual_refresh_body
    assert "force=True" in manual_refresh_body
    assert "validate=False" in manual_refresh_body
    assert "self._silent_review_refresh()" not in manual_refresh_body
    flag_accuracy_body = source[
        source.index("def _flag_current_piece_inaccurate_translations"):
        source.index("@staticmethod", source.index("def _flag_current_piece_inaccurate_translations"))
    ]
    assert "self._start_flag_accuracy_button_animation()" in flag_accuracy_body
    assert "self._queue_stop_flag_accuracy_button_animation()" in flag_accuracy_body
    assert "_review_context_menu_open" in source
    assert "_set_review_context_menu_open(True)" in source
    assert "_pause_review_preload_for_context_menu" in source
    assert "_resume_review_background_after_context_menu" in source
    context_start = source.index("def _translate_piece_list_context_selection")
    context_end = source.index("def _clear_piece_list_context_menu", context_start)
    context_body = source[context_start:context_end]
    assert "setCurrentRow(clicked_row)" not in context_body
    assert "blockSignals(True)" in context_body
    assert "_set_review_context_menu_open(True)" in context_body
    text_menu_start = source.index("def _show_review_text_context_menu")
    text_menu_end = source.index("def _clear_review_text_context_menu", text_menu_start)
    text_menu_body = source[text_menu_start:text_menu_end]
    assert "menu.exec(" not in text_menu_body
    assert "menu.popup(anchor_widget.mapToGlobal(anchor_pos))" in text_menu_body
    assert "popup_widget=None" in text_menu_body
    assert "popup_pos=None" in text_menu_body
    assert "self._review_text_context_menu = menu" in text_menu_body
    assert "_set_review_context_menu_open(True)" in text_menu_body
    assert "def _wire_source_preview_context_menu" in source
    assert 'container.setObjectName("SdlReviewSourceText")' in source
    assert 'label.setObjectName("SdlReviewSourceRawText")' in source
    silent_refresh_body = source[
        source.index("def _silent_review_refresh"):
        source.index("def refresh_review_data", source.index("def _silent_review_refresh"))
    ]
    assert "if self._review_context_menu_is_open():" in silent_refresh_body
    assert "self._queue_review_refresh_scan(" in silent_refresh_body
    assert "_current_review_signature" not in silent_refresh_body
    assert "_current_machine_translation_signature" not in silent_refresh_body
    assert "_current_review_autogen_signature" not in silent_refresh_body
    assert "refresh_review_data(" not in silent_refresh_body
    preload_queue_body = source[
        source.index("def _queue_review_page_preloads"):
        source.index("def _start_next_review_preload", source.index("def _queue_review_page_preloads"))
    ]
    assert "if self._review_context_menu_is_open():" in preload_queue_body
    preload_batch_body = source[
        source.index("def _run_review_preload_batch"):
        source.index("def _request_render_piece", source.index("def _run_review_preload_batch"))
    ]
    assert "if self._review_context_menu_is_open():" in preload_batch_body
    dirty_refresh_body = source[
        source.index("def _refresh_current_visible_dirty_source_previews"):
        source.index("def _refresh_visible_review_row_source_previews", source.index("def _refresh_current_visible_dirty_source_previews"))
    ]
    assert "if self._review_context_menu_is_open():" in dirty_refresh_body
    assert "self.piece_list.selectAll()" in source
    assert "_translate_piece_rows_tooltips" in source
    assert "_start_piece_list_tooltip_translation" in source
    assert "self.piece_list.setUniformItemSizes(True)" in source
    assert "self.piece_list.setMinimumWidth(242)" in source
    assert "self.piece_list.setMaximumWidth(286)" in source
    assert "self.piece_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)" in source
    assert "def _output_dir_has_sdlxliff_sidecars" in source
    assert "_queue_review_refresh_scan(force=False, current_path=self.current_path, delay_ms=25)" in source
    show_text_analysis_body = source[
        source.index("def _show_text_analysis"):
        source.index("text_analysis_btn.clicked.connect", source.index("def _show_text_analysis"))
    ]
    assert "_open_or_reuse_sdlxliff_review(" in show_text_analysis_body
    assert "_generate_sdlxliff_sidecars_from_completed_entries(" not in show_text_analysis_body
    item_review_body = source[
        source.index("def _open_sdlxliff_review_for_item"):
        source.index("def _open_file_for_item", source.index("def _open_sdlxliff_review_for_item"))
    ]
    assert "_open_or_reuse_sdlxliff_review(" in item_review_body
    assert "_generate_sdlxliff_sidecars_from_completed_entries(" not in item_review_body
    assert "_review_generation_progress = Signal(object)" in source
    assert "progress_callback=self._emit_review_generation_progress" in source
    assert "QProgressBar" in source
    assert "self.generation_progress_bar = QProgressBar()" in source
    assert "self.loading_progress_bar = loading_progress" in source
    assert "def _set_generation_progress" in source
    assert "def _set_loading_progress" in source
    assert "self._set_loading_progress(" in source
    assert "def _queue_generated_sidecar_stream_piece" in source
    assert "def _flush_generated_sidecar_stream_pieces" in source
    assert "preserve_generated_stream" in source
    assert 'elif stage == "created":' in source
    created_stage_body = source[
        source.index('elif stage == "created":'):
        source.index('elif stage in {"checking", "missing_source"', source.index('elif stage == "created":'))
    ]
    assert "_queue_generated_sidecar_stream_piece" in created_stage_body
    assert "_append_generated_sidecar_stream_piece" not in created_stage_body
    assert "spine_positions = self._read_spine_positions(allow_deep_search=not stream_sidebar)" in source
    assert "REVIEW_PRELOAD_RADIUS = 2" in source
    assert "REVIEW_PRELOAD_BATCH_SIZE = 8" in source
    assert "REVIEW_PRELOAD_IDLE_MS = 350" in source
    assert "REVIEW_MAX_CACHED_PAGES = 7" in source
    assert "REVIEW_SYNC_RENDER_ROW_LIMIT = 80" in source
    assert "wait(pending, timeout=0.025, return_when=FIRST_COMPLETED)" in source
    assert "_pump_review_loading_events" in source
    pump_body = source[
        source.index("def _pump_review_loading_events"):
        source.index("def _remove_review_page_widget", source.index("def _pump_review_loading_events"))
    ]
    assert "_review_event_pump_active" in pump_body
    assert "finally:" in pump_body
    generation_prepare_body = source[
        source.index("def _prepare_generation_streaming_piece_list"):
        source.index("def _append_generated_sidecar_stream_piece", source.index("def _prepare_generation_streaming_piece_list"))
    ]
    assert "_pump_review_loading_events" not in generation_prepare_body
    generation_stream_body = source[
        source.index("def _append_generated_sidecar_stream_piece"):
        source.index("def _review_generation_summary", source.index("def _append_generated_sidecar_stream_piece"))
    ]
    assert "_pump_review_loading_events" not in generation_stream_body
    assert "self.pieces = self._load_pieces(stream_sidebar=not seamless)" in source
    assert "if not self._streamed_piece_list_populated:" in source
    assert "def _prepare_streaming_piece_list" in source
    assert "def _stream_piece_list_item" in source
    assert "def _finish_streaming_piece_list" in source
    assert "def _prepare_generation_streaming_piece_list" in source
    assert "def _append_generated_sidecar_stream_piece" in source
    assert "_prepare_streaming_piece_list(work_items)" in source
    assert "if stream_sidebar:" in source
    assert "for idx, (path, metadata) in enumerate(work_items):" in source
    assert "_stream_piece_list_item(next_stream_index, pieces[next_stream_index])" in source
    assert "_finish_streaming_piece_list()" in source
    assert "_append_generated_sidecar_stream_piece" in source
    assert 'QListWidgetItem(f"{label} loading...")' not in source
    assert "flush_streamed_pieces(limit=12)" in source
    assert "len(rows) <= self.REVIEW_SYNC_RENDER_ROW_LIMIT" in source
    assert "Could not prepare SDLXLIFF review rows" in source
    assert "_last_review_selection_change = time.monotonic()" in source
    assert "_review_selection_recently_changed" in source
    assert "_queue_review_page_cache_trim" in source
    assert "_trim_review_page_cache" in source
    assert "_review_data_preload_finished = Signal(int, object)" in source
    assert "_review_data_preload_finished.connect(self._apply_review_data_preload)" in source
    assert "_review_refresh_scan_finished = Signal(int, object)" in source
    assert "_review_refresh_scan_finished.connect(self._apply_review_refresh_scan)" in source
    assert "_review_piece_reload_finished = Signal(int, object)" in source
    assert "_review_piece_reload_finished.connect(self._apply_async_review_piece_reload)" in source
    assert "def _queue_review_refresh_scan" in source
    assert "def _start_review_refresh_scan" in source
    assert "def _build_review_refresh_scan_result" in source
    assert "def _regenerate_review_sidecars_for_refresh_scan" in source
    assert "def _apply_review_refresh_scan" in source
    assert "def _queue_async_review_piece_reload" in source
    assert "def _apply_async_review_piece_reload" in source
    assert "def _changed_review_signature_paths" in source
    assert "def _review_signature_path_set_changed" in source
    assert "_review_refresh_scan_validate" in source
    assert "validate=validate" in source
    assert "_review_refresh_scan_running" in source
    assert "_review_refresh_scan_requested" in source
    assert "_review_piece_reload_running" in source
    assert "_review_piece_reload_requested" in source
    assert 'name="sdlxliff-review-refresh-scan"' in source
    assert 'name="sdlxliff-review-piece-reload"' in source
    apply_scan_body = source[
        source.index("def _apply_review_refresh_scan"):
        source.index("def _current_review_signature", source.index("def _apply_review_refresh_scan"))
    ]
    assert 'initial_load = not bool(getattr(self, "_review_data_loaded", False) and self.pieces)' in apply_scan_body
    assert 'force_full_reload = bool(result.get("sidecar_path_set_changed") or result.get("settings_changed"))' in apply_scan_body
    assert 'changed_paths = None if force_full_reload else result.get("changed_sidecar_paths")' in apply_scan_body
    assert "_queue_async_review_piece_reload(" in apply_scan_body
    assert "changed_paths=changed_paths" in apply_scan_body
    assert "defer_stop_refresh_animation" in apply_scan_body
    assert "seamless=not initial_load" in apply_scan_body
    piece_reload_body = source[
        source.index("def _queue_async_review_piece_reload"):
        source.index("def _apply_async_review_piece_reload", source.index("def _queue_async_review_piece_reload"))
    ]
    assert "threading.Thread" in piece_reload_body
    assert "self._load_pieces(stream_sidebar=False)" in piece_reload_body
    assert "pieces_by_path" in piece_reload_body
    assert "changed_path_set" in piece_reload_body
    assert "_pump_review_loading_events" not in piece_reload_body
    piece_reload_apply_body = source[
        source.index("def _apply_async_review_piece_reload"):
        source.index("def _apply_review_refresh_scan", source.index("def _apply_async_review_piece_reload"))
    ]
    assert 'if result.get("partial"):' in piece_reload_apply_body
    assert "self._refresh_piece_list_item(row)" in piece_reload_apply_body
    assert "self._populate_piece_list()" in piece_reload_apply_body
    assert "Loaded SDLXLIFF entry" in source
    refresh_signature = source[
        source.index("def refresh_review_data"):
        source.index("if self._refreshing_review_data:", source.index("def refresh_review_data"))
    ]
    assert "skip_autogen=False" in refresh_signature
    assert "autogen_signature=None" in refresh_signature
    assert "mt_signature=None" in refresh_signature
    refresh_body = source[
        source.index("def refresh_review_data"):
        source.index("def reopen_for_path", source.index("def refresh_review_data"))
    ]
    assert "False if skip_autogen else self._maybe_regenerate_review_sidecars" in refresh_body
    assert "self._start_review_data_preload()" in source
    assert "_build_review_piece_render_model_from_rows" in source
    assert "_review_piece_render_model" in source
    assert "_piece_render_snapshot" in source
    assert "name=\"sdlxliff-review-data-preload\"" in source
    assert "row_model=None" in source
    assert "row_model=row_model" in source
    data_preload_body = source[
        source.index("def _start_review_data_preload"):
        source.index("def _apply_review_data_preload", source.index("def _start_review_data_preload"))
    ]
    assert "threading.Thread" in data_preload_body
    assert "_build_review_piece_render_model_from_rows" in data_preload_body
    assert "time.sleep(0.001)" in data_preload_body
    assert "QLabel(" not in data_preload_body
    assert "QFrame(" not in data_preload_body
    assert "QPlainTextEdit(" not in data_preload_body
    preload_order_body = source[
        source.index("def _review_preload_order"):
        source.index("def _queue_review_page_preloads", source.index("def _review_preload_order"))
    ]
    assert "range(1, self.REVIEW_PRELOAD_RADIUS + 1)" in preload_order_body
    assert "range(len(self.pieces))" not in preload_order_body
    assert "current_row + distance" in preload_order_body
    assert "current_row - distance" in preload_order_body
    assert "self._review_selection_recently_changed()" in preload_queue_body
    assert "self._queue_review_page_cache_trim(current_row)" in preload_queue_body
    assert "self.REVIEW_PRELOAD_BATCH_SIZE" in preload_batch_body
    assert "self.REVIEW_PRELOAD_STEP_MS" in preload_batch_body
    assert "self._review_selection_recently_changed()" in preload_batch_body
    assert "_heading_paragraph_tag_changed" in source
    assert "heading/paragraph tag changed" in source
    align_body = source[
        source.index("def _align_review_units"):
        source.index("def _review_piece_non_empty_count", source.index("def _align_review_units"))
    ]
    assert align_body.index("source_remaining = len(source_units) - i") < align_body.index("if self._review_units_are_compatible(src, tgt):")
    assert "source_remaining > target_remaining" in align_body
    assert "target_remaining > source_remaining" in align_body
    assert "refresh=piece_index == current_row" in source
    assert "current_row = self._displayed_piece_row()" in source
    assert "if row == self._displayed_piece_row()" in source
    assert "_refresh_visible_review_row_source_preview" in source
    assert "_patch_review_row_machine_translation_preview" in source
    assert "_update_review_row_source_previews" in source
    assert "_source_preview_dirty" in source
    assert "_queue_refresh_current_visible_dirty_source_previews" in source
    assert "piece_index = self._displayed_piece_row()" in source
    assert "QTimer.singleShot(0, self._queue_refresh_current_visible_dirty_source_previews)" in source
    assert "_current_machine_translation_signature" in source
    assert "_reload_machine_translation_previews" in source
    assert "if self._tooltip_translation_running:" in source
    assert 'elif result.get("machine_translation_changed"):' in source
    assert "self.refresh_review_data(" in source
    assert "_write_machine_translation_entries" in source
    assert "persist=False" in source
    render_start = source.index("def _render_piece")
    render_end = source.index("\n\nclass RetranslationMixin", render_start)
    render_body = source[render_start:render_end]
    assert "while row_state[\"idx\"] < len(rows)" not in render_body
    assert "batch_size = 12" in render_body
    assert "render_timer.start(1)" in render_body
    assert "if show_loading:\n            self._show_review_loading_page()" not in render_body
    assert "if show_loading:\n            self.rows_stack.setCurrentWidget(page)" in render_body
    assert "self._finish_rows_rebuild(final=False)" in render_body
    assert "self._refresh_review_stream_geometry(final=False)" in render_body
    assert "self.rows_stack, self.scroll.viewport(), self.scroll" in render_body
    assert "widget.setUpdatesEnabled(False)" in render_body
    assert "widget.setUpdatesEnabled(True)" in render_body
    assert "widget.update()" in render_body
    assert "updates_enabled=False" in render_body
    assert "batch_frames.append(frame)" in render_body
    sync_body = source[
        source.index("def _sync_review_scroll_range"):
        source.index("def _save_current_review_scroll", source.index("def _sync_review_scroll_range"))
    ]
    assert "page.setMinimumHeight(content_height)" in sync_body
    assert "page.setMaximumHeight(content_height)" in sync_body
    assert "page.resize(max(1, page.width()), content_height)" in sync_body
    assert "QApplication.processEvents(QEventLoop.AllEvents, 10)" not in render_body
    assert "self._update_review_row_source_previews(piece_index, pending_rows, visible_only=True)" in source
    assert "self._update_review_row_source_previews(row, changed_rows, visible_only=True)" in source
    assert "Machine Translation \\u2192" in source
    assert "_open_google_translate" not in source
    assert "Translate tooltip" not in source
    assert "Retranslate tooltip" not in source
    assert "Inject Machine Translation" in source
    assert "inject_machine_translation_callback" in source
    assert "tooltip_translation_pending" in source
    assert "_machine_translation_pending_text" in source
    assert "Translating with {self._machine_translation_provider_label()}" in source
    assert "SdlReviewSourceText" in source
    assert "SdlReviewMachineTranslation" in source
    assert "SdlReviewMachineTranslationPending" in source
    assert "QListWidget#SdlReviewPieceList::item:hover:!selected" in source
    selected_style = source[
        source.index("QListWidget#SdlReviewPieceList::item:selected {{"):
        source.index("QListWidget#SdlReviewPieceList::item:selected:hover {{")
    ]
    assert not re.search(r"(?m)^\s*color\s*:", selected_style)
    assert "border: 1px dashed #8a6f2a" in source
    assert "padding: 5px 8px; font-size: 8pt" in source
    assert "border-left: 3px solid #5aa7d8" in source
    assert "background: rgba(23, 37, 54, 185)" in source
    assert "font-size: 7pt" in source
    assert "Google tooltip:" not in source
    assert "Copy translated tooltip" not in source
    assert "Inject tooltip translation into output" not in source
    apply_start = source.index("def _apply_tooltip_translations")
    apply_end = source.index("def _selected_text_for_widget")
    apply_body = source[apply_start:apply_end]
    assert "_update_review_row_source_previews" in apply_body
    assert "_refresh_visible_review_row_source_previews" not in apply_body
    assert "_discard_piece_page" not in apply_body
    assert "_render_piece" not in apply_body
    assert "message = f\"Generated {len(translations)} {provider_label} machine translation preview(s)\"" in apply_body
    assert "if error:" in apply_body
    assert "message = f\"{message}. {self._compact_machine_translation_error(error)}\"" in apply_body
    assert "row_data[\"tooltip_translation_error\"] = self._compact_machine_translation_error(error)" in apply_body
    assert "row_data[\"tooltip_translation_error_detail\"] = str(error or \"\")" in apply_body
    batch_start = source.index("def _start_piece_list_tooltip_translation")
    batch_end = source.index("def _translate_piece_rows_tooltips", batch_start)
    batch_body = source[batch_start:batch_end]
    assert "_discard_piece_page" not in batch_body
    assert "result_note = self._machine_translation_result_note(result)" in batch_body
    assert "error = self._append_machine_translation_note(error, result_note)" in batch_body
    assert "status_context[\"piece_index\"] = piece_index" in batch_body


def test_sdlxliff_review_compacts_machine_translation_endpoint_errors():
    google_error = (
        "All Google Translate endpoints failed:\n"
        "  • https://translate.google.co.in/translate_a/single: HTTP 429: Rate Limited (too many requests)\n"
        "  • https://translate.google.com/translate_a/single: HTTP 429: Rate Limited (too many requests)\n"
    )
    assert SDLXLIFFReviewDialog._compact_machine_translation_error(google_error) == (
        "Google failed: HTTP 429 Rate Limited (too many requests) on 2 endpoints"
    )
    assert SDLXLIFFReviewDialog._compact_machine_translation_error(
        "Auto fell back after Google endpoints failed: a, b, c"
    ) == "Auto fell back after Google failed on 3 endpoints"
    assert SDLXLIFFReviewDialog._compact_machine_translation_error(
        "Auto fell back to DeepL after Google endpoints failed: a, b"
    ) == "Auto fell back to DeepL after Google failed on 2 endpoints"


def test_sdlxliff_review_tooltip_batch_wraps_and_parses_by_html_tag():
    dialog = SDLXLIFFReviewDialog.__new__(SDLXLIFFReviewDialog)
    work = [
        (0, ("piece", 0, "Title"), "10년 후가 두렵다", "h1"),
        (1, ("piece", 1, "Body"), "안녕하세요 독자님들.", "p"),
    ]

    batch_html = dialog._tooltip_batch_html(work)
    translations = dialog._extract_tooltip_batch_translations(
        '<h1 data-sdl-tip="0">I fear ten years later</h1>'
        '<p data-sdl-tip="1">Hello, readers.</p>',
        work,
    )

    assert '<h1 data-sdl-tip="0">10년 후가 두렵다</h1>' in batch_html
    assert '<p data-sdl-tip="1">안녕하세요 독자님들.</p>' in batch_html
    assert translations == {
        ("piece", 0, "Title"): "I fear ten years later",
        ("piece", 1, "Body"): "Hello, readers.",
    }
    assert dialog._review_row_height("안녕하세요 독자님들.", "Hello, readers.", "Hello, readers.") >= (
        dialog.REVIEW_ROW_MIN_HEIGHT + 30
    )


def test_sdlxliff_review_machine_translation_injection_preserves_editor_undo(qtbot):
    dialog = SDLXLIFFReviewDialog.__new__(SDLXLIFFReviewDialog)
    editor = QPlainTextEdit("Original output")
    qtbot.addWidget(editor)

    dialog._inject_machine_translation_to_target(0, 0, "Machine output", editor)

    assert editor.toPlainText() == "Machine output"
    editor.undo()
    assert editor.toPlainText() == "Original output"


def test_sdlxliff_machine_translation_path_uses_machine_translation_subfolder(tmp_path):
    sidecar = tmp_path / "SDLXLIFF" / "response_piece_0002.html.sdlxliff"

    assert _sdlxliff_machine_translation_path(str(tmp_path), "response_piece_0002.html") == str(
        tmp_path / "SDLXLIFF" / "Machine_Translation" / "response_piece_0002.html.json"
    )
    assert _sdlxliff_machine_translation_path("", str(sidecar)) == str(
        tmp_path / "SDLXLIFF" / "Machine_Translation" / "response_piece_0002.html.json"
    )


def test_sdlxliff_review_signature_tracks_machine_translation_deletions(tmp_path):
    sdl_dir = tmp_path / "SDLXLIFF"
    mt_dir = sdl_dir / "Machine_Translation"
    mt_dir.mkdir(parents=True)
    (sdl_dir / "response_piece_0002.html.sdlxliff").write_text("sidecar", encoding="utf-8")
    preview_path = mt_dir / "response_piece_0002.html.json"
    preview_path.write_text('{"entries": {}}', encoding="utf-8")

    dialog = SDLXLIFFReviewDialog.__new__(SDLXLIFFReviewDialog)
    dialog.output_dir = str(tmp_path)
    dialog._book_entries = []

    sidecar_signature = dialog._current_review_signature()
    with_preview = dialog._current_machine_translation_signature()
    preview_path.unlink()
    without_preview = dialog._current_machine_translation_signature()
    mt_dir.rmdir()
    without_folder = dialog._current_machine_translation_signature()

    assert dialog._current_review_signature() == sidecar_signature
    assert with_preview != without_preview
    assert without_preview != without_folder
    assert any(entry[0] == "machine_translation_dir" and entry[2] == -1 for entry in without_folder)


def test_sdlxliff_review_persists_and_reloads_machine_translation_preview(tmp_path):
    output_name = "response_chapter0001.html"
    _write_html_sdlxliff_sidecar(
        str(tmp_path),
        output_name,
        {"original_basename": "chapter0001.xhtml"},
        "<p>Source sentence.</p>",
        "<p>Translated sentence.</p>",
    )
    sidecar = tmp_path / "SDLXLIFF" / f"{output_name}.sdlxliff"
    dialog = SDLXLIFFReviewDialog.__new__(SDLXLIFFReviewDialog)
    dialog.output_dir = str(tmp_path)
    dialog._config = {"output_language": "English"}
    piece = dialog._build_piece(str(sidecar), 0, {"output_name": output_name})

    dialog._set_row_tooltip_translation(piece, piece["rows"][0], "Machine preview sentence.")

    preview_path = tmp_path / "SDLXLIFF" / "Machine_Translation" / f"{output_name}.json"
    assert preview_path.is_file()
    stored_preview = json.loads(preview_path.read_text(encoding="utf-8"))
    assert list(stored_preview["entries"].values())[0]["translation"] == "Machine preview sentence."

    new_dialog = SDLXLIFFReviewDialog.__new__(SDLXLIFFReviewDialog)
    new_dialog.output_dir = str(tmp_path)
    new_dialog._config = {"output_language": "English"}
    reloaded = new_dialog._build_piece(str(sidecar), 0, {"output_name": output_name})

    assert reloaded["rows"][0]["tooltip_translation"] == "Machine preview sentence."
    assert new_dialog._row_tooltip_translation(reloaded, reloaded["rows"][0]) == "Machine preview sentence."


def test_sdlxliff_review_generate_preview_includes_stored_first_row():
    dialog = SDLXLIFFReviewDialog.__new__(SDLXLIFFReviewDialog)
    dialog._tooltip_translation_running = False
    dialog._config = {"output_language": "English"}
    piece = {
        "path": "piece.sdlxliff",
        "rows": [
            {"row_index": 0, "source_index": 0, "source_tag": "p", "source": "Already stored."},
            {"row_index": 1, "source_index": 1, "source_tag": "p", "source": "Needs preview."},
        ],
    }
    dialog.pieces = [piece]
    dialog._set_row_tooltip_translation(piece, piece["rows"][0], "Cached preview.")
    captured = {}
    dialog._current_piece_row = lambda: 0
    dialog._start_tooltip_translation = lambda row, work, ready_text="": captured.update(
        {"row": row, "work": work, "ready_text": ready_text}
    )

    dialog._translate_current_piece_tooltips()

    assert captured["row"] == 0
    assert [item[0] for item in captured["work"]] == [0, 1]
    assert captured["ready_text"] == "Preview Ready"


def test_sdlxliff_review_selected_piece_preview_jobs_include_row_zero():
    dialog = SDLXLIFFReviewDialog.__new__(SDLXLIFFReviewDialog)
    dialog._tooltip_translation_running = False
    dialog._config = {"output_language": "English"}
    dialog.pieces = [
        {
            "path": "piece0.sdlxliff",
            "rows": [
                {"row_index": 0, "source_index": 0, "source_tag": "p", "source": "First piece row zero."},
            ],
        },
        {
            "path": "piece1.sdlxliff",
            "rows": [
                {"row_index": 0, "source_index": 0, "source_tag": "p", "source": "Second piece row zero."},
                {"row_index": 1, "source_index": 1, "source_tag": "p", "source": "Second piece row one."},
            ],
        },
    ]
    captured = {}
    dialog._start_piece_list_tooltip_translation = lambda jobs: captured.update({"jobs": jobs})

    dialog._translate_piece_rows_tooltips([1, 0, 1])

    assert [piece_row for piece_row, _work in captured["jobs"]] == [0, 1]
    assert [item[0] for item in captured["jobs"][0][1]] == [0]
    assert [item[0] for item in captured["jobs"][1][1]] == [0, 1]


def test_sdlxliff_review_source_preview_marks_identical_machine_translation(qtbot):
    dialog = SDLXLIFFReviewDialog.__new__(SDLXLIFFReviewDialog)
    widget = dialog._text_label(
        "Synthetic title row",
        tooltip_translation="Synthetic title row",
    )
    qtbot.addWidget(widget)

    labels = widget.findChildren(QLabel)
    assert any(label.text() == "Synthetic title row" for label in labels)


def test_sdlxliff_review_translator_note_uses_note_placeholder(qtbot):
    dialog = SDLXLIFFReviewDialog.__new__(SDLXLIFFReviewDialog)
    label = dialog._text_label(
        "",
        missing=False,
        empty_placeholder="[Translator Note]",
    )
    qtbot.addWidget(label)

    assert label.text() == "[Translator Note]"


def test_sdlxliff_review_rejects_untranslated_google_preview_batch():
    dialog = SDLXLIFFReviewDialog.__new__(SDLXLIFFReviewDialog)
    work = [
        (0, "row-0", "Synthetic heading input", "h1"),
        (1, "row-1", "Synthetic paragraph input with enough words for extraction.", "p"),
    ]
    raw_html = dialog._tooltip_batch_html(work)

    parsed = dialog._extract_tooltip_batch_translations(raw_html, work)
    valid, error = dialog._validate_tooltip_batch_translations(parsed, work)

    assert parsed
    assert valid == {}
    assert "returned source text unchanged" in error
    assert "refusing to save raw source" in error


def test_sdlxliff_review_does_not_persist_identical_machine_translation():
    dialog = SDLXLIFFReviewDialog.__new__(SDLXLIFFReviewDialog)
    piece = {"rows": []}
    row = {"source": "Synthetic paragraph input with enough words for extraction."}

    dialog._set_row_tooltip_translation(piece, row, "Synthetic paragraph input with enough words for extraction.")

    assert "tooltip_translation" not in row


def test_sdlxliff_review_row_index_property_preserves_zero(qtbot):
    frame = QFrame()
    qtbot.addWidget(frame)
    frame.setProperty("sdl_row_index", 0)

    assert SDLXLIFFReviewDialog._review_row_index_property(frame) == 0


def test_sdlxliff_review_machine_translation_ignores_changed_source_or_language(tmp_path):
    output_name = "response_chapter0001.html"
    _write_html_sdlxliff_sidecar(
        str(tmp_path),
        output_name,
        {"original_basename": "chapter0001.xhtml"},
        "<p>Original source.</p>",
        "<p>Translated sentence.</p>",
    )
    sidecar = tmp_path / "SDLXLIFF" / f"{output_name}.sdlxliff"
    dialog = SDLXLIFFReviewDialog.__new__(SDLXLIFFReviewDialog)
    dialog.output_dir = str(tmp_path)
    dialog._config = {"output_language": "English"}
    piece = dialog._build_piece(str(sidecar), 0, {"output_name": output_name})
    dialog._set_row_tooltip_translation(piece, piece["rows"][0], "Cached English preview.")

    _write_html_sdlxliff_sidecar(
        str(tmp_path),
        output_name,
        {"original_basename": "chapter0001.xhtml"},
        "<p>Changed source.</p>",
        "<p>Translated sentence.</p>",
    )
    changed_source_dialog = SDLXLIFFReviewDialog.__new__(SDLXLIFFReviewDialog)
    changed_source_dialog.output_dir = str(tmp_path)
    changed_source_dialog._config = {"output_language": "English"}
    changed_source_piece = changed_source_dialog._build_piece(str(sidecar), 0, {"output_name": output_name})
    assert changed_source_piece["rows"][0].get("tooltip_translation", "") == ""

    _write_html_sdlxliff_sidecar(
        str(tmp_path),
        output_name,
        {"original_basename": "chapter0001.xhtml"},
        "<p>Original source.</p>",
        "<p>Translated sentence.</p>",
    )
    changed_language_dialog = SDLXLIFFReviewDialog.__new__(SDLXLIFFReviewDialog)
    changed_language_dialog.output_dir = str(tmp_path)
    changed_language_dialog._config = {"output_language": "Spanish"}
    changed_language_piece = changed_language_dialog._build_piece(str(sidecar), 0, {"output_name": output_name})
    assert changed_language_piece["rows"][0].get("tooltip_translation", "") == ""


def test_retranslation_cleanup_deletes_machine_translation_preview_with_sdlxliff_sidecars():
    source = (SRC / "Retranslation_GUI.py").read_text(encoding="utf-8")
    retranslate_start = source.index("def retranslate_selected")
    retranslate_body = source[retranslate_start:source.index("# Add buttons", retranslate_start)]

    assert '"Machine_Translation"' in source
    assert "_machine_translation_path_for_output_file" in retranslate_body
    assert "machine_translation_deleted_count" in retranslate_body
    assert "Deleted Machine Translation preview" in retranslate_body


def test_sdlxliff_review_summary_updates_when_target_row_is_emptied():
    dialog = SDLXLIFFReviewDialog.__new__(SDLXLIFFReviewDialog)
    rows = [
        {"source_tag": "p", "source": "Source 1", "target_tag": "p", "target": "Target 1", "status": "green"},
        {"source_tag": "p", "source": "Source 2", "target_tag": "p", "target": "", "status": "red"},
    ]
    piece = {"rows": rows}

    dialog._refresh_piece_summary(piece)

    assert piece["source_count"] == 2
    assert piece["target_count"] == 1
    assert piece["red_count"] == 1
    assert piece["yellow_count"] == 1
    assert piece["mismatch"] is True
    assert piece["count_ratio"] == 0.5
    assert rows[0]["status"] == "yellow"
    assert rows[0]["reason"].startswith("top translated-column skew")


def test_retranslation_show_model_info_defaults_on_but_respects_saved_false():
    mixin = RetranslationMixin.__new__(RetranslationMixin)
    mixin.config = {}
    mixin._retranslation_dialog_cache = {}

    assert mixin._get_retranslation_show_model_info_state() is True

    mixin.config = {mixin._RETRANSLATION_SHOW_MODEL_INFO_CONFIG_KEY: False}

    assert mixin._get_retranslation_show_model_info_state() is False


def test_dynamic_request_splitting_defaults_off():
    glossary_gui = (SRC / "GlossaryManager_GUI.py").read_text(encoding="utf-8")
    translator_gui = (SRC / "translator_gui.py").read_text(encoding="utf-8")
    async_processor = (SRC / "async_api_processor.py").read_text(encoding="utf-8")
    txt_extractor = (SRC / "extract_glossary_from_txt.py").read_text(encoding="utf-8")
    epub_extractor = (SRC / "extract_glossary_from_epub.py").read_text(encoding="utf-8")
    glossary_manager = (SRC / "GlossaryManager.py").read_text(encoding="utf-8")
    android_screen = (SRC / "android" / "extract_glossary_screen.py").read_text(encoding="utf-8")

    assert "config.get('glossary_enable_chapter_split', False)" in glossary_gui
    assert "glossary_enable_chapter_split_checkbox.setChecked(False)" in glossary_gui
    assert "'glossary_enable_chapter_split',\n            False," in translator_gui
    assert "return '1', '99', '1' if chapter_split else '0'" in translator_gui
    assert "os.environ['GLOSSARY_ENABLE_CHAPTER_SPLIT'] = _balanced_chapter_split" in translator_gui
    assert "self.gui.config.get('glossary_enable_chapter_split', False)" in async_processor
    assert 'os.getenv("GLOSSARY_ENABLE_CHAPTER_SPLIT", "0") == "1"' in txt_extractor
    assert 'os.getenv("GLOSSARY_ENABLE_CHAPTER_SPLIT", "0") == "1"' in epub_extractor
    assert 'os.getenv("GLOSSARY_ENABLE_CHAPTER_SPLIT", "0") == "1"' in glossary_manager
    assert 'cfg.get("glossary_enable_chapter_split", False)' in android_screen

    combined = "\n".join([
        glossary_gui,
        translator_gui,
        async_processor,
        txt_extractor,
        epub_extractor,
        glossary_manager,
        android_screen,
    ])
    assert "glossary_enable_chapter_split', True" not in combined
    assert 'glossary_enable_chapter_split", True' not in combined
    assert 'os.getenv("GLOSSARY_ENABLE_CHAPTER_SPLIT", "1")' not in combined


def test_retranslation_autogenerates_sdlxliff_sidecars_from_completed_entries(tmp_path, monkeypatch):
    source = tmp_path / "chapter0001.xhtml"
    output = tmp_path / "response_chapter0001.html"
    source.write_text("<h1>Source Title</h1><p>Source body.</p>", encoding="utf-8")
    output.write_text("<h1>Target Title</h1><p>Target body.</p>", encoding="utf-8")
    progress = {
        "chapters": {
            "1": {
                "actual_num": 1,
                "status": "completed",
                "output_file": output.name,
                "original_basename": source.name,
            }
        }
    }
    monkeypatch.setenv("OUTPUT_SDLXLIFF", "0")
    mixin = RetranslationMixin.__new__(RetranslationMixin)
    progress_events = []

    stats = mixin._generate_sdlxliff_sidecars_from_completed_entries(
        str(tmp_path),
        progress_data=progress,
        progress_callback=progress_events.append,
    )

    sidecar = tmp_path / "SDLXLIFF" / "response_chapter0001.html.sdlxliff"
    assert stats["created"] == 1
    assert sidecar.is_file()
    assert os.environ["OUTPUT_SDLXLIFF"] == "0"
    assert [event["stage"] for event in progress_events] == ["start", "checking", "created", "finished"]
    assert progress_events[1]["output_name"] == output.name

    dialog = SDLXLIFFReviewDialog.__new__(SDLXLIFFReviewDialog)
    source_html, target_html = dialog._read_sdlxliff_html_pair(str(sidecar))

    assert "Source Title" in source_html
    assert "Source body." in source_html
    assert "Target Title" in target_html
    assert "Target body." in target_html


def test_manual_editing_generates_untranslated_and_pending_sidecars_without_output(tmp_path, monkeypatch):
    untranslated_source = tmp_path / "chapter0001.xhtml"
    completed_source = tmp_path / "chapter0002.xhtml"
    pending_source = tmp_path / "chapter0003.xhtml"
    untranslated_source.write_text("<h1>Source Title</h1><p>Source body.</p>", encoding="utf-8")
    completed_source.write_text("<h1>Other Title</h1><p>Other body.</p>", encoding="utf-8")
    pending_source.write_text("<h1>Pending Title</h1><p>Pending body.</p>", encoding="utf-8")
    progress_manager_entries = [
        {
            "status": "not_translated",
            "filename": untranslated_source.name,
            "href": untranslated_source.name,
            "output_file": "response_chapter0001.html",
        },
        {
            "status": "completed",
            "filename": completed_source.name,
            "href": completed_source.name,
            "output_file": "response_chapter0002.html",
        },
        {
            "status": "pending",
            "filename": pending_source.name,
            "href": pending_source.name,
            "output_file": "response_chapter0003.html",
        },
    ]
    monkeypatch.setenv("OUTPUT_SDLXLIFF", "0")
    monkeypatch.setenv("RETAIN_SOURCE_EXTENSION", "0")
    mixin = RetranslationMixin.__new__(RetranslationMixin)

    stats = mixin._generate_sdlxliff_sidecars_from_untranslated_entries(
        str(tmp_path),
        progress_manager_entries,
    )

    sidecar = tmp_path / "SDLXLIFF" / "response_chapter0001.html.sdlxliff"
    pending_sidecar = tmp_path / "SDLXLIFF" / "response_chapter0003.html.sdlxliff"
    assert stats["total"] == 2
    assert stats["created"] == 2
    assert sidecar.is_file()
    assert pending_sidecar.is_file()
    assert not (tmp_path / "response_chapter0001.html").exists()
    assert not (tmp_path / "SDLXLIFF" / "response_chapter0002.html.sdlxliff").exists()
    assert _is_manual_untranslated_sdlxliff(sidecar)
    assert _is_manual_editing_sdlxliff(sidecar)
    assert _is_manual_untranslated_sdlxliff(pending_sidecar)
    assert os.environ["OUTPUT_SDLXLIFF"] == "0"

    dialog = SDLXLIFFReviewDialog.__new__(SDLXLIFFReviewDialog)
    source_html, target_html = dialog._read_sdlxliff_html_pair(str(sidecar))
    assert source_html != target_html
    assert "Source body." not in target_html
    target_soup = BeautifulSoup(target_html, "html.parser")
    assert target_soup.get_text(" ", strip=True) == ""
    assert target_soup.find("h1") is not None
    assert target_soup.find("p") is not None


def test_manual_untranslated_sidecar_is_dimmed_and_first_edit_creates_html(tmp_path, monkeypatch):
    monkeypatch.setenv("OUTPUT_SDLXLIFF", "1")
    monkeypatch.setenv("RETAIN_SOURCE_EXTENSION", "0")
    output_name = "response_chapter0001.html"
    source_html = "<html><body><p>Untranslated source.</p></body></html>"
    sidecar = _shared_write_html_sdlxliff_sidecar(
        str(tmp_path),
        output_name,
        {"original_basename": "chapter0001.xhtml"},
        source_html,
        source_html,
        raise_errors=True,
        manual_untranslated=True,
    )

    dialog = SDLXLIFFReviewDialog.__new__(SDLXLIFFReviewDialog)
    dialog.output_dir = str(tmp_path)
    dialog._last_review_signature = None
    dialog._current_review_signature = lambda: ()
    dialog._last_autogen_signature = None
    dialog._current_review_autogen_signature = lambda: ("viewer-save",)
    piece = dialog._build_piece(
        sidecar,
        0,
        {"output_name": output_name},
    )
    item = QListWidgetItem("manual")

    dialog._apply_piece_list_item_style(item, piece)
    assert item.foreground().color().alpha() == 115
    assert dialog._sdlxliff_sidecar_needs_source_regeneration(sidecar) is False
    assert piece["source_count"] == 1
    assert piece["target_count"] == 0
    assert piece["rows"][0]["source"] == "Untranslated source."
    assert piece["rows"][0]["target"] == ""
    assert piece["rows"][0]["target_index"] == 0

    edited_html = dialog._target_html_with_edit(
        piece,
        piece["rows"][0],
        "Manually edited output.",
    )
    dialog._write_piece_target_html(piece, edited_html)

    output_path = tmp_path / output_name
    assert output_path.read_text(encoding="utf-8") == edited_html
    assert piece["manual_untranslated"] is False
    assert not _is_manual_untranslated_sdlxliff(sidecar)
    assert _is_manual_editing_sdlxliff(sidecar)
    assert dialog._last_autogen_signature == ("viewer-save",)
    _source, saved_target = dialog._read_sdlxliff_html_pair(sidecar)
    assert "Manually edited output." in saved_target

    reloaded_piece = dialog._build_piece(
        sidecar,
        0,
        {"output_name": output_name},
    )
    assert reloaded_piece["manual_editing"] is True
    assert reloaded_piece["target_count"] == 1
    assert reloaded_piece["rows"][0]["target"] == "Manually edited output."
    assert reloaded_piece["rows"][0]["target_index"] == 0


def test_manual_editing_generated_html_applies_image_rename_map(tmp_path, monkeypatch):
    monkeypatch.setenv("OUTPUT_SDLXLIFF", "1")
    monkeypatch.setenv("RETAIN_SOURCE_EXTENSION", "0")
    output_name = "response_chapter0001.html"
    source_html = (
        '<html><body><p>Untranslated source.</p>'
        '<img src="../Images/original.png">'
        '<svg><image href="../Images/diagram.svg"></image></svg>'
        '<div style="background-image: url(\'../Images/background.webp\')"></div>'
        '</body></html>'
    )
    sidecar = _shared_write_html_sdlxliff_sidecar(
        str(tmp_path),
        output_name,
        {"original_basename": "chapter0001.xhtml"},
        source_html,
        source_html,
        raise_errors=True,
        manual_untranslated=True,
    )
    (tmp_path / "image_rename_map.json").write_text(
        json.dumps(
            {
                "original.png": "chapter0001_img_1.png",
                "diagram.svg": "chapter0001_img_2.svg",
                "background.webp": "chapter0001_img_3.webp",
            }
        ),
        encoding="utf-8",
    )

    dialog = SDLXLIFFReviewDialog.__new__(SDLXLIFFReviewDialog)
    dialog.output_dir = str(tmp_path)
    dialog._last_review_signature = None
    dialog._current_review_signature = lambda: ()
    piece = dialog._build_piece(sidecar, 0, {"output_name": output_name})
    edited_html = dialog._target_html_with_edit(
        piece,
        piece["rows"][0],
        "Manually edited output.",
    )

    saved_html = dialog._write_piece_target_html(piece, edited_html)

    output_html = (tmp_path / output_name).read_text(encoding="utf-8")
    assert output_html == saved_html
    assert '../Images/chapter0001_img_1.png' in output_html
    assert '../Images/chapter0001_img_2.svg' in output_html
    assert '../Images/chapter0001_img_3.webp' in output_html
    assert "original.png" not in output_html
    assert "diagram.svg" not in output_html
    assert "background.webp" not in output_html
    _source, saved_target = dialog._read_sdlxliff_html_pair(sidecar)
    assert saved_target == output_html


def test_manual_edit_save_leaves_existing_progress_entry_untouched(tmp_path, monkeypatch):
    monkeypatch.setenv("OUTPUT_SDLXLIFF", "1")
    monkeypatch.setenv("RETAIN_SOURCE_EXTENSION", "0")
    output_name = "response_chapter0001.html"
    sidecar = _shared_write_html_sdlxliff_sidecar(
        str(tmp_path),
        output_name,
        {"original_basename": "chapter0001.xhtml"},
        "<html><body><p>Source.</p></body></html>",
        "<html><body><p>Source.</p></body></html>",
        raise_errors=True,
        manual_untranslated=True,
    )
    progress_path = tmp_path / "translation_progress.json"
    original_progress = {
        "chapters": {
            "1": {
                "actual_num": 1,
                "status": "not_translated",
                "output_file": output_name,
                "original_basename": "chapter0001.xhtml",
            }
        },
        "completed_list": [],
    }
    progress_path.write_text(json.dumps(original_progress), encoding="utf-8")

    dialog = SDLXLIFFReviewDialog.__new__(SDLXLIFFReviewDialog)
    dialog.output_dir = str(tmp_path)
    dialog.current_path = sidecar
    dialog._config = {"retain_source_extension": False}
    dialog._last_review_signature = None
    dialog._current_review_signature = lambda: ()
    piece = dialog._build_piece(
        sidecar,
        0,
        {
            "output_name": output_name,
            "original_name": "chapter0001.xhtml",
            "progress_key": "1",
        },
    )
    edited_html = dialog._target_html_with_edit(piece, piece["rows"][0], "Manual target.")

    dialog._write_piece_target_html(piece, edited_html)

    assert json.loads(progress_path.read_text(encoding="utf-8")) == original_progress
    assert dialog._piece_needs_manual_green_override(piece) is True

    result = dialog._mark_piece_progress_completed(piece)

    assert result == {"ok": True, "matched": 1, "error": ""}
    completed_entry = json.loads(progress_path.read_text(encoding="utf-8"))["chapters"]["1"]
    assert completed_entry["status"] == "completed"
    assert "manual_editing_pending" not in completed_entry


def test_manual_edit_save_seeds_pending_progress_before_creating_html(tmp_path, monkeypatch):
    monkeypatch.setenv("OUTPUT_SDLXLIFF", "1")
    monkeypatch.setenv("RETAIN_SOURCE_EXTENSION", "0")
    output_name = "response_chapter0019.html"
    sidecar = _shared_write_html_sdlxliff_sidecar(
        str(tmp_path),
        output_name,
        {"original_basename": "chapter0019.xhtml"},
        "<html><body><p>Source.</p></body></html>",
        "<html><body><p>Source.</p></body></html>",
        raise_errors=True,
        manual_untranslated=True,
    )

    dialog = SDLXLIFFReviewDialog.__new__(SDLXLIFFReviewDialog)
    dialog.output_dir = str(tmp_path)
    dialog.current_path = sidecar
    dialog._config = {"retain_source_extension": False}
    dialog._last_review_signature = None
    dialog._current_review_signature = lambda: ()
    dialog._last_autogen_signature = None
    dialog._current_review_autogen_signature = lambda: ("manual-save",)
    piece = dialog._build_piece(
        sidecar,
        0,
        {
            "output_name": output_name,
            "original_name": "chapter0019.xhtml",
            "chapter_num": 19,
        },
    )
    output_path = tmp_path / output_name
    progress_write_preceded_output = []
    real_write = dialog._write_review_progress_data

    def traced_progress_write(path, progress_data):
        progress_write_preceded_output.append(not output_path.exists())
        return real_write(path, progress_data)

    dialog._write_review_progress_data = traced_progress_write
    edited_html = dialog._target_html_with_edit(
        piece, piece["rows"][0], "Manual target."
    )

    dialog._write_piece_target_html(piece, edited_html)

    progress_path = tmp_path / "translation_progress.json"
    progress_data = json.loads(progress_path.read_text(encoding="utf-8"))
    assert progress_write_preceded_output == [True]
    assert output_path.is_file()
    assert list(progress_data["chapters"]) == ["19"]
    pending_entry = progress_data["chapters"]["19"]
    assert pending_entry["actual_num"] == 19
    assert pending_entry["status"] == "pending"
    assert pending_entry["manual_editing_pending"] is True
    assert pending_entry["output_file"] == output_name
    assert pending_entry["original_basename"] == "chapter0019.xhtml"
    assert progress_data["completed_list"] == []
    assert piece["progress_key"] == "19"
    assert piece["manual_editing_pending"] is True


def test_manual_edit_save_retains_source_name_and_extension_when_enabled(tmp_path, monkeypatch):
    monkeypatch.setenv("OUTPUT_SDLXLIFF", "1")
    monkeypatch.setenv("RETAIN_SOURCE_EXTENSION", "1")
    old_output_name = "response_chapter0001.html"
    retained_output_name = "chapter0001.xhtml"
    sidecar = _shared_write_html_sdlxliff_sidecar(
        str(tmp_path),
        old_output_name,
        {"original_basename": retained_output_name},
        "<html><body><p>Source.</p></body></html>",
        "<html><body><p>Source.</p></body></html>",
        raise_errors=True,
        manual_untranslated=True,
    )
    progress_path = tmp_path / "translation_progress.json"
    progress_path.write_text(
        json.dumps(
            {
                "chapters": {
                    "1": {
                        "actual_num": 1,
                        "status": "not_translated",
                        "output_file": old_output_name,
                        "original_basename": retained_output_name,
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    dialog = SDLXLIFFReviewDialog.__new__(SDLXLIFFReviewDialog)
    dialog.output_dir = str(tmp_path)
    dialog.current_path = sidecar
    dialog._config = {"retain_source_extension": True}
    dialog._last_review_signature = None
    dialog._current_review_signature = lambda: ()
    piece = dialog._build_piece(
        sidecar,
        0,
        {
            "output_name": old_output_name,
            "original_name": retained_output_name,
            "progress_key": "1",
        },
    )
    edited_html = dialog._target_html_with_edit(piece, piece["rows"][0], "Manual target.")

    dialog._write_piece_target_html(piece, edited_html)

    assert not (tmp_path / old_output_name).exists()
    assert (tmp_path / retained_output_name).read_text(encoding="utf-8") == edited_html
    assert piece["output_name"] == retained_output_name
    assert piece["path"] == str(
        tmp_path / "SDLXLIFF" / f"{retained_output_name}.sdlxliff"
    )
    assert not os.path.exists(sidecar)
    retained_chapters = json.loads(
        progress_path.read_text(encoding="utf-8")
    )["chapters"]
    assert retained_chapters["1"] == {
        "actual_num": 1,
        "status": "not_translated",
        "output_file": old_output_name,
        "original_basename": retained_output_name,
    }
    new_entries = [
        entry
        for entry in retained_chapters.values()
        if entry.get("output_file") == retained_output_name
    ]
    assert len(new_entries) == 1
    assert new_entries[0]["status"] == "pending"
    assert new_entries[0]["manual_editing_pending"] is True

    renamed_sidecar = piece["path"]
    progress_map = dialog._read_progress_metadata()
    metadata = dialog._sidecar_metadata(renamed_sidecar, 0, progress_map, {})
    reloaded_piece = dialog._build_piece(renamed_sidecar, 0, metadata)
    assert reloaded_piece["output_name"] == retained_output_name
    assert reloaded_piece["manual_editing_pending"] is True
    assert dialog._piece_needs_manual_green_override(reloaded_piece) is True


def test_sdlxliff_mark_as_completed_updates_matching_progress_and_undoes(tmp_path):
    output_name = "response_chapter0001.html"
    sidecar = _shared_write_html_sdlxliff_sidecar(
        str(tmp_path),
        output_name,
        {"original_basename": "chapter0001.xhtml"},
        "<html><body><p>Untranslated source.</p></body></html>",
        "<html><body><p>Untranslated source.</p></body></html>",
        raise_errors=True,
        manual_untranslated=True,
    )
    progress_path = tmp_path / "translation_progress.json"
    original_progress = {
        "chapters": {
            "7": {
                "actual_num": 7,
                "status": "qa_failed",
                "output_file": f"nested\\{output_name}",
                "original_basename": "chapter0001.xhtml",
                "failure_reason": "blocked",
                "qa_issues_found": ["test issue"],
                "model_name": "test-model",
                "last_updated": 10.0,
            },
            "7_duplicate": {
                "actual_num": 7,
                "status": "pending",
                "output_file": output_name,
                "original_basename": "chapter0001.xhtml",
                "last_updated": 11.0,
            },
            "8": {
                "actual_num": 8,
                "status": "completed",
                "output_file": "response_chapter0002.html",
                "original_basename": "chapter0002.xhtml",
                "last_updated": 12.0,
            },
        },
        "completed_list": [
            {
                "num": 8,
                "idx": 0,
                "title": "chapter0002.xhtml",
                "file": "response_chapter0002.html",
                "key": "8",
            }
        ],
    }
    progress_path.write_text(json.dumps(original_progress), encoding="utf-8")

    live_progress = json.loads(json.dumps(original_progress))
    dialog = SDLXLIFFReviewDialog.__new__(SDLXLIFFReviewDialog)
    dialog.output_dir = str(tmp_path)
    dialog._config = {"output_language": "English"}
    dialog._book_entries = []
    dialog._last_autogen_signature = None
    dialog._sdlxliff_autogen_progress_data = live_progress
    piece = dialog._build_piece(sidecar, 0, {"output_name": output_name})

    assert dialog._piece_needs_manual_green_override(piece) is True
    result = dialog._mark_piece_progress_completed(piece)
    assert result == {"ok": True, "matched": 2, "error": ""}
    assert dialog._apply_manual_green_override_to_piece(piece) is True
    assert dialog._persist_piece_manual_green_override(piece) is True

    completed_progress = json.loads(progress_path.read_text(encoding="utf-8"))
    for key in ("7", "7_duplicate"):
        entry = completed_progress["chapters"][key]
        assert entry["status"] == "completed"
        assert entry["manually_marked_completed"] is True
        assert "failure_reason" not in entry
        assert "qa_issues_found" not in entry
    assert completed_progress["chapters"]["7"]["model_name"] == "test-model"
    assert live_progress["chapters"]["7"]["status"] == "completed"
    assert piece["mismatch"] is False

    reloaded_dialog = SDLXLIFFReviewDialog.__new__(SDLXLIFFReviewDialog)
    reloaded_dialog.output_dir = str(tmp_path)
    reloaded_dialog._config = {"output_language": "English"}
    reloaded_dialog._book_entries = []
    reloaded_dialog._last_autogen_signature = None
    reloaded_dialog._sdlxliff_autogen_progress_data = {}
    reloaded_piece = reloaded_dialog._build_piece(
        sidecar,
        0,
        {"output_name": output_name},
    )
    assert reloaded_piece["manual_green_override"] is True

    assert reloaded_dialog._clear_piece_manual_green_override(
        reloaded_piece,
        persist=True,
    ) is True
    restored_progress = json.loads(progress_path.read_text(encoding="utf-8"))
    assert restored_progress == original_progress
    override_data = json.loads(
        (tmp_path / "SDLXLIFF" / "review_status_overrides.json").read_text(
            encoding="utf-8"
        )
    )
    assert override_data["entries"] == {}


def test_sdlxliff_completion_context_action_uses_new_label_and_handler():
    source = (SRC / "Retranslation_GUI.py").read_text(encoding="utf-8")

    assert "Mark as Completed" in source
    assert "Mark Red/Yellow Sidecar as Green" not in source
    assert "self._mark_review_sidecars_completed(selected_rows)" in source


def test_manual_editing_sidecar_machine_translation_preview_button(tmp_path, monkeypatch, qtbot):
    output_name = "response_chapter0001.html"
    source_html = "<html><body>" + "".join(
        f"<p>수동 번역 원문 {index}입니다.</p>"
        for index in range(12)
    ) + "</body></html>"
    sidecar = _shared_write_html_sdlxliff_sidecar(
        str(tmp_path),
        output_name,
        {"original_basename": "chapter0001.xhtml"},
        source_html,
        source_html,
        raise_errors=True,
        manual_untranslated=True,
    )

    class Translator:
        call_sizes = []

        @classmethod
        def translate(cls, batch_html):
            soup = BeautifulSoup(batch_html, "html.parser")
            nodes = soup.find_all(attrs={"data-sdl-tip": True})
            cls.call_sizes.append(len(nodes))
            for node in nodes:
                node.string = f"Manual sidecar preview {node['data-sdl-tip']}."
            return {"translatedText": str(soup)}

    dialog = SDLXLIFFReviewDialog(
        str(tmp_path),
        sidecar,
        config={"output_language": "English"},
    )
    qtbot.addWidget(dialog)
    dialog.show()
    monkeypatch.setattr(
        dialog,
        "_machine_translation_translator",
        lambda _target_code, status_callback=None: Translator(),
    )
    qtbot.waitUntil(lambda: bool(dialog.pieces), timeout=5000)
    assert dialog.pieces[0]["manual_untranslated"] is True
    assert dialog.pieces[0]["target_count"] == 0

    dialog.translate_tooltips_btn.click()

    qtbot.waitUntil(
        lambda: not dialog._tooltip_translation_running,
        timeout=5000,
    )
    assert Translator.call_sizes == [12]
    assert all(
        row["tooltip_translation"].startswith("Manual sidecar preview ")
        for row in dialog.pieces[0]["rows"]
    )
    preview_path = (
        tmp_path
        / "SDLXLIFF"
        / "Machine_Translation"
        / f"{output_name}.json"
    )
    assert preview_path.is_file()
    preview_data = json.loads(preview_path.read_text(encoding="utf-8"))
    assert len(preview_data["entries"]) == 12


def test_sdlxliff_notepad_mode_is_one_rendered_editable_browser(tmp_path, qtbot):
    from PySide6.QtWebEngineCore import QWebEngineSettings
    from PySide6.QtWebEngineWidgets import QWebEngineView

    output_name = "response_chapter0001.html"
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    (images_dir / "chapter0001_img_1.png").write_bytes(base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
    ))
    (tmp_path / "image_rename_map.json").write_text(
        json.dumps({"cover.png": "chapter0001_img_1.png"}), encoding="utf-8"
    )
    html = (
        '<html><head><meta charset="utf-8"><title></title></head>'
        '<body><div><img src="../Images/cover.png"><p>Translated line '
        '<strong>protected</strong></p><p id="empty"></p><br></div></body></html>'
    )
    sidecar = _shared_write_html_sdlxliff_sidecar(
        str(tmp_path),
        output_name,
        {"original_basename": "chapter0001.xhtml"},
        '<html><body><p>Source line</p><p id="empty"></p></body></html>',
        html,
        raise_errors=True,
    )
    dialog = SDLXLIFFReviewDialog(
        str(tmp_path),
        sidecar,
        config={
            "output_language": "English",
            SDLXLIFFReviewDialog.TWO_COLUMN_LAYOUT_CONFIG_KEY: False,
        },
    )
    qtbot.addWidget(dialog)
    dialog.show()

    qtbot.waitUntil(
        lambda: bool(dialog.pieces) and 0 in dialog._piece_render_complete,
        timeout=5000,
    )
    browser = dialog.rows_widget.findChild(QWebEngineView, "SdlReviewNotepadBrowser")

    def js_value(script):
        values = []
        browser.page().runJavaScript(script, values.append)
        qtbot.waitUntil(lambda: bool(values), timeout=5000)
        return values[0]

    assert dialog.two_column_layout_btn.text() == "Notepad"
    assert browser is not None
    assert browser.toolTip() == ""
    qtbot.waitUntil(
        lambda: js_value("document.querySelectorAll('[data-sdl-notepad-text]').length") >= 2,
        timeout=5000,
    )
    assert browser._sdl_edit_poll_timer.isActive()
    visible_text = js_value("document.body.innerText")
    assert "Translated line" in visible_text
    assert "protected" in visible_text
    assert "<html" not in visible_text
    assert "<p>" not in visible_text
    assert js_value("document.querySelectorAll('img').length") == 1
    assert browser.settings().testAttribute(QWebEngineSettings.AutoLoadImages) is True
    assert browser.settings().testAttribute(
        QWebEngineSettings.LocalContentCanAccessRemoteUrls
    ) is True
    assert browser.settings().testAttribute(
        QWebEngineSettings.LocalContentCanAccessFileUrls
    ) is True
    assert js_value("document.querySelector('img').src").startswith("file:")
    assert js_value(
        "document.querySelector('img').getAttribute('data-sdl-notepad-original-src')"
    ) == "../Images/cover.png"
    qtbot.waitUntil(
        lambda: js_value(
            "document.querySelector('img').complete && document.querySelector('img').naturalWidth > 0"
        ) is True,
        timeout=5000,
    )
    assert js_value("getComputedStyle(document.querySelector('p')).display") == "block"
    assert js_value("document.designMode") == "off"
    assert js_value("getComputedStyle(document.body).backgroundColor") == "rgb(30, 30, 30)"
    assert js_value("getComputedStyle(document.body).color") == "rgb(232, 237, 242)"
    assert js_value("document.body.isContentEditable") is False
    assert js_value("document.body.getAttribute('contenteditable')") == "false"
    assert js_value("document.querySelector('p').isContentEditable") is False
    assert js_value("document.querySelector('strong').isContentEditable") is False
    assert js_value("document.querySelector('[data-sdl-notepad-text]').isContentEditable") is True
    assert js_value("document.querySelectorAll('[data-sdl-notepad-break]').length") == 1
    assert js_value("document.querySelector('[data-sdl-notepad-break]').isContentEditable") is True
    assert js_value(
        "document.querySelector('[data-sdl-notepad-break]').firstElementChild.tagName"
    ) == "BR"
    assert not js_value("document.querySelector('[data-sdl-notepad-user-tag]')")
    assert js_value(
        """
        (() => {
            const host = document.querySelector('[data-sdl-notepad-break]');
            host.focus();
            const selection = window.getSelection();
            const range = document.createRange();
            range.selectNodeContents(host);
            range.collapse(false);
            selection.removeAllRanges();
            selection.addRange(range);
            const event = new KeyboardEvent('keydown', {
                key: 'Backspace', bubbles: true, cancelable: true
            });
            host.dispatchEvent(event);
            return event.defaultPrevented && host.querySelector('br') !== null;
        })();
        """
    ) is True
    assert js_value(
        "document.querySelector('[data-sdl-notepad-text]').getAttribute('contenteditable')"
    ) == "true"
    assert js_value(
        "document.querySelector('[data-sdl-notepad-text]').getAttribute('data-sdl-notepad-source')"
    ) == "Source line"
    assert js_value("getComputedStyle(document.querySelector('#sdl-notepad-source-tooltip')).whiteSpace") == "pre-wrap"
    assert js_value(
        """
        (() => {
            const hosts = Array.from(document.querySelectorAll('[data-sdl-notepad-text]'));
            const first = hosts[0];
            const second = document.querySelector('#empty [data-sdl-notepad-text]');
            first.focus();
            const selection = window.getSelection();
            const range = document.createRange();
            range.setStart(first.firstChild, Math.min(2, first.firstChild.length));
            range.collapse(true);
            selection.removeAllRanges();
            selection.addRange(range);
            const down = new KeyboardEvent('keydown', {
                key: 'ArrowDown', bubbles: true, cancelable: true
            });
            first.dispatchEvent(down);
            const movedDown = document.activeElement === second;
            const up = new KeyboardEvent('keydown', {
                key: 'ArrowUp', bubbles: true, cancelable: true
            });
            second.dispatchEvent(up);
            return JSON.stringify([
                down.defaultPrevented,
                movedDown,
                up.defaultPrevented,
                document.activeElement === first
            ]);
        })();
        """
    ) == "[true,true,true,true]"
    assert js_value(
        """
        (() => {
            const fixture = document.createElement('div');
            const makeHost = text => {
                const host = document.createElement('span');
                host.setAttribute('data-sdl-notepad-text', '1');
                host.setAttribute('contenteditable', 'true');
                host.appendChild(document.createTextNode(text));
                return host;
            };
            const splitRow = document.createElement('p');
            const splitLeft = makeHost('[');
            const splitRight = makeHost('same visible row]');
            splitRow.append(splitLeft, splitRight);
            const manualBreakRow = document.createElement('p');
            const manualHost = makeHost('Translator');
            const firstBreak = document.createElement('br');
            const secondBreak = document.createElement('br');
            firstBreak.setAttribute('data-sdl-notepad-user-tag', 'br');
            secondBreak.setAttribute('data-sdl-notepad-user-tag', 'br');
            manualHost.append(firstBreak, secondBreak);
            manualBreakRow.appendChild(manualHost);
            const followingRow = document.createElement('p');
            const followingHost = makeHost('Following row');
            followingRow.appendChild(followingHost);
            fixture.append(splitRow, manualBreakRow, followingRow);
            document.body.appendChild(fixture);

            splitLeft.focus();
            const selection = window.getSelection();
            const range = document.createRange();
            range.selectNodeContents(splitLeft);
            range.collapse(false);
            selection.removeAllRanges();
            selection.addRange(range);
            const press = (host, key) => {
                const event = new KeyboardEvent('keydown', {
                    key, bubbles: true, cancelable: true
                });
                host.dispatchEvent(event);
                return event.defaultPrevented;
            };
            const afterBreak = lineBreak => {
                const current = selection.rangeCount ? selection.getRangeAt(0) : null;
                return !!current
                    && current.startContainer === lineBreak.parentNode
                    && current.startOffset === Array.from(
                        lineBreak.parentNode.childNodes
                    ).indexOf(lineBreak) + 1;
            };

            const firstDown = press(splitLeft, 'ArrowDown');
            const skippedInlineSibling = document.activeElement === manualHost;
            const secondDown = press(manualHost, 'ArrowDown');
            const selectedFirstEmptyLine = afterBreak(firstBreak);
            const thirdDown = press(manualHost, 'ArrowDown');
            const selectedSecondEmptyLine = afterBreak(secondBreak);
            const fourthDown = press(manualHost, 'ArrowDown');
            const reachedFollowingRow = document.activeElement === followingHost;
            const up = press(followingHost, 'ArrowUp');
            const returnedToLastEmptyLine = afterBreak(secondBreak);
            fixture.remove();
            return JSON.stringify([
                firstDown,
                skippedInlineSibling,
                secondDown,
                selectedFirstEmptyLine,
                thirdDown,
                selectedSecondEmptyLine,
                fourthDown,
                reachedFollowingRow,
                up,
                returnedToLastEmptyLine
            ]);
        })();
        """
    ) == "[true,true,true,true,true,true,true,true,true,true]"
    assert js_value(
        """
        (() => {
            const hosts = Array.from(
                document.querySelectorAll('[data-sdl-notepad-text]')
            );
            const breakHost = document.querySelector('[data-sdl-notepad-break]');
            const breakIndex = hosts.indexOf(breakHost);
            const previous = hosts[breakIndex - 1];
            previous.focus();
            const selection = window.getSelection();
            const range = document.createRange();
            range.selectNodeContents(previous);
            range.collapse(false);
            selection.removeAllRanges();
            selection.addRange(range);
            const down = new KeyboardEvent('keydown', {
                key: 'ArrowDown', bubbles: true, cancelable: true
            });
            previous.dispatchEvent(down);
            const movedIntoBreak = document.activeElement === breakHost;
            const caretIsText = selection.anchorNode
                && selection.anchorNode.nodeType === Node.TEXT_NODE;
            const up = new KeyboardEvent('keydown', {
                key: 'ArrowUp', bubbles: true, cancelable: true
            });
            breakHost.dispatchEvent(up);
            return JSON.stringify([
                down.defaultPrevented,
                movedIntoBreak,
                caretIsText,
                up.defaultPrevented,
                document.activeElement === previous,
                breakHost.querySelector('br') !== null
            ]);
        })();
        """
    ) == "[true,true,true,true,true,true]"
    assert js_value(
        """
        (() => {
            const paragraph = document.querySelector('p');
            const hosts = Array.from(
                paragraph.querySelectorAll('[data-sdl-notepad-text]')
            );
            hosts.forEach(host => { host.textContent = ''; });
            hosts[0].dispatchEvent(new InputEvent('input', {bubbles: true}));
            const markedRed = paragraph.hasAttribute(
                'data-sdl-notepad-user-empty-container'
            ) && getComputedStyle(paragraph).boxShadow.includes('rgb(220, 53, 69)');
            const originalEmptyStayedNeutral = !document.querySelector('#empty').hasAttribute(
                'data-sdl-notepad-user-empty-container'
            );
            const userAddedHost = document.querySelector(
                '#empty [data-sdl-notepad-text]'
            );
            userAddedHost.textContent = 'Temporary translator note';
            userAddedHost.dispatchEvent(new InputEvent('input', {bubbles: true}));
            userAddedHost.textContent = '';
            userAddedHost.dispatchEvent(new InputEvent('input', {bubbles: true}));
            const deletedUserAdditionStayedNeutral = !document.querySelector(
                '#empty'
            ).hasAttribute('data-sdl-notepad-user-empty-container');
            hosts[0].textContent = 'Translated line ';
            hosts[1].textContent = 'protected';
            hosts[0].dispatchEvent(new InputEvent('input', {bubbles: true}));
            return JSON.stringify([
                markedRed,
                originalEmptyStayedNeutral,
                deletedUserAdditionStayedNeutral,
                !paragraph.hasAttribute('data-sdl-notepad-user-empty-container')
            ]);
        })();
        """
    ) == "[true,true,true,true]"
    assert js_value(
        """
        (() => {
            const host = document.querySelector('[data-sdl-notepad-text]');
            host.dispatchEvent(new MouseEvent('mouseover', {
                bubbles: true, clientX: 20, clientY: 20
            }));
            const tooltip = document.querySelector('#sdl-notepad-source-tooltip');
            return tooltip.style.display === 'block' && tooltip.textContent === 'Source line';
        })();
        """
    ) is True
    context_menu = dialog._show_notepad_browser_context_menu(
        browser,
        browser.rect().center(),
        "Full wrapped source text for this sentence.",
        {"bold": True, "italic": False, "underline": True},
    )
    assert "padding: 6px 18px 6px 6px" in context_menu.styleSheet()
    assert "QMenu::indicator { width: 0px; height: 0px; }" in context_menu.styleSheet()
    source_menu_label = context_menu.findChild(QLabel, "SdlNotepadContextSourceText")
    assert source_menu_label is not None
    assert source_menu_label.wordWrap() is True
    assert source_menu_label.text() == "Full wrapped source text for this sentence."
    menu_actions = {
        action.objectName(): action for action in context_menu.actions()
        if action.objectName()
    }
    bold_action = menu_actions["SdlNotepadFormatBoldAction"]
    italic_action = menu_actions["SdlNotepadFormatItalicAction"]
    underline_action = menu_actions["SdlNotepadFormatUnderlineAction"]
    assert bold_action.isCheckable() is True
    assert bold_action.isChecked() is True
    assert bold_action.text() == "✓ Bold"
    assert italic_action.isCheckable() is True
    assert italic_action.isChecked() is False
    assert italic_action.text() == "Italic"
    assert underline_action.isCheckable() is True
    assert underline_action.isChecked() is True
    assert underline_action.text() == "✓ Underline"
    visible_action_order = [
        action.text().removeprefix("✓ ")
        for action in context_menu.actions() if action.text()
    ]
    assert visible_action_order.index("Bold") < visible_action_order.index("Undo")
    assert visible_action_order.index("Italic") < visible_action_order.index("Undo")
    assert visible_action_order.index("Underline") < visible_action_order.index("Undo")
    context_menu.close()
    qtbot.waitUntil(
        lambda: getattr(dialog, "_review_text_context_menu", None) is None,
        timeout=2000,
    )
    assert js_value(
        """
        (() => {
            const host = document.querySelector('strong [data-sdl-notepad-text]');
            const selection = window.getSelection();
            const range = document.createRange();
            range.selectNodeContents(host);
            selection.removeAllRanges();
            selection.addRange(range);
            const rect = host.getBoundingClientRect();
            host.dispatchEvent(new MouseEvent('contextmenu', {
                bubbles: true,
                cancelable: true,
                clientX: Math.round(rect.left + rect.width / 2),
                clientY: Math.round(rect.top + rect.height / 2)
            }));
            return selection.toString() === 'protected';
        })();
        """
    ) is True
    dialog._request_notepad_browser_context_menu(browser, browser.rect().center())
    qtbot.waitUntil(
        lambda: getattr(dialog, "_review_text_context_menu", None) is not None,
        timeout=5000,
    )
    detected_menu = dialog._review_text_context_menu
    detected_actions = {
        action.objectName(): action for action in detected_menu.actions()
        if action.objectName()
    }
    assert detected_actions["SdlNotepadFormatBoldAction"].isChecked() is True
    assert detected_actions["SdlNotepadFormatBoldAction"].text() == "✓ Bold"
    assert detected_actions["SdlNotepadFormatItalicAction"].isChecked() is False
    assert detected_actions["SdlNotepadFormatItalicAction"].text() == "Italic"
    detected_menu.close()
    history_shortcut_result = js_value(
        """
        (() => { try {
            const host = document.querySelector('p > [data-sdl-notepad-text]');
            host.focus();
            const selection = window.getSelection();
            const range = document.createRange();
            range.selectNodeContents(host);
            selection.removeAllRanges();
            selection.addRange(range);
            document.execCommand('insertText', false, 'Undo candidate');
            const undo = new KeyboardEvent('keydown', {
                key: 'z', ctrlKey: true, bubbles: true, cancelable: true
            });
            host.dispatchEvent(undo);
            const textAfterUndo = host.textContent;
            const redo = new KeyboardEvent('keydown', {
                key: 'y', ctrlKey: true, bubbles: true, cancelable: true
            });
            host.dispatchEvent(redo);
            const textAfterRedo = host.textContent;
            const cleanup = new KeyboardEvent('keydown', {
                key: 'z', ctrlKey: true, bubbles: true, cancelable: true
            });
            host.dispatchEvent(cleanup);
            return JSON.stringify([
                undo.defaultPrevented,
                textAfterUndo === 'Translated line ',
                redo.defaultPrevented,
                textAfterRedo === 'Undo candidate',
                cleanup.defaultPrevented,
                host.textContent === 'Translated line '
            ]);
        } catch (error) { return String(error && error.stack || error); }
        })();
        """
    )
    assert history_shortcut_result == "[true,true,true,true,true,true]"
    enter_result = js_value(
        """
        (() => { try {
            const hosts = Array.from(document.querySelectorAll('[data-sdl-notepad-text]'));
            const first = hosts[0];
            first.focus();
            const selection = window.getSelection();
            const range = document.createRange();
            range.selectNodeContents(first);
            range.collapse(false);
            selection.removeAllRanges();
            selection.addRange(range);
            const event = new KeyboardEvent('keydown', {
                key: 'Enter', bubbles: true, cancelable: true
            });
            first.dispatchEvent(event);
            const created = first.querySelectorAll('br').length === 1;
            const indicator = first.closest('p');
            const marked = first.querySelector('br').getAttribute(
                'data-sdl-notepad-user-tag'
            ) === 'br' && indicator.hasAttribute('data-sdl-notepad-user-tag-container');
            const undo = new KeyboardEvent('keydown', {
                key: 'z', ctrlKey: true, bubbles: true, cancelable: true
            });
            first.dispatchEvent(undo);
            const undone = first.querySelectorAll('br').length === 0;
            const indicatorUndone = !indicator.hasAttribute(
                'data-sdl-notepad-user-tag-container'
            );
            const redo = new KeyboardEvent('keydown', {
                key: 'y', ctrlKey: true, bubbles: true, cancelable: true
            });
            first.dispatchEvent(redo);
            return JSON.stringify([
                event.defaultPrevented,
                created,
                marked,
                undo.defaultPrevented,
                undone,
                indicatorUndone,
                redo.defaultPrevented,
                first.querySelector('br').getAttribute('data-sdl-notepad-user-tag') === 'br',
                indicator.hasAttribute('data-sdl-notepad-user-tag-container'),
                document.activeElement === first
            ]);
        } catch (error) { return String(error && error.stack || error); }
        })();
        """
    )
    assert enter_result == "[true,true,true,true,true,true,true,true,true,true]"
    user_break_delete_result = js_value(
        """
        (() => { try {
            const host = document.querySelector('p > [data-sdl-notepad-text]');
            const before = host.querySelectorAll('br').length;
            const backspace = new KeyboardEvent('keydown', {
                key: 'Backspace', bubbles: true, cancelable: true
            });
            host.dispatchEvent(backspace);
            const deleted = host.querySelectorAll('br').length === before - 1;
            const undoBackspace = new KeyboardEvent('keydown', {
                key: 'z', ctrlKey: true, bubbles: true, cancelable: true
            });
            host.dispatchEvent(undoBackspace);
            const backspaceUndone = host.querySelectorAll('br').length === before;
            const redoBackspace = new KeyboardEvent('keydown', {
                key: 'y', ctrlKey: true, bubbles: true, cancelable: true
            });
            host.dispatchEvent(redoBackspace);
            const backspaceRedone = host.querySelectorAll('br').length === before - 1;
            const recreate = new KeyboardEvent('keydown', {
                key: 'Enter', bubbles: true, cancelable: true
            });
            host.dispatchEvent(recreate);
            const recreatedBreak = host.querySelector('br');
            const selection = window.getSelection();
            const range = document.createRange();
            range.setStartBefore(recreatedBreak);
            range.collapse(true);
            selection.removeAllRanges();
            selection.addRange(range);
            const forwardDelete = new KeyboardEvent('keydown', {
                key: 'Delete', bubbles: true, cancelable: true
            });
            host.dispatchEvent(forwardDelete);
            const forwardDeleted = host.querySelectorAll('br').length === before - 1;
            const undoDelete = new KeyboardEvent('keydown', {
                key: 'z', ctrlKey: true, bubbles: true, cancelable: true
            });
            host.dispatchEvent(undoDelete);
            return JSON.stringify([
                backspace.defaultPrevented,
                deleted,
                undoBackspace.defaultPrevented,
                backspaceUndone,
                redoBackspace.defaultPrevented,
                backspaceRedone,
                recreate.defaultPrevented,
                forwardDelete.defaultPrevented,
                forwardDeleted,
                undoDelete.defaultPrevented,
                host.querySelectorAll('br').length === before
            ]);
        } catch (error) { return String(error && error.stack || error); }
        })();
        """
    )
    assert user_break_delete_result == "[true,true,true,true,true,true,true,true,true,true,true]"
    one_above_break_fallback_result = js_value(
        """
        (() => { try {
            const container = document.createElement('div');
            container.innerHTML =
                '<span data-sdl-notepad-text="1" contenteditable="true">' +
                    '<br data-sdl-notepad-user-tag="br"></span>' +
                '<span data-sdl-notepad-text="1" contenteditable="true"></span>' +
                '<span data-sdl-notepad-text="1" contenteditable="true"></span>';
            document.body.appendChild(container);
            const hosts = Array.from(container.querySelectorAll('[data-sdl-notepad-text]'));
            const selection = window.getSelection();
            const placeAtStart = host => {
                host.focus();
                const range = document.createRange();
                range.selectNodeContents(host);
                range.collapse(true);
                selection.removeAllRanges();
                selection.addRange(range);
            };
            placeAtStart(hosts[2]);
            const tooFar = new KeyboardEvent('keydown', {
                key: 'Backspace', bubbles: true, cancelable: true
            });
            hosts[2].dispatchEvent(tooFar);
            const distantBreakPreserved = hosts[0].querySelector('br') !== null;
            placeAtStart(hosts[1]);
            const adjacent = new KeyboardEvent('keydown', {
                key: 'Backspace', bubbles: true, cancelable: true
            });
            hosts[1].dispatchEvent(adjacent);
            const adjacentBreakDeleted = hosts[0].querySelector('br') === null;
            container.remove();
            return JSON.stringify([
                tooFar.defaultPrevented,
                distantBreakPreserved,
                adjacent.defaultPrevented,
                adjacentBreakDeleted
            ]);
        } catch (error) { return String(error && error.stack || error); }
        })();
        """
    )
    assert one_above_break_fallback_result == "[true,true,true,true]"
    boundary_delete_result = js_value(
        """
        (() => { try {
            const host = document.querySelector('p > [data-sdl-notepad-text]');
            const selection = window.getSelection();
            const placeCaret = atStart => {
                host.focus();
                const range = document.createRange();
                range.selectNodeContents(host);
                range.collapse(atStart);
                selection.removeAllRanges();
                selection.addRange(range);
            };
            placeCaret(true);
            const beforeBackspace = document.body.innerHTML;
            const backspace = new KeyboardEvent('keydown', {
                key: 'Backspace', bubbles: true, cancelable: true
            });
            host.dispatchEvent(backspace);
            const backspaceUnchanged = beforeBackspace === document.body.innerHTML;
            placeCaret(false);
            const beforeDelete = document.body.innerHTML;
            const forwardDelete = new KeyboardEvent('keydown', {
                key: 'Delete', bubbles: true, cancelable: true
            });
            host.dispatchEvent(forwardDelete);
            return JSON.stringify([
                backspace.defaultPrevented,
                backspaceUnchanged,
                forwardDelete.defaultPrevented,
                beforeDelete === document.body.innerHTML
            ]);
        } catch (error) { return String(error && error.stack || error); }
        })();
        """
    )
    assert boundary_delete_result == "[true,true,true,true]"
    empty_paragraph_result = js_value(
        """
        (() => { try {
            const paragraph = document.querySelector('#empty');
            const host = paragraph.querySelector('[data-sdl-notepad-text]');
            host.focus();
            const selection = window.getSelection();
            const range = document.createRange();
            range.selectNodeContents(host);
            range.collapse(true);
            selection.removeAllRanges();
            selection.addRange(range);
            const event = new KeyboardEvent('keydown', {
                key: 'Backspace', bubbles: true, cancelable: true
            });
            host.dispatchEvent(event);
            return JSON.stringify([
                event.defaultPrevented,
                document.querySelector('#empty') === paragraph,
                paragraph.isConnected
            ]);
        } catch (error) { return String(error && error.stack || error); }
        })();
        """
    )
    assert empty_paragraph_result == "[true,true,true]"
    assert js_value(
        """
        (() => {
            const host = document.querySelector('p > [data-sdl-notepad-text]');
            host.focus();
            const selection = window.getSelection();
            const range = document.createRange();
            range.selectNodeContents(host);
            range.collapse(true);
            selection.removeAllRanges();
            selection.addRange(range);
            const event = new InputEvent('beforeinput', {
                inputType: 'deleteContentBackward', bubbles: true, cancelable: true
            });
            host.dispatchEvent(event);
            return event.defaultPrevented;
        })();
        """
    ) is True
    assert js_value(
        """
        (() => {
            const host = document.querySelector('[data-sdl-notepad-text]');
            host.focus();
            const selection = window.getSelection();
            const range = document.createRange();
            range.selectNodeContents(host);
            range.collapse(false);
            selection.removeAllRanges();
            selection.addRange(range);
            const before = host.querySelectorAll('br').length;
            const event = new InputEvent('beforeinput', {
                inputType: 'insertParagraph', bubbles: true, cancelable: true
            });
            host.dispatchEvent(event);
            return event.defaultPrevented
                && host.querySelectorAll('br').length === before + 1;
        })();
        """
    ) is True
    assert js_value(
        """
        (() => {
            const host = document.querySelector('[data-sdl-notepad-text]');
            host.focus();
            const selection = window.getSelection();
            const range = document.createRange();
            range.selectNodeContents(host);
            range.collapse(false);
            selection.removeAllRanges();
            selection.addRange(range);
            const before = host.querySelectorAll('br').length;
            const event = new InputEvent('beforeinput', {
                inputType: 'insertLineBreak', bubbles: true, cancelable: true
            });
            host.dispatchEvent(event);
            return event.defaultPrevented
                && host.querySelectorAll('br').length === before + 1;
        })();
        """
    ) is True
    inline_format_result = js_value(
        """
        (() => { try {
            const paragraph = document.querySelector('#empty');
            const host = paragraph.querySelector('[data-sdl-notepad-text]');
            host.textContent = 'Styled addition';
            host.dispatchEvent(new InputEvent('input', {bubbles: true}));
            const selectHost = () => {
                host.focus();
                const selection = window.getSelection();
                const range = document.createRange();
                range.selectNodeContents(host);
                selection.removeAllRanges();
                selection.addRange(range);
            };
            selectHost();
            host.dispatchEvent(new MouseEvent('contextmenu', {bubbles: true}));
            window.getSelection().removeAllRanges();
            const bold = window.__sdlApplyInlineFormat('bold', true);
            window.getSelection().removeAllRanges();
            const italic = window.__sdlApplyInlineFormat('italic', true);
            window.getSelection().removeAllRanges();
            const underline = window.__sdlApplyInlineFormat('underline', true);
            return JSON.stringify([
                bold, italic, underline,
                host.querySelectorAll('strong').length === 1,
                host.querySelectorAll('em').length === 1,
                host.querySelectorAll('u').length === 1,
                window.getSelection().toString() === 'Styled addition',
                paragraph.isConnected,
                paragraph.textContent === 'Styled addition'
            ]);
        } catch (error) { return String(error && error.stack || error); }
        })();
        """
    )
    assert inline_format_result == "[true,true,true,true,true,true,true,true,true]"
    assert dialog.rows_widget.findChild(QPlainTextEdit, "SdlReviewNotepadEditor") is None
    assert dialog.rows_widget.findChildren(QFrame, "SdlReviewRow") == []
    assert dialog.rows_widget.findChildren(QPlainTextEdit, "SdlReviewTargetEdit") == []

    browser.page().runJavaScript(
        """
        (() => {
            const editor = document.querySelector('p > [data-sdl-notepad-text]');
            editor.innerHTML = 'Edited in the rendered page. <br>';
            editor.dispatchEvent(new InputEvent('input', {bubbles: true}));
        })();
        """
    )
    output_path = tmp_path / output_name
    qtbot.waitUntil(
        lambda: output_path.is_file()
        and "Edited in the rendered page." in output_path.read_text(encoding="utf-8"),
        timeout=5000,
    )
    saved_soup = BeautifulSoup(output_path.read_text(encoding="utf-8"), "html.parser")
    assert saved_soup.find("p").find("br") is not None
    assert saved_soup.find("strong").get_text(strip=True) == "protected"
    saved_addition = saved_soup.find("p", id="empty")
    assert saved_addition is not None
    assert saved_addition.get_text(strip=True) == "Styled addition"
    assert saved_addition.find("strong") is not None
    assert saved_addition.find("em") is not None
    assert saved_addition.find("u") is not None
    assert saved_soup.find(attrs={"data-sdl-notepad-text": True}) is None
    assert saved_soup.find(attrs={"data-sdl-notepad-source": True}) is None
    assert saved_soup.find(attrs={"data-sdl-notepad-original-editable": True}) is None
    assert saved_soup.find(attrs={"data-sdl-notepad-original-src": True}) is None
    assert saved_soup.find(attrs={"data-sdl-notepad-user-tag": True}) is None
    assert saved_soup.find(attrs={"data-sdl-notepad-user-tag-container": True}) is None
    assert saved_soup.find(attrs={"data-sdl-notepad-original-had-text": True}) is None
    assert saved_soup.find(attrs={"data-sdl-notepad-user-empty-container": True}) is None
    assert saved_soup.find("img").get("src") == "../Images/chapter0001_img_1.png"
    assert saved_soup.find(id="sdl-notepad-source-tooltip") is None
    assert output_path.read_text(encoding="utf-8").count("Edited in the rendered page.") == 1
    assert output_path.read_text(encoding="utf-8").count("protected") == 1
    notepad_tag_order = dialog.pieces[0]["_notepad_tag_order"]
    assert notepad_tag_order[:10] == [
        "html", "head", "meta", "title", "body", "div", "img", "p", "br", "strong"
    ]
    assert notepad_tag_order.count("p") == 2
    assert notepad_tag_order.count("strong") == 2
    assert "em" in notepad_tag_order
    assert "u" in notepad_tag_order
    assert notepad_tag_order.count("br") == 2
    assert notepad_tag_order[-1] == "br"

    # Image-path normalization reloads the rendered browser from clean saved
    # HTML. Source metadata must be reattached for both populated and
    # user-emptied rows without refilling the latter with source text.
    qtbot.waitUntil(
        lambda: js_value(
            "document.querySelector('p').getAttribute('data-sdl-notepad-source')"
            " === 'Source line' && !!document.querySelector('#sdl-notepad-source-tooltip')"
            " && document.querySelector('img').getAttribute("
            "'data-sdl-notepad-original-src')"
            " === '../Images/chapter0001_img_1.png'"
        ) is True,
        timeout=5000,
    )
    assert js_value(
        """
        (() => {
            const paragraph = document.querySelector('p');
            const hosts = Array.from(
                paragraph.querySelectorAll('[data-sdl-notepad-text]')
            );
            const snapshots = hosts.map(host => host.innerHTML);
            const tooltip = document.querySelector('#sdl-notepad-source-tooltip');
            tooltip.style.display = 'none';
            hosts[0].dispatchEvent(new MouseEvent('mouseover', {
                bubbles: true, clientX: 20, clientY: 20
            }));
            const populatedHover = tooltip.style.display === 'block'
                && tooltip.textContent === 'Source line';
            hosts.forEach(host => { host.textContent = ''; });
            hosts[0].dispatchEvent(new InputEvent('input', {bubbles: true}));
            tooltip.style.display = 'none';
            paragraph.dispatchEvent(new MouseEvent('mouseover', {
                bubbles: true, clientX: 20, clientY: 20
            }));
            const deletedHover = tooltip.style.display === 'block'
                && tooltip.textContent === 'Source line'
                && paragraph.hasAttribute('data-sdl-notepad-user-empty-container');
            hosts.forEach((host, index) => { host.innerHTML = snapshots[index]; });
            hosts[0].dispatchEvent(new InputEvent('input', {bubbles: true}));
            return JSON.stringify([populatedHover, deletedHover]);
        })();
        """
    ) == "[true,true]"

    dialog.two_column_layout_btn.click()
    assert dialog.two_column_layout_btn.text() == "Compact"


def test_notepad_applies_persisted_edge_markers_before_user_interaction(tmp_path, qtbot):
    from PySide6.QtWebEngineWidgets import QWebEngineView

    output_name = "response_chapter0001.html"
    source_html = (
        '<html><body><p id="manual">Source line<br/></p>'
        '<p id="deleted">Deleted source line.</p></body></html>'
    )
    target_html = (
        '<html><body><p id="manual">Translator<br/><br/></p>'
        '<p id="deleted"></p></body></html>'
    )
    sidecar = _shared_write_html_sdlxliff_sidecar(
        str(tmp_path),
        output_name,
        {"original_basename": "chapter0001.xhtml"},
        source_html,
        target_html,
        raise_errors=True,
    )
    dialog = SDLXLIFFReviewDialog(
        str(tmp_path),
        sidecar,
        config={SDLXLIFFReviewDialog.TWO_COLUMN_LAYOUT_CONFIG_KEY: False},
    )
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.waitUntil(
        lambda: bool(dialog.pieces) and 0 in dialog._piece_render_complete,
        timeout=5000,
    )
    browser = dialog.rows_widget.findChild(QWebEngineView, "SdlReviewNotepadBrowser")
    assert browser is not None

    def js_value(script):
        values = []
        browser.page().runJavaScript(script, values.append)
        qtbot.waitUntil(lambda: bool(values), timeout=5000)
        return values[0]

    qtbot.waitUntil(
        lambda: js_value(
            "document.querySelector('#manual')?.hasAttribute("
            "'data-sdl-notepad-user-tag-container') === true"
        ) is True,
        timeout=5000,
    )
    assert js_value(
        "getComputedStyle(document.querySelector('#manual')).boxShadow"
        ".includes('rgb(215, 168, 0)')"
    ) is True

    # Reopen the saved blank target without source fallback. This is the same
    # initialization path used when a normalization reload preserves a row the
    # user deliberately emptied.
    piece = dialog.pieces[0]
    dialog._set_notepad_browser_html(
        browser,
        piece,
        dialog._notepad_initial_document_html(piece, fill_untranslated=False),
    )
    qtbot.waitUntil(
        lambda: js_value(
            "document.querySelector('#deleted')?.hasAttribute("
            "'data-sdl-notepad-user-empty-container') === true"
        ) is True,
        timeout=5000,
    )
    assert js_value(
        "getComputedStyle(document.querySelector('#deleted')).boxShadow"
        ".includes('rgb(220, 53, 69)')"
    ) is True
    assert js_value(
        "document.querySelector('#manual')?.hasAttribute("
        "'data-sdl-notepad-user-tag-container') === true"
    ) is True
    assert js_value("window.__sdlNotepadDirty") is False


def test_manual_untranslated_notepad_renders_source_fallback_and_creates_output(tmp_path, qtbot):
    from PySide6.QtWebEngineWidgets import QWebEngineView

    output_name = "response_chapter0001.html"
    source_html = "<html><body><p>아직 번역되지 않은 원문입니다.</p></body></html>"
    sidecar = _shared_write_html_sdlxliff_sidecar(
        str(tmp_path),
        output_name,
        {"original_basename": "chapter0001.xhtml"},
        source_html,
        source_html,
        raise_errors=True,
        manual_untranslated=True,
    )
    dialog = SDLXLIFFReviewDialog(
        str(tmp_path),
        sidecar,
        config={SDLXLIFFReviewDialog.TWO_COLUMN_LAYOUT_CONFIG_KEY: False},
    )
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.waitUntil(
        lambda: bool(dialog.pieces) and 0 in dialog._piece_render_complete,
        timeout=5000,
    )
    browser = dialog.rows_widget.findChild(QWebEngineView, "SdlReviewNotepadBrowser")

    def js_value(script):
        values = []
        browser.page().runJavaScript(script, values.append)
        qtbot.waitUntil(lambda: bool(values), timeout=5000)
        return values[0]

    assert browser is not None
    qtbot.waitUntil(
        lambda: js_value("document.querySelectorAll('[data-sdl-notepad-text]').length") >= 1,
        timeout=5000,
    )
    assert "아직 번역되지 않은 원문입니다." in js_value("document.body.innerText")
    assert "<p>" not in js_value("document.body.innerText")
    browser.page().runJavaScript(
        """
        (() => {
            const editor = document.querySelector('p > [data-sdl-notepad-text]');
            editor.textContent = 'Manually translated.';
            editor.dispatchEvent(new InputEvent('input', {bubbles: true}));
        })();
        """
    )
    output_path = tmp_path / output_name
    qtbot.waitUntil(
        lambda: output_path.is_file() and "Manually translated." in output_path.read_text(encoding="utf-8"),
        timeout=5000,
    )

    assert dialog.pieces[0]["manual_untranslated"] is False
    assert "Manually translated." in dialog.pieces[0]["target_html"]


def test_notepad_initial_html_keeps_translation_and_expands_source_markup_for_empty_nodes():
    dialog = SDLXLIFFReviewDialog.__new__(SDLXLIFFReviewDialog)
    document = dialog._notepad_initial_document_html({
        "source_html": (
            "<html><body><p>Source one</p>"
            "<p>Source <strong>two</strong><br/><rb>ruby base</rb></p></body></html>"
        ),
        "target_html": (
            "<html><body><p>Translated one</p><p><br/></p></body></html>"
        ),
    })
    soup = BeautifulSoup(document, "html.parser")
    paragraphs = soup.find_all("p")

    assert paragraphs[0].get_text(" ", strip=True) == "Translated one"
    assert paragraphs[0].get("data-sdl-notepad-source") == "Source one"
    assert paragraphs[1].get_text(" ", strip=True) == "Source two ruby base"
    assert paragraphs[1].find("strong") is not None
    assert paragraphs[1].find("br") is not None
    assert paragraphs[1].find("rb") is not None
    assert "<strong>two</strong><br/><rb>ruby base</rb>" in document

    unfilled_document = dialog._notepad_initial_document_html(
        {
            "source_html": "<html><body><p>Source only</p></body></html>",
            "target_html": "<html><body><p></p></body></html>",
        },
        fill_untranslated=False,
    )
    unfilled_paragraph = BeautifulSoup(unfilled_document, "html.parser").find("p")
    assert unfilled_paragraph.get_text(strip=True) == ""
    assert unfilled_paragraph["data-sdl-notepad-source"] == "Source only"

    reopened_manual_document = dialog._notepad_initial_document_html(
        {
            "source_html": "<html><body><p>&nbsp;<br/></p></body></html>",
            "target_html": (
                "<html><body><p>&nbsp;Translator<br/><br/><br/></p></body></html>"
            ),
        },
        fill_untranslated=False,
    )
    reopened_breaks = BeautifulSoup(
        reopened_manual_document, "html.parser"
    ).find_all("br")
    assert [line_break.get("data-sdl-notepad-user-tag") for line_break in reopened_breaks] == [
        "br",
        "br",
        None,
    ]


def test_notepad_browser_cleanup_removes_only_editor_scaffolding():
    document = (
        '<!DOCTYPE html><html><head><style id="sdl-notepad-guard-style">x</style></head>'
        '<body contenteditable="false" data-sdl-notepad-original-editable="true">'
        '<img src="file:///tmp/rendered.png" '
        'data-sdl-notepad-original-src="../Images/original.png">'
        '<p data-sdl-notepad-source="Full source"><span contenteditable="plaintext-only" '
        'data-sdl-notepad-text="1" data-sdl-notepad-source="Full source">Edited </span>'
        '<strong><span contenteditable="true" data-sdl-notepad-text="1">bold</span></strong>'
        '<br></p><div id="sdl-notepad-source-tooltip">Full source</div></body></html>'
    )

    cleaned = SDLXLIFFReviewDialog._clean_notepad_browser_html(document)
    soup = BeautifulSoup(cleaned, "html.parser")

    assert soup.find("style", id="sdl-notepad-guard-style") is None
    assert soup.find(id="sdl-notepad-source-tooltip") is None
    assert soup.find(attrs={"data-sdl-notepad-text": True}) is None
    assert soup.find(attrs={"data-sdl-notepad-source": True}) is None
    assert soup.find(attrs={"data-sdl-notepad-original-editable": True}) is None
    assert soup.find("body").get("contenteditable") == "true"
    assert soup.find("img").get("src") == "../Images/original.png"
    assert soup.find("img").get("data-sdl-notepad-original-src") is None
    assert soup.find("p").get_text(" ", strip=True) == "Edited bold"
    assert soup.find("strong").get_text(strip=True) == "bold"
    assert soup.find("br") is not None


def test_progress_manager_exposes_manual_editing_toggle_for_not_translated_rows():
    source = (SRC / "Retranslation_GUI.py").read_text(encoding="utf-8")
    assert 'QCheckBox("Manual editing")' in source
    assert "manual_editing_cb.setChecked(self._get_retranslation_manual_editing_state())" in source
    assert "self._persist_retranslation_manual_editing_state(enabled)" in source
    assert "_progress_manager_untranslated_entries" in source
    assert "_generate_sdlxliff_sidecars_from_untranslated_entries" in source
    assert "'manual_editing_cb': manual_editing_cb" in source
    assert "'generate_manual_editing_sidecars': _generate_manual_editing_sidecars" in source
    assert "(_manual_editing_enabled() and _progress_item_is_html(display_info))" in source
    assert "if callable(generate_sidecars):" in source


def test_progress_manager_manual_editing_generation_does_not_block_gui_thread():
    source = (SRC / "Retranslation_GUI.py").read_text(encoding="utf-8")
    start = source.index("def _generate_manual_editing_sidecars(on_finished=None)")
    end = source.index("def _bool_setting", start)
    body = source[start:end]
    toggle_start = source.index("def _on_manual_editing_toggled", start)
    toggle_end = source.index("def _update_text_analysis_button", toggle_start)
    toggle_body = source[toggle_start:toggle_end]

    assert 'name="manual-sdlxliff-sidecar-generation"' in body
    assert "manual_generation_state['running']" in body
    assert "manual_generation_bridge.progress.emit" in body
    assert "now - last_progress_emit[0] < 0.1" in body
    assert "QApplication.processEvents" not in body
    assert "manual_editing_cb.setEnabled(False)" not in body
    assert "QTimer.singleShot(" in toggle_body
    assert "QTimer.singleShot(0, _generate_manual_editing_sidecars)" not in source
    assert "for delay in (0, 75, 250, 750)" not in source


def test_manual_editing_preference_persists_through_app_config():
    class StubRetranslation(RetranslationMixin):
        def __init__(self):
            self.config = {}
            self.saved = 0

        def save_config(self, show_message=False):
            self.saved += 1

    stub = StubRetranslation()

    assert stub._get_retranslation_manual_editing_state() is False
    stub._persist_retranslation_manual_editing_state(True)

    assert stub.config[stub._RETRANSLATION_MANUAL_EDITING_CONFIG_KEY] is True
    assert stub._get_retranslation_manual_editing_state() is True
    assert stub.saved == 1


def test_sdlxliff_viewer_refresh_generates_manual_entries_without_output_html(tmp_path):
    manual_entries = [{
        "status": "not_translated",
        "filename": "chapter0001.xhtml",
        "output_file": "chapter0001.xhtml",
    }]

    class Owner:
        def __init__(self):
            self.calls = []

        def _generate_sdlxliff_sidecars_from_untranslated_entries(
            self,
            output_dir,
            entries,
            file_path=None,
            progress_callback=None,
        ):
            self.calls.append((output_dir, entries, file_path))
            return {
                "total": 1,
                "considered": 1,
                "created": 1,
                "skipped": 0,
                "missing_source": 0,
                "missing_output": 0,
                "failed": 0,
                "paths": [str(tmp_path / "SDLXLIFF" / "chapter0001.xhtml.sdlxliff")],
                "errors": [],
            }

    owner = Owner()
    dialog = SDLXLIFFReviewDialog.__new__(SDLXLIFFReviewDialog)
    dialog.output_dir = str(tmp_path)
    dialog._config = {dialog.MANUAL_EDITING_CONFIG_KEY: True}
    dialog._sdlxliff_autogen_owner = owner
    dialog._sdlxliff_autogen_file_path = str(tmp_path / "book.epub")
    dialog._sdlxliff_autogen_manual_entries = manual_entries
    dialog._book_index = -1
    dialog._book_entries = []

    signature = dialog._current_review_autogen_signature()
    assert any(entry[0] == "manual_html" for entry in signature)
    stats = dialog._regenerate_review_sidecars_for_refresh_scan(
        current_signature=signature,
    )

    assert stats["created"] == 1
    assert owner.calls == [(
        str(tmp_path),
        manual_entries,
        str(tmp_path / "book.epub"),
    )]


def test_sdlxliff_viewer_refresh_only_checks_missing_manual_entries(tmp_path):
    existing_entry = {
        "status": "not_translated",
        "filename": "chapter0001.xhtml",
        "output_file": "response_chapter0001.html",
    }
    missing_entry = {
        "status": "not_translated",
        "filename": "chapter0002.xhtml",
        "output_file": "response_chapter0002.html",
    }
    (tmp_path / "response_chapter0001.html").write_text(
        "<html><body><p>Already being edited.</p></body></html>",
        encoding="utf-8",
    )

    class Owner:
        config = {"retain_source_extension": False}

        def __init__(self):
            self.entries = []

        def _generate_sdlxliff_sidecars_from_untranslated_entries(
            self,
            _output_dir,
            entries,
            file_path=None,
            progress_callback=None,
        ):
            self.entries.extend(entries)
            return {
                "total": len(entries),
                "considered": len(entries),
                "created": len(entries),
                "skipped": 0,
                "missing_source": 0,
                "missing_output": 0,
                "failed": 0,
                "paths": [],
                "errors": [],
            }

    owner = Owner()
    dialog = SDLXLIFFReviewDialog.__new__(SDLXLIFFReviewDialog)
    dialog.output_dir = str(tmp_path)
    dialog._config = {dialog.MANUAL_EDITING_CONFIG_KEY: True}
    dialog._sdlxliff_autogen_owner = owner
    dialog._sdlxliff_autogen_file_path = None
    dialog._sdlxliff_autogen_manual_entries = [existing_entry, missing_entry]
    dialog._book_index = -1
    dialog._book_entries = []

    stats = dialog._regenerate_manual_review_sidecars_for_refresh_scan()

    assert stats["total"] == 1
    assert owner.entries == [missing_entry]

    # Once the second output exists, an auto-refresh must not invoke the
    # bulk generator at all (and therefore cannot flash skipped progress).
    (tmp_path / "response_chapter0002.html").write_text(
        "<html><body><p>Now being edited too.</p></body></html>",
        encoding="utf-8",
    )
    owner.entries.clear()
    assert dialog._regenerate_manual_review_sidecars_for_refresh_scan() is None
    assert owner.entries == []


def test_sdlxliff_refresh_discards_scan_result_already_integrated_by_editor_save():
    dialog = SDLXLIFFReviewDialog.__new__(SDLXLIFFReviewDialog)
    dialog._review_refresh_scan_token = 7
    dialog._review_refresh_scan_running = True
    dialog._review_refresh_scan_requested = False
    dialog._last_review_signature = ("saved-sidecar",)
    dialog._last_autogen_signature = ("saved-output",)
    dialog._last_machine_translation_signature = ("mt",)
    dialog._tooltip_translation_running = False
    dialog._queue_stop_refresh_button_animation = lambda _delay: None

    reload_attempts = []
    dialog._queue_async_review_piece_reload = lambda **_kwargs: reload_attempts.append(True)
    dialog.refresh_review_data = lambda **_kwargs: reload_attempts.append(True)

    dialog._apply_review_refresh_scan(
        7,
        {
            "force": False,
            "review_signature": ("saved-sidecar",),
            "machine_translation_signature": ("mt",),
            "autogen_signature": ("saved-output",),
            # These flags were computed against the scan's older baseline.
            "sidecar_changed": True,
            "autogen_changed": True,
            "sidecars_generated": False,
            "machine_translation_changed": False,
            "stats": None,
            "error": "",
        },
    )

    assert reload_attempts == []
    assert dialog._review_refresh_scan_running is False


def test_retranslation_sdlxliff_generation_skips_current_sidecar_even_with_overwrite(tmp_path, monkeypatch):
    source = tmp_path / "chapter0001.xhtml"
    output = tmp_path / "response_chapter0001.html"
    sidecar_dir = tmp_path / "SDLXLIFF"
    sidecar = sidecar_dir / "response_chapter0001.html.sdlxliff"
    source.write_text("<h1>Source Title</h1><p>Source body.</p>", encoding="utf-8")
    output.write_text("<h1>Target Title</h1><p>Target body.</p>", encoding="utf-8")
    sidecar_dir.mkdir()
    sidecar.write_text("existing sidecar must not be rewritten", encoding="utf-8")
    old_ns = 1_700_000_000_000_000_000
    new_ns = old_ns + 5_000_000_000
    os.utime(output, ns=(old_ns, old_ns))
    os.utime(sidecar, ns=(new_ns, new_ns))
    progress = {
        "chapters": {
            "1": {
                "actual_num": 1,
                "status": "completed",
                "output_file": output.name,
                "original_basename": source.name,
            }
        }
    }
    monkeypatch.setenv("OUTPUT_SDLXLIFF", "0")
    mixin = RetranslationMixin.__new__(RetranslationMixin)
    progress_events = []

    stats = mixin._generate_sdlxliff_sidecars_from_completed_entries(
        str(tmp_path),
        progress_data=progress,
        overwrite=True,
        progress_callback=progress_events.append,
    )

    assert stats["created"] == 0
    assert stats["skipped"] == 1
    assert sidecar.read_text(encoding="utf-8") == "existing sidecar must not be rewritten"
    assert [event["stage"] for event in progress_events] == ["start", "checking", "skipped", "finished"]
    assert os.environ["OUTPUT_SDLXLIFF"] == "0"


def test_retranslation_sdlxliff_manifest_reuses_unchanged_output_after_newer_touch(tmp_path, monkeypatch):
    source = tmp_path / "chapter0001.xhtml"
    output = tmp_path / "response_chapter0001.html"
    source.write_text("<h1>Source Title</h1><p>Source body.</p>", encoding="utf-8")
    output.write_text("<h1>Target Title</h1><p>Target body.</p>", encoding="utf-8")
    progress = {
        "chapters": {
            "1": {
                "actual_num": 1,
                "status": "completed",
                "output_file": output.name,
                "original_basename": source.name,
            }
        }
    }
    monkeypatch.setenv("OUTPUT_SDLXLIFF", "0")
    mixin = RetranslationMixin.__new__(RetranslationMixin)

    first_stats = mixin._generate_sdlxliff_sidecars_from_completed_entries(
        str(tmp_path),
        progress_data=progress,
    )
    sidecar = tmp_path / "SDLXLIFF" / f"{output.name}.sdlxliff"
    manifest_path = tmp_path / "SDLXLIFF" / "sdlxliff_manifest.json"
    first_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    first_record = first_manifest["entries"]["chapter0001"]
    sidecar_payload = sidecar.read_bytes()

    newer_ns = max(sidecar.stat().st_mtime_ns, output.stat().st_mtime_ns) + 5_000_000_000
    os.utime(output, ns=(newer_ns, newer_ns))
    assert output.stat().st_mtime_ns > sidecar.stat().st_mtime_ns

    second_stats = mixin._generate_sdlxliff_sidecars_from_completed_entries(
        str(tmp_path),
        progress_data=progress,
        overwrite=True,
    )
    refreshed_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    refreshed_record = refreshed_manifest["entries"]["chapter0001"]

    assert first_stats["created"] == 1
    assert second_stats["created"] == 0
    assert second_stats["skipped"] == 1
    assert sidecar.read_bytes() == sidecar_payload
    assert refreshed_record["output_sha256"] == first_record["output_sha256"]
    assert refreshed_record["output_mtime_ns"] == newer_ns


def test_sdlxliff_review_stale_scan_ignores_timestamp_only_output_change_with_manifest(tmp_path, monkeypatch):
    source = tmp_path / "chapter0001.xhtml"
    output = tmp_path / "response_chapter0001.html"
    source.write_text("<h1>Source Title</h1>", encoding="utf-8")
    output.write_text("<h1>Target Title</h1>", encoding="utf-8")
    progress = {
        "chapters": {
            "1": {
                "status": "completed",
                "output_file": output.name,
                "original_basename": source.name,
            }
        }
    }
    (tmp_path / "translation_progress.json").write_text(
        json.dumps(progress),
        encoding="utf-8",
    )
    monkeypatch.setenv("OUTPUT_SDLXLIFF", "0")
    mixin = RetranslationMixin.__new__(RetranslationMixin)
    mixin._generate_sdlxliff_sidecars_from_completed_entries(
        str(tmp_path),
        progress_data=progress,
    )
    sidecar = tmp_path / "SDLXLIFF" / f"{output.name}.sdlxliff"
    newer_ns = max(sidecar.stat().st_mtime_ns, output.stat().st_mtime_ns) + 5_000_000_000
    os.utime(output, ns=(newer_ns, newer_ns))

    dialog = SDLXLIFFReviewDialog.__new__(SDLXLIFFReviewDialog)
    dialog.output_dir = str(tmp_path)
    dialog._book_entries = []
    dialog._book_index = 0
    signature = dialog._current_review_autogen_signature()

    assert dialog._stale_review_sidecar_outputs(str(tmp_path), signature) == []
    manifest = json.loads(
        (tmp_path / "SDLXLIFF" / "sdlxliff_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["entries"]["chapter0001"]["output_mtime_ns"] == newer_ns


def test_retranslation_sdlxliff_manifest_detects_content_change_even_when_sidecar_is_newer(tmp_path, monkeypatch):
    source = tmp_path / "chapter0001.xhtml"
    output = tmp_path / "response_chapter0001.html"
    source.write_text("<h1>Source Title</h1><p>Source body.</p>", encoding="utf-8")
    output.write_text("<h1>Old Target</h1>", encoding="utf-8")
    progress = {
        "chapters": {
            "1": {
                "status": "completed",
                "output_file": output.name,
                "original_basename": source.name,
            }
        }
    }
    monkeypatch.setenv("OUTPUT_SDLXLIFF", "0")
    mixin = RetranslationMixin.__new__(RetranslationMixin)

    mixin._generate_sdlxliff_sidecars_from_completed_entries(
        str(tmp_path),
        progress_data=progress,
    )
    sidecar = tmp_path / "SDLXLIFF" / f"{output.name}.sdlxliff"
    output.write_text("<h1>New Target</h1>", encoding="utf-8")
    future_ns = output.stat().st_mtime_ns + 5_000_000_000
    os.utime(sidecar, ns=(future_ns, future_ns))

    stats = mixin._generate_sdlxliff_sidecars_from_completed_entries(
        str(tmp_path),
        progress_data=progress,
        overwrite=True,
    )
    dialog = SDLXLIFFReviewDialog.__new__(SDLXLIFFReviewDialog)
    _source_html, target_html = dialog._read_sdlxliff_html_pair(str(sidecar))

    assert sidecar.stat().st_mtime_ns > output.stat().st_mtime_ns
    assert stats["created"] == 1
    assert "New Target" in target_html
    assert "Old Target" not in target_html


def test_retranslation_sdlxliff_missing_manifest_uses_legacy_mtime_freshness(tmp_path):
    output = tmp_path / "response_chapter0001.html"
    sidecar_dir = tmp_path / "SDLXLIFF"
    sidecar = sidecar_dir / f"{output.name}.sdlxliff"
    output.write_text("<h1>Target</h1>", encoding="utf-8")
    sidecar_dir.mkdir()
    sidecar.write_text("legacy sidecar", encoding="utf-8")
    mixin = RetranslationMixin.__new__(RetranslationMixin)
    old_ns = 1_700_000_000_000_000_000
    new_ns = old_ns + 5_000_000_000

    os.utime(output, ns=(old_ns, old_ns))
    os.utime(sidecar, ns=(new_ns, new_ns))
    assert not (sidecar_dir / "sdlxliff_manifest.json").exists()
    assert mixin._sdlxliff_sidecar_current_for_output(str(sidecar), str(output)) is True

    os.utime(output, ns=(new_ns, new_ns))
    os.utime(sidecar, ns=(old_ns, old_ns))
    assert mixin._sdlxliff_sidecar_current_for_output(str(sidecar), str(output)) is False


def test_retranslation_sdlxliff_manifest_detects_changed_source_html(tmp_path, monkeypatch):
    source = tmp_path / "chapter0001.xhtml"
    output = tmp_path / "response_chapter0001.html"
    source.write_text("<h1>Old Source</h1>", encoding="utf-8")
    output.write_text("<h1>Target</h1>", encoding="utf-8")
    progress = {
        "chapters": {
            "1": {
                "status": "completed",
                "output_file": output.name,
                "original_basename": source.name,
            }
        }
    }
    monkeypatch.setenv("OUTPUT_SDLXLIFF", "0")
    mixin = RetranslationMixin.__new__(RetranslationMixin)

    mixin._generate_sdlxliff_sidecars_from_completed_entries(
        str(tmp_path),
        progress_data=progress,
    )
    sidecar = tmp_path / "SDLXLIFF" / f"{output.name}.sdlxliff"
    source.write_text("<h1>New Source</h1>", encoding="utf-8")
    future_ns = max(source.stat().st_mtime_ns, sidecar.stat().st_mtime_ns) + 5_000_000_000
    os.utime(sidecar, ns=(future_ns, future_ns))

    stats = mixin._generate_sdlxliff_sidecars_from_completed_entries(
        str(tmp_path),
        progress_data=progress,
        overwrite=True,
    )
    dialog = SDLXLIFFReviewDialog.__new__(SDLXLIFFReviewDialog)
    source_html, _target_html = dialog._read_sdlxliff_html_pair(str(sidecar))

    assert stats["created"] == 1
    assert "New Source" in source_html
    assert "Old Source" not in source_html


def test_retranslation_sdlxliff_generation_reports_missing_source_counts(tmp_path, monkeypatch):
    output = tmp_path / "response_chapter0001.html"
    output.write_text("<h1>Target Title</h1><p>Target body.</p>", encoding="utf-8")
    progress = {
        "chapters": {
            "1": {
                "actual_num": 1,
                "status": "completed",
                "output_file": output.name,
                "original_basename": "missing_source.xhtml",
            }
        }
    }
    monkeypatch.setenv("OUTPUT_SDLXLIFF", "0")
    mixin = RetranslationMixin.__new__(RetranslationMixin)
    progress_events = []

    stats = mixin._generate_sdlxliff_sidecars_from_completed_entries(
        str(tmp_path),
        progress_data=progress,
        progress_callback=progress_events.append,
    )

    assert stats["total"] == 1
    assert stats["considered"] == 1
    assert stats["created"] == 0
    assert stats["missing_source"] == 1
    assert [event["stage"] for event in progress_events] == ["start", "checking", "missing_source", "finished"]

    dialog = SDLXLIFFReviewDialog.__new__(SDLXLIFFReviewDialog)
    summary = dialog._review_generation_summary(stats)
    assert "Generated 0/1 SDLXLIFF sidecar(s)" in summary
    assert "missing source 1" in summary


def test_retranslation_autogenerated_sdlxliff_prefers_source_epub_raw(tmp_path, monkeypatch):
    source_epub = tmp_path / "raw.epub"
    output = tmp_path / "response_chapter0001.html"
    misleading_local = tmp_path / "chapter0001.xhtml"
    raw_source = "<h1>원본 제목</h1><p>원본 문장입니다.</p>"
    translated = "<h1>Translated Title</h1><p>Translated body.</p>"

    with zipfile.ZipFile(source_epub, "w") as zf:
        zf.writestr("OEBPS/chapter0001.xhtml", raw_source)
    (tmp_path / "source_epub.txt").write_text(str(source_epub), encoding="utf-8")
    output.write_text(translated, encoding="utf-8")
    misleading_local.write_text(translated, encoding="utf-8")
    progress = {
        "chapters": {
            "1": {
                "actual_num": 1,
                "status": "completed",
                "output_file": output.name,
                "original_basename": "chapter0001.xhtml",
            }
        }
    }
    monkeypatch.setenv("OUTPUT_SDLXLIFF", "0")
    mixin = RetranslationMixin.__new__(RetranslationMixin)

    stats = mixin._generate_sdlxliff_sidecars_from_completed_entries(
        str(tmp_path),
        progress_data=progress,
    )

    sidecar = tmp_path / "SDLXLIFF" / "response_chapter0001.html.sdlxliff"
    assert stats["created"] == 1

    dialog = SDLXLIFFReviewDialog.__new__(SDLXLIFFReviewDialog)
    source_html, target_html = dialog._read_sdlxliff_html_pair(str(sidecar))

    assert "원본 제목" in source_html
    assert "원본 문장입니다" in source_html
    assert "Translated Title" not in source_html
    assert "Translated Title" in target_html


def test_retranslation_autogenerated_sdlxliff_falls_back_to_current_input_epub_name(tmp_path, monkeypatch):
    output_dir = tmp_path / "Moved Novel"
    moved_dir = tmp_path / "new location"
    output_dir.mkdir()
    moved_dir.mkdir()
    moved_epub = moved_dir / "Moved Novel.epub"
    output = output_dir / "response_chapter0001.html"
    raw_source = "<h1>Moved source title</h1><p>Moved source body.</p>"
    translated = "<h1>Translated Title</h1><p>Translated body.</p>"

    with zipfile.ZipFile(moved_epub, "w") as zf:
        zf.writestr("OEBPS/chapter0001.xhtml", raw_source)
    (output_dir / "source_epub.txt").write_text(str(tmp_path / "old location" / "Moved Novel.epub"), encoding="utf-8")
    output.write_text(translated, encoding="utf-8")
    progress = {
        "chapters": {
            "1": {
                "actual_num": 1,
                "status": "completed",
                "output_file": output.name,
                "original_basename": "chapter0001.xhtml",
            }
        }
    }
    monkeypatch.setenv("OUTPUT_SDLXLIFF", "0")
    mixin = RetranslationMixin.__new__(RetranslationMixin)
    mixin.selected_files = [str(moved_epub)]
    mixin.config = {"last_input_files": [str(tmp_path / "wrong.epub")]}

    stats = mixin._generate_sdlxliff_sidecars_from_completed_entries(
        str(output_dir),
        progress_data=progress,
    )

    sidecar = output_dir / "SDLXLIFF" / "response_chapter0001.html.sdlxliff"
    assert stats["created"] == 1

    dialog = SDLXLIFFReviewDialog.__new__(SDLXLIFFReviewDialog)
    source_html, target_html = dialog._read_sdlxliff_html_pair(str(sidecar))

    assert "Moved source title" in source_html
    assert "Moved source body." in source_html
    assert "Translated Title" not in source_html
    assert "Translated Title" in target_html
    assert (output_dir / "source_epub.txt").read_text(encoding="utf-8") == str(moved_epub.resolve())


def test_retranslation_autogenerated_sdlxliff_reads_extracted_epub_source_dir(tmp_path, monkeypatch):
    input_root = tmp_path / "input" / "final fantasy vi the novel"
    output_dir = tmp_path / "output" / "final fantasy vi the novel"
    source_chapter_dir = input_root / "EPUB"
    source_chapter_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)
    raw_source = "<html><body><h1>Raw FFVI Title</h1><p>Raw FFVI body.</p></body></html>"
    translated = "<html><body><h1>Translated Title</h1><p>Translated body.</p></body></html>"
    (input_root / "content.opf").write_text(
        """<?xml version="1.0" encoding="utf-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="2.0">
  <manifest><item id="p1" href="EPUB/piece_0001.xhtml" media-type="application/xhtml+xml"/></manifest>
  <spine><itemref idref="p1"/></spine>
</package>
""",
        encoding="utf-8",
    )
    (source_chapter_dir / "piece_0001.xhtml").write_text(raw_source, encoding="utf-8")
    output = output_dir / "response_piece_0001.html"
    output.write_text(translated, encoding="utf-8")
    progress = {
        "chapters": {
            "1": {
                "actual_num": 1,
                "status": "completed",
                "output_file": output.name,
            }
        }
    }
    monkeypatch.setenv("OUTPUT_SDLXLIFF", "0")
    mixin = RetranslationMixin.__new__(RetranslationMixin)
    mixin.selected_files = [str(input_root)]
    mixin.config = {}

    stats = mixin._generate_sdlxliff_sidecars_from_completed_entries(
        str(output_dir),
        progress_data=progress,
    )

    sidecar = output_dir / "SDLXLIFF" / "response_piece_0001.html.sdlxliff"
    assert stats["created"] == 1
    assert sidecar.is_file()
    assert (output_dir / "source_epub.txt").read_text(encoding="utf-8") == str(input_root.resolve())

    dialog = SDLXLIFFReviewDialog.__new__(SDLXLIFFReviewDialog)
    source_html, target_html = dialog._read_sdlxliff_html_pair(str(sidecar))
    assert "Raw FFVI Title" in source_html
    assert "Raw FFVI body." in source_html
    assert "Translated Title" not in source_html
    assert "Translated Title" in target_html


def test_retranslation_autogenerated_sdlxliff_uses_single_selected_extracted_epub_when_names_differ(tmp_path, monkeypatch):
    input_root = tmp_path / "final fantasy vi the novel"
    output_dir = tmp_path / "Credits OMORIO"
    source_chapter_dir = input_root / "EPUB"
    source_chapter_dir.mkdir(parents=True)
    output_dir.mkdir()
    raw_source = "<html><body><h1>Raw Different Folder Title</h1><p>Raw different folder body.</p></body></html>"
    translated = "<html><body><h1>Translated Title</h1><p>Translated body.</p></body></html>"
    (input_root / "content.opf").write_text(
        """<?xml version="1.0" encoding="utf-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="2.0">
  <manifest><item id="p1" href="EPUB/piece_0001.xhtml" media-type="application/xhtml+xml"/></manifest>
  <spine><itemref idref="p1"/></spine>
</package>
""",
        encoding="utf-8",
    )
    (source_chapter_dir / "piece_0001.xhtml").write_text(raw_source, encoding="utf-8")
    output = output_dir / "response_piece_0001.html"
    output.write_text(translated, encoding="utf-8")
    progress = {
        "chapters": {
            "1": {
                "actual_num": 1,
                "status": "completed",
                "output_file": output.name,
            }
        }
    }
    monkeypatch.setenv("OUTPUT_SDLXLIFF", "0")
    mixin = RetranslationMixin.__new__(RetranslationMixin)
    mixin.selected_files = [str(input_root)]
    mixin.config = {}

    stats = mixin._generate_sdlxliff_sidecars_from_completed_entries(
        str(output_dir),
        progress_data=progress,
    )

    sidecar = output_dir / "SDLXLIFF" / "response_piece_0001.html.sdlxliff"
    assert stats["created"] == 1
    assert (output_dir / "source_epub.txt").read_text(encoding="utf-8") == str(input_root.resolve())

    dialog = SDLXLIFFReviewDialog.__new__(SDLXLIFFReviewDialog)
    source_html, target_html = dialog._read_sdlxliff_html_pair(str(sidecar))
    assert "Raw Different Folder Title" in source_html
    assert "Raw different folder body." in source_html
    assert "Translated Title" not in source_html
    assert "Translated Title" in target_html


def test_sdlxliff_review_spine_positions_include_relative_epub_paths(tmp_path):
    output_dir = tmp_path / "final fantasy vi the novel"
    epub_dir = output_dir / "EPUB"
    epub_dir.mkdir(parents=True)
    (output_dir / "content.opf").write_text(
        """<?xml version="1.0" encoding="utf-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="2.0">
  <manifest><item id="p1" href="EPUB/piece_0001.xhtml" media-type="application/xhtml+xml"/></manifest>
  <spine><itemref idref="p1"/></spine>
</package>
""",
        encoding="utf-8",
    )
    dialog = SDLXLIFFReviewDialog.__new__(SDLXLIFFReviewDialog)
    dialog.output_dir = str(output_dir)

    positions = dialog._read_spine_positions()

    assert positions["epub/piece_0001.xhtml"] == 0
    assert positions["epub/piece_0001"] == 0
    assert positions["piece_0001.xhtml"] == 0
    assert positions["piece_0001"] == 0


def test_manual_sidecar_generation_uses_source_content_opf_order(tmp_path, monkeypatch):
    output_dir = tmp_path / "ordered-book"
    output_dir.mkdir()
    epub_path = tmp_path / "ordered-book.epub"
    opf = """<?xml version="1.0" encoding="utf-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0">
  <manifest>
    <item id="cover" href="Text/cover.xhtml" media-type="application/xhtml+xml"/>
    <item id="c2" href="Text/chapter0002.xhtml" media-type="application/xhtml+xml"/>
    <item id="c1" href="Text/chapter0001.xhtml" media-type="application/xhtml+xml"/>
  </manifest>
  <spine><itemref idref="cover"/><itemref idref="c2"/><itemref idref="c1"/></spine>
</package>"""
    with zipfile.ZipFile(epub_path, "w") as epub:
        epub.writestr("OEBPS/content.opf", opf)
        epub.writestr("OEBPS/Text/cover.xhtml", "<p>Cover text</p>")
        epub.writestr("OEBPS/Text/chapter0002.xhtml", "<p>Second in the spine</p>")
        epub.writestr("OEBPS/Text/chapter0001.xhtml", "<p>Third in the spine</p>")

    entries = [
        {
            "status": "not_translated",
            "original_basename": "chapter0001.xhtml",
            "original_filename": "Text/chapter0001.xhtml",
            "output_file": "response_chapter0001.html",
        },
        {
            "status": "not_translated",
            "original_basename": "cover.xhtml",
            "original_filename": "Text/cover.xhtml",
            "output_file": "response_cover.html",
            "is_special": True,
        },
        {
            "status": "not_translated",
            "original_basename": "chapter0002.xhtml",
            "original_filename": "Text/chapter0002.xhtml",
            "output_file": "response_chapter0002.html",
        },
    ]
    events = []
    monkeypatch.setenv("OUTPUT_SDLXLIFF", "0")
    monkeypatch.setenv("RETAIN_SOURCE_EXTENSION", "0")
    mixin = RetranslationMixin.__new__(RetranslationMixin)
    mixin.config = {}

    stats = mixin._generate_sdlxliff_sidecars_from_untranslated_entries(
        str(output_dir),
        entries,
        file_path=str(epub_path),
        progress_callback=events.append,
    )

    assert stats["created"] == 3
    created = [event for event in events if event["stage"] == "created"]
    assert [event["output_name"] for event in created] == [
        "response_cover.html",
        "response_chapter0002.html",
        "response_chapter0001.html",
    ]
    assert [event["opf_position"] for event in created] == [0, 1, 2]


def test_sdlxliff_review_uses_opf_over_stale_progress_order_and_labels_special_as_zero(tmp_path):
    (tmp_path / "content.opf").write_text(
        """<?xml version="1.0" encoding="utf-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0">
  <manifest>
    <item id="info" href="Text/info.xhtml" media-type="application/xhtml+xml"/>
    <item id="c2" href="Text/chapter0002.xhtml" media-type="application/xhtml+xml"/>
    <item id="c1" href="Text/chapter0001.xhtml" media-type="application/xhtml+xml"/>
  </manifest>
  <spine><itemref idref="info"/><itemref idref="c2"/><itemref idref="c1"/></spine>
</package>""",
        encoding="utf-8",
    )
    progress = {"chapters": {}}
    specs = [
        ("info.xhtml", "response_info.html", 77, 2),
        ("chapter0002.xhtml", "response_chapter0002.html", 2, 1),
        ("chapter0001.xhtml", "response_chapter0001.html", 1, 0),
    ]
    for key, (source_name, output_name, actual_num, stale_position) in enumerate(specs):
        _shared_write_html_sdlxliff_sidecar(
            str(tmp_path),
            output_name,
            {"original_basename": source_name},
            f"<html><body><p>{source_name}</p></body></html>",
            f"<html><body><p>{source_name}</p></body></html>",
            raise_errors=True,
        )
        progress["chapters"][str(key)] = {
            "status": "completed",
            "output_file": output_name,
            "original_basename": source_name,
            "actual_num": actual_num,
            "opf_position": stale_position,
        }
    (tmp_path / "translation_progress.json").write_text(
        json.dumps(progress),
        encoding="utf-8",
    )
    dialog = SDLXLIFFReviewDialog.__new__(SDLXLIFFReviewDialog)
    dialog.output_dir = str(tmp_path)
    dialog.current_path = ""

    pieces = dialog._load_pieces()

    assert [piece["output_name"] for piece in pieces] == [
        "response_info.html",
        "response_chapter0002.html",
        "response_chapter0001.html",
    ]
    assert [piece["opf_position"] for piece in pieces] == [0, 1, 2]
    assert pieces[0]["chapter_num"] == 0
    assert pieces[0]["review_label"] == "[001] Ch.000 |"


def test_sdlxliff_review_deduplicates_retained_and_response_named_sidecars(tmp_path):
    retained_name = "chapter0001.xhtml"
    response_name = "response_chapter0001.html"
    for output_name in (retained_name, response_name):
        _shared_write_html_sdlxliff_sidecar(
            str(tmp_path),
            output_name,
            {"original_basename": retained_name},
            "<html><body><p>Source</p></body></html>",
            "<html><body><p>Translated</p></body></html>",
            raise_errors=True,
        )
    (tmp_path / "translation_progress.json").write_text(
        json.dumps({
            "chapters": {
                "1": {
                    "status": "completed",
                    "output_file": response_name,
                    "original_basename": retained_name,
                    "actual_num": 1,
                }
            }
        }),
        encoding="utf-8",
    )
    dialog = SDLXLIFFReviewDialog.__new__(SDLXLIFFReviewDialog)
    dialog.output_dir = str(tmp_path)
    dialog.current_path = ""

    pieces = dialog._load_pieces()

    assert len(pieces) == 1
    assert pieces[0]["output_name"] == response_name


def test_manual_sidecar_generation_recognizes_alternate_filename_mode(tmp_path, monkeypatch):
    source_name = "chapter0001.xhtml"
    (tmp_path / source_name).write_text("<html><body><p>Source</p></body></html>", encoding="utf-8")
    _shared_write_html_sdlxliff_sidecar(
        str(tmp_path),
        source_name,
        {"original_basename": source_name},
        "<html><body><p>Source</p></body></html>",
        "<html><body><p></p></body></html>",
        raise_errors=True,
        manual_untranslated=True,
    )
    monkeypatch.setenv("RETAIN_SOURCE_EXTENSION", "0")
    mixin = RetranslationMixin.__new__(RetranslationMixin)
    mixin.config = {}

    stats = mixin._generate_sdlxliff_sidecars_from_untranslated_entries(
        str(tmp_path),
        [{
            "status": "not_translated",
            "original_basename": source_name,
            "output_file": "response_chapter0001.html",
        }],
    )

    assert stats["created"] == 0
    assert stats["skipped"] == 1
    assert not (tmp_path / "SDLXLIFF" / "response_chapter0001.html.sdlxliff").exists()


def test_sdlxliff_notepad_edit_updates_div_u_in_place():
    dialog = SDLXLIFFReviewDialog.__new__(SDLXLIFFReviewDialog)
    piece = {
        "target_html": '<html><body><div class="u">Old output</div></body></html>',
    }
    row = {
        "row_index": 0,
        "source_tag": "p",
        "target_tag": "p",
        "source_tag_label": "p",
        "target_index": 0,
    }

    edited = dialog._target_html_with_edit(piece, row, "New output")
    soup = BeautifulSoup(edited, "html.parser")

    assert soup.find("div", class_="u").get_text(strip=True) == "New output"
    assert soup.find("p") is None


def test_sdlxliff_review_treats_div_u_source_blocks_as_paragraph_units(tmp_path):
    source = """
<html><body>
  <div class="u">The Girl With the Magitek Armor</div>
  <div class="u"></div>
  <div class="u"><em>Final Fantasy 6- The Novel</em></div>
  <div class="u">Written by me: Celes Chere</div>
</body></html>
"""
    target = """
<html><body>
  <p>Translated armor title</p>
  <p><em>Translated novel subtitle</em></p>
  <p>Translated author line</p>
</body></html>
"""
    _write_html_sdlxliff_sidecar(
        str(tmp_path),
        "response_piece_0002.html",
        {"original_basename": "piece_0002.xhtml"},
        source,
        target,
    )
    sidecar = tmp_path / "SDLXLIFF" / "response_piece_0002.html.sdlxliff"
    dialog = SDLXLIFFReviewDialog.__new__(SDLXLIFFReviewDialog)

    piece = dialog._build_piece(str(sidecar), 0, {"output_name": "response_piece_0002.html", "display_position": 2})

    assert piece["source_count"] == 3
    assert piece["target_count"] == 3
    assert [row["source_tag"] for row in piece["rows"]] == ["p", "p", "p"]
    assert [row["source_tag_label"] for row in piece["rows"]] == ["p", "p(2)", "p(3)"]
    assert [row["target_tag_label"] for row in piece["rows"]] == ["p", "p(2)", "p(3)"]
    assert [row["source"] for row in piece["rows"]] == [
        "The Girl With the Magitek Armor",
        "Final Fantasy 6- The Novel",
        "Written by me: Celes Chere",
    ]
    assert piece["red_count"] == 0


def test_sdlxliff_review_numbered_tag_label_text_uses_compact_empty_labels():
    assert SDLXLIFFReviewDialog._tag_label_text("p", "p", "p(2)", "p(2)") == "p(2)"
    assert SDLXLIFFReviewDialog._tag_label_text("h1", "h2", "h1", "h2") == "h1 -> h2"
    assert SDLXLIFFReviewDialog._tag_label_text("p", "", "p(3)", "") == "Empty(3)"
    assert SDLXLIFFReviewDialog._tag_label_text("", "p", "", "p(4)") == "Added(4)"
    assert SDLXLIFFReviewDialog._tag_label_rich_text("p(2)") == 'p<span style="font-size: 8pt;">(2)</span>'
    assert SDLXLIFFReviewDialog._tag_label_rich_text("Empty(33)") == 'Empty<span style="font-size: 8pt;">(33)</span>'


def test_sdlxliff_review_paragraphs_and_list_items_share_ordinal_counter():
    units = [
        {"tag": "p", "text": "One"},
        {"tag": "p", "text": "Two"},
        {"tag": "li", "text": "Three"},
        {"tag": "p", "text": "Four"},
        {"tag": "h1", "text": "Heading"},
        {"tag": "li", "text": "Five"},
    ]

    annotated = SDLXLIFFReviewDialog._annotate_review_tag_labels(units)

    assert [unit["tag_label"] for unit in annotated] == [
        "p", "p(2)", "li(3)", "p(4)", "h1", "li(5)"
    ]
    assert [unit["tag_ordinal"] for unit in annotated] == [1, 2, 3, 4, 1, 5]


def test_sdlxliff_review_tag_label_font_shrinks_for_numbered_p_to_li_pair():
    short_size = SDLXLIFFReviewDialog._tag_label_font_point_size("p(2)")
    converted_size = SDLXLIFFReviewDialog._tag_label_font_point_size(
        "p(11) -> li(11)"
    )

    assert short_size == SDLXLIFFReviewDialog.REVIEW_TAG_LABEL_MAX_FONT_PT
    assert converted_size < short_size
    assert converted_size >= SDLXLIFFReviewDialog.REVIEW_TAG_LABEL_MIN_FONT_PT


def test_sdlxliff_review_regenerates_sidecar_when_source_column_is_empty(tmp_path, monkeypatch):
    output_dir = tmp_path / "Moved Novel"
    moved_dir = tmp_path / "new location"
    sidecar_dir = output_dir / "SDLXLIFF"
    output_dir.mkdir()
    moved_dir.mkdir()
    sidecar_dir.mkdir()
    moved_epub = moved_dir / "Moved Novel.epub"
    output = output_dir / "response_chapter0001.html"
    sidecar = sidecar_dir / "response_chapter0001.html.sdlxliff"
    raw_source = "<h1>Real source title</h1><p>Real source body.</p>"
    translated = "<h1>Translated Title</h1><p>Translated body.</p>"

    with zipfile.ZipFile(moved_epub, "w") as zf:
        zf.writestr("OEBPS/chapter0001.xhtml", raw_source)
    (output_dir / "source_epub.txt").write_text(str(tmp_path / "old location" / "Moved Novel.epub"), encoding="utf-8")
    output.write_text(translated, encoding="utf-8")
    (output_dir / "translation_progress.json").write_text(
        json.dumps(
            {
                "chapters": {
                    "1": {
                        "actual_num": 1,
                        "status": "completed",
                        "output_file": output.name,
                        "original_basename": "chapter0001.xhtml",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    sidecar.write_text(
        f"""<?xml version="1.0" encoding="utf-8"?>
<xliff xmlns="urn:oasis:names:tc:xliff:document:1.2" version="1.2">
  <file original="chapter0001.xhtml" source-language="ko-KR" target-language="en-US">
    <body>
      <trans-unit id="html">
        <source><![CDATA[<html><body><h1></h1><p></p></body></html>]]></source>
        <target><![CDATA[{translated}]]></target>
      </trans-unit>
    </body>
  </file>
</xliff>
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("OUTPUT_SDLXLIFF", "0")
    mixin = RetranslationMixin.__new__(RetranslationMixin)
    mixin.selected_files = [str(moved_epub)]
    mixin.config = {}
    dialog = SDLXLIFFReviewDialog.__new__(SDLXLIFFReviewDialog)
    dialog.output_dir = str(output_dir)
    dialog._book_entries = []
    dialog._book_index = 0
    dialog._last_autogen_signature = dialog._current_review_autogen_signature()
    dialog._last_invalid_sidecar_regen_key = None
    dialog._sdlxliff_autogen_owner = mixin

    assert dialog._sdlxliff_sidecar_needs_source_regeneration(str(sidecar)) is True
    assert dialog._maybe_regenerate_review_sidecars(force=False) is True

    source_html, target_html = dialog._read_sdlxliff_html_pair(str(sidecar))
    assert "Real source title" in source_html
    assert "Real source body." in source_html
    assert "Translated Title" not in source_html
    assert "Translated Title" in target_html
    assert dialog._sdlxliff_sidecar_needs_source_regeneration(str(sidecar)) is False
    assert (output_dir / "source_epub.txt").read_text(encoding="utf-8") == str(moved_epub.resolve())


def test_sdlxliff_review_detects_sidecar_when_source_matches_target(tmp_path):
    sidecar = tmp_path / "response_chapter0001.html.sdlxliff"
    same_html = "<html><body><h1>Same Title</h1><p>Same body.</p></body></html>"
    sidecar.write_text(
        f"""<?xml version="1.0" encoding="utf-8"?>
<xliff xmlns="urn:oasis:names:tc:xliff:document:1.2" version="1.2">
  <file original="chapter0001.xhtml" source-language="ko-KR" target-language="en-US">
    <body>
      <trans-unit id="html">
        <source><![CDATA[{same_html}]]></source>
        <target><![CDATA[{same_html}]]></target>
      </trans-unit>
    </body>
  </file>
</xliff>
""",
        encoding="utf-8",
    )
    dialog = SDLXLIFFReviewDialog.__new__(SDLXLIFFReviewDialog)

    assert dialog._sdlxliff_sidecar_needs_source_regeneration(str(sidecar)) is True


def test_sdlxliff_review_filters_empty_sidecars_from_piece_list(tmp_path):
    sidecar_dir = tmp_path / "SDLXLIFF"
    sidecar_dir.mkdir()
    empty_sidecar = sidecar_dir / "response_cover.html.sdlxliff"
    text_sidecar = sidecar_dir / "response_chapter0001.html.sdlxliff"
    empty_sidecar.write_text(
        """<?xml version="1.0" encoding="utf-8"?>
<xliff xmlns="urn:oasis:names:tc:xliff:document:1.2" version="1.2">
  <file original="cover.xhtml" source-language="ko-KR" target-language="en-US">
    <body>
      <trans-unit id="html">
        <source><![CDATA[<html><body></body></html>]]></source>
        <target><![CDATA[<html><body></body></html>]]></target>
      </trans-unit>
    </body>
  </file>
</xliff>
""",
        encoding="utf-8",
    )
    text_sidecar.write_text(
        """<?xml version="1.0" encoding="utf-8"?>
<xliff xmlns="urn:oasis:names:tc:xliff:document:1.2" version="1.2">
  <file original="chapter0001.xhtml" source-language="ko-KR" target-language="en-US">
    <body>
      <trans-unit id="html">
        <source><![CDATA[<html><body><p>Source</p></body></html>]]></source>
        <target><![CDATA[<html><body><p>Target</p></body></html>]]></target>
      </trans-unit>
    </body>
  </file>
</xliff>
""",
        encoding="utf-8",
    )
    _shared_write_html_sdlxliff_sidecar(
        str(tmp_path),
        "response_chapter0002.html",
        {"original_basename": "chapter0002.xhtml"},
        "<html><body></body></html>",
        "<html><body></body></html>",
        raise_errors=True,
        manual_untranslated=True,
    )
    dialog = SDLXLIFFReviewDialog.__new__(SDLXLIFFReviewDialog)
    dialog.output_dir = str(tmp_path)
    dialog.current_path = ""

    pieces = dialog._load_pieces()

    assert [piece["name"] for piece in pieces] == [
        "response_chapter0001.html.sdlxliff",
    ]
    assert pieces[0]["source_count"] == 1
    assert pieces[0]["target_count"] == 1


def test_sdlxliff_review_autorefresh_regenerates_sidecar_from_changed_output(tmp_path, monkeypatch):
    source = tmp_path / "chapter0001.xhtml"
    output = tmp_path / "response_chapter0001.html"
    source.write_text("<h1>Source Title</h1><p>Source body.</p>", encoding="utf-8")
    output.write_text("<h1>Target Title</h1><p>Target body.</p>", encoding="utf-8")
    (tmp_path / "translation_progress.json").write_text(
        json.dumps(
            {
                "chapters": {
                    "1": {
                        "actual_num": 1,
                        "status": "completed",
                        "output_file": output.name,
                        "original_basename": source.name,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("OUTPUT_SDLXLIFF", "0")
    mixin = RetranslationMixin.__new__(RetranslationMixin)
    dialog = SDLXLIFFReviewDialog.__new__(SDLXLIFFReviewDialog)
    dialog.output_dir = str(tmp_path)
    dialog._book_entries = []
    dialog._book_index = 0
    dialog._last_autogen_signature = None
    dialog._sdlxliff_autogen_owner = mixin

    assert dialog._maybe_regenerate_review_sidecars(force=True) is True

    sidecar = tmp_path / "SDLXLIFF" / "response_chapter0001.html.sdlxliff"
    assert sidecar.is_file()
    output.write_text(
        "<h1>Target Title</h1><p>Target body.</p><p>Added output entry.</p>",
        encoding="utf-8",
    )

    assert dialog._maybe_regenerate_review_sidecars(force=False) is True
    piece = dialog._build_piece(str(sidecar), 0, {"output_name": output.name})

    assert piece["source_count"] == 2
    assert piece["target_count"] == 3
    assert piece["mismatch"] is True
    assert piece["rows"][-1]["target"] == "Added output entry."
    assert piece["rows"][-1]["source"] == ""


def test_sdlxliff_review_initial_scan_regenerates_stale_sidecar_from_output(tmp_path, monkeypatch):
    source = tmp_path / "chapter0001.xhtml"
    output = tmp_path / "response_chapter0001.html"
    sidecar_dir = tmp_path / "SDLXLIFF"
    sidecar = sidecar_dir / "response_chapter0001.html.sdlxliff"
    source.write_text("<h1>Source Title</h1><p>Source body.</p>", encoding="utf-8")
    output.write_text("<h1>Refined Title</h1><p>Refined body.</p>", encoding="utf-8")
    (tmp_path / "translation_progress.json").write_text(
        json.dumps(
            {
                "chapters": {
                    "1": {
                        "actual_num": 1,
                        "status": "completed",
                        "output_file": output.name,
                        "original_basename": source.name,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    sidecar_dir.mkdir()
    sidecar.write_text(
        """<?xml version="1.0" encoding="utf-8"?>
<xliff xmlns="urn:oasis:names:tc:xliff:document:1.2" version="1.2">
  <file original="chapter0001.xhtml" source-language="ko-KR" target-language="en-US">
    <body>
      <trans-unit id="html">
        <source><![CDATA[<h1>Source Title</h1><p>Source body.</p>]]></source>
        <target><![CDATA[<h1>Old Title</h1><p>Old body.</p>]]></target>
      </trans-unit>
    </body>
  </file>
</xliff>
""",
        encoding="utf-8",
    )
    old_ns = 1_700_000_000_000_000_000
    new_ns = old_ns + 5_000_000_000
    os.utime(sidecar, ns=(old_ns, old_ns))
    os.utime(output, ns=(new_ns, new_ns))
    monkeypatch.setenv("OUTPUT_SDLXLIFF", "0")
    mixin = RetranslationMixin.__new__(RetranslationMixin)
    dialog = SDLXLIFFReviewDialog.__new__(SDLXLIFFReviewDialog)
    dialog.output_dir = str(tmp_path)
    dialog._book_entries = []
    dialog._book_index = 0
    dialog._sdlxliff_autogen_owner = mixin

    result = dialog._build_review_refresh_scan_result(
        force=False,
        current_path=str(sidecar),
        last_review_signature=None,
        last_mt_signature=None,
        last_autogen_signature=None,
    )
    piece = dialog._build_piece(str(sidecar), 0, {"output_name": output.name})

    assert result["error"] == ""
    assert result["sidecars_generated"] is True
    assert piece["rows"][0]["target"] == "Refined Title"
    assert piece["rows"][1]["target"] == "Refined body."


def test_sdlxliff_review_manual_refresh_regenerates_current_sidecar(tmp_path, monkeypatch):
    source = tmp_path / "chapter0001.xhtml"
    output = tmp_path / "response_chapter0001.html"
    source.write_text("<h1>Source Title</h1><p>Source body.</p>", encoding="utf-8")
    output.write_text("<h1>Target Title</h1><p>Target body.</p>", encoding="utf-8")
    (tmp_path / "translation_progress.json").write_text(
        json.dumps(
            {
                "chapters": {
                    "1": {
                        "actual_num": 1,
                        "status": "completed",
                        "output_file": output.name,
                        "original_basename": source.name,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("OUTPUT_SDLXLIFF", "0")
    mixin = RetranslationMixin.__new__(RetranslationMixin)
    dialog = SDLXLIFFReviewDialog.__new__(SDLXLIFFReviewDialog)
    dialog.output_dir = str(tmp_path)
    dialog._book_entries = []
    dialog._book_index = 0
    dialog._sdlxliff_autogen_owner = mixin

    assert dialog._maybe_regenerate_review_sidecars(force=True) is True
    sidecar = tmp_path / "SDLXLIFF" / "response_chapter0001.html.sdlxliff"
    sidecar.write_text("manual refresh must replace this current sidecar", encoding="utf-8")
    future_ns = 1_900_000_000_000_000_000
    os.utime(sidecar, ns=(future_ns, future_ns))

    result = dialog._build_review_refresh_scan_result(
        force=True,
        current_path=str(sidecar),
        last_review_signature=dialog._current_review_signature(),
        last_mt_signature=dialog._current_machine_translation_signature(),
        last_autogen_signature=dialog._current_review_autogen_signature(),
    )
    piece = dialog._build_piece(str(sidecar), 0, {"output_name": output.name})

    assert result["error"] == ""
    assert result["sidecars_generated"] is True
    assert piece["rows"][0]["target"] == "Target Title"
    assert piece["rows"][1]["target"] == "Target body."


def test_sdlxliff_review_manual_refresh_ignores_single_file_open_scope(tmp_path, monkeypatch):
    source_one = tmp_path / "chapter0001.xhtml"
    source_two = tmp_path / "chapter0002.xhtml"
    output_one = tmp_path / "response_chapter0001.html"
    output_two = tmp_path / "response_chapter0002.html"
    source_one.write_text("<h1>Source One</h1><p>Source body one.</p>", encoding="utf-8")
    source_two.write_text("<h1>Source Two</h1><p>Source body two.</p>", encoding="utf-8")
    output_one.write_text("<h1>Target One</h1><p>Target body one.</p>", encoding="utf-8")
    output_two.write_text("<h1>Target Two</h1><p>Target body two.</p>", encoding="utf-8")
    (tmp_path / "translation_progress.json").write_text(
        json.dumps(
            {
                "chapters": {
                    "1": {
                        "actual_num": 1,
                        "status": "completed",
                        "output_file": output_one.name,
                        "original_basename": source_one.name,
                    },
                    "2": {
                        "actual_num": 2,
                        "status": "completed",
                        "output_file": output_two.name,
                        "original_basename": source_two.name,
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("OUTPUT_SDLXLIFF", "0")
    mixin = RetranslationMixin.__new__(RetranslationMixin)
    dialog = SDLXLIFFReviewDialog.__new__(SDLXLIFFReviewDialog)
    dialog.output_dir = str(tmp_path)
    dialog._book_entries = []
    dialog._book_index = 0
    dialog._sdlxliff_autogen_owner = mixin
    dialog._sdlxliff_autogen_output_files = [output_one.name]

    assert dialog._maybe_regenerate_review_sidecars(force=True) is True
    sidecar_one = tmp_path / "SDLXLIFF" / f"{output_one.name}.sdlxliff"
    sidecar_two = tmp_path / "SDLXLIFF" / f"{output_two.name}.sdlxliff"
    sidecar_one.write_text("manual refresh must replace current sidecar", encoding="utf-8")
    sidecar_two.write_text("manual refresh must also replace non-current sidecar", encoding="utf-8")

    result = dialog._build_review_refresh_scan_result(
        force=True,
        current_path=str(sidecar_one),
        last_review_signature=dialog._current_review_signature(),
        last_mt_signature=dialog._current_machine_translation_signature(),
        last_autogen_signature=dialog._current_review_autogen_signature(),
    )
    piece_one = dialog._build_piece(str(sidecar_one), 0, {"output_name": output_one.name})
    piece_two = dialog._build_piece(str(sidecar_two), 1, {"output_name": output_two.name})

    assert result["error"] == ""
    assert result["sidecars_generated"] is True
    assert result["stats"]["created"] == 2
    assert piece_one["rows"][0]["target"] == "Target One"
    assert piece_two["rows"][0]["target"] == "Target Two"


def test_sdlxliff_review_autorefresh_regenerates_deleted_sidecar_folder(tmp_path, monkeypatch):
    import shutil

    source = tmp_path / "chapter0001.xhtml"
    output = tmp_path / "response_chapter0001.html"
    source.write_text("<h1>Source Title</h1><p>Source body.</p>", encoding="utf-8")
    output.write_text("<h1>Target Title</h1><p>Target body.</p>", encoding="utf-8")
    (tmp_path / "translation_progress.json").write_text(
        json.dumps(
            {
                "chapters": {
                    "1": {
                        "actual_num": 1,
                        "status": "completed",
                        "output_file": output.name,
                        "original_basename": source.name,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("OUTPUT_SDLXLIFF", "0")
    mixin = RetranslationMixin.__new__(RetranslationMixin)
    dialog = SDLXLIFFReviewDialog.__new__(SDLXLIFFReviewDialog)
    dialog.output_dir = str(tmp_path)
    dialog._book_entries = []
    dialog._book_index = 0
    dialog._last_autogen_signature = None
    dialog._sdlxliff_autogen_owner = mixin

    assert dialog._maybe_regenerate_review_sidecars(force=True) is True
    sidecar_dir = tmp_path / "SDLXLIFF"
    sidecar = sidecar_dir / "response_chapter0001.html.sdlxliff"
    assert sidecar.is_file()

    dialog._last_autogen_signature = dialog._current_review_autogen_signature()
    shutil.rmtree(sidecar_dir)

    assert dialog._maybe_regenerate_review_sidecars(force=False) is True
    assert sidecar.is_file()


def test_sdlxliff_review_refresh_worker_regenerates_deleted_sidecar_folder(tmp_path, monkeypatch):
    import shutil

    source = tmp_path / "chapter0001.xhtml"
    output = tmp_path / "response_chapter0001.html"
    source.write_text("<h1>Source Title</h1><p>Source body.</p>", encoding="utf-8")
    output.write_text("<h1>Target Title</h1><p>Target body.</p>", encoding="utf-8")
    (tmp_path / "translation_progress.json").write_text(
        json.dumps(
            {
                "chapters": {
                    "1": {
                        "actual_num": 1,
                        "status": "completed",
                        "output_file": output.name,
                        "original_basename": source.name,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("OUTPUT_SDLXLIFF", "0")
    mixin = RetranslationMixin.__new__(RetranslationMixin)
    dialog = SDLXLIFFReviewDialog.__new__(SDLXLIFFReviewDialog)
    dialog.output_dir = str(tmp_path)
    dialog._book_entries = []
    dialog._book_index = 0
    dialog._last_autogen_signature = None
    dialog._sdlxliff_autogen_owner = mixin

    assert dialog._maybe_regenerate_review_sidecars(force=True) is True
    sidecar_dir = tmp_path / "SDLXLIFF"
    sidecar = sidecar_dir / "response_chapter0001.html.sdlxliff"
    last_review_signature = dialog._current_review_signature()
    last_mt_signature = dialog._current_machine_translation_signature()
    last_autogen_signature = dialog._current_review_autogen_signature()
    shutil.rmtree(sidecar_dir)

    result = dialog._build_review_refresh_scan_result(
        force=False,
        current_path=str(sidecar),
        last_review_signature=last_review_signature,
        last_mt_signature=last_mt_signature,
        last_autogen_signature=last_autogen_signature,
    )

    assert result["error"] == ""
    assert result["sidecars_generated"] is True
    assert result["sidecar_changed"] is True
    assert sidecar.is_file()


def test_sdlxliff_review_manual_validation_does_not_regenerate_current_sidecars(tmp_path, monkeypatch):
    source = tmp_path / "chapter0001.xhtml"
    output = tmp_path / "response_chapter0001.html"
    source.write_text("<h1>Source Title</h1><p>Source body.</p>", encoding="utf-8")
    output.write_text("<h1>Target Title</h1><p>Target body.</p>", encoding="utf-8")
    (tmp_path / "translation_progress.json").write_text(
        json.dumps(
            {
                "chapters": {
                    "1": {
                        "actual_num": 1,
                        "status": "completed",
                        "output_file": output.name,
                        "original_basename": source.name,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("OUTPUT_SDLXLIFF", "0")
    mixin = RetranslationMixin.__new__(RetranslationMixin)
    dialog = SDLXLIFFReviewDialog.__new__(SDLXLIFFReviewDialog)
    dialog.output_dir = str(tmp_path)
    dialog._book_entries = []
    dialog._book_index = 0
    dialog._last_autogen_signature = None
    dialog._sdlxliff_autogen_owner = mixin

    assert dialog._maybe_regenerate_review_sidecars(force=True) is True
    last_review_signature = dialog._current_review_signature()
    last_mt_signature = dialog._current_machine_translation_signature()
    last_autogen_signature = dialog._current_review_autogen_signature()

    class FailingOwner:
        def _generate_sdlxliff_sidecars_from_completed_entries(self, *args, **kwargs):
            raise AssertionError("manual validation should not regenerate current sidecars")

    dialog._sdlxliff_autogen_owner = FailingOwner()
    result = dialog._build_review_refresh_scan_result(
        force=False,
        validate=True,
        current_path=str(tmp_path / "SDLXLIFF" / "response_chapter0001.html.sdlxliff"),
        last_review_signature=last_review_signature,
        last_mt_signature=last_mt_signature,
        last_autogen_signature=last_autogen_signature,
    )

    assert result["error"] == ""
    assert result["stats"] is None
    assert result["sidecars_generated"] is False
    assert result["sidecar_changed"] is False


def test_sdlxliff_generation_streaming_does_not_replace_existing_sidecar_list(monkeypatch):
    dialog = SDLXLIFFReviewDialog.__new__(SDLXLIFFReviewDialog)
    dialog.output_dir = "existing-output"
    dialog.pieces = []
    dialog.piece_list = SimpleNamespace(count=lambda: 0)
    monkeypatch.setattr(
        dialog,
        "_sdlxliff_sidecar_paths_for_output_dir",
        lambda _output_dir: ["existing-output/SDLXLIFF/response_chapter0001.html.sdlxliff"],
    )

    assert dialog._prepare_generation_streaming_piece_list(total=1) is False


def test_sdlxliff_generation_finish_only_preserves_active_stream(monkeypatch):
    dialog = SDLXLIFFReviewDialog.__new__(SDLXLIFFReviewDialog)
    dialog.pieces = [{"path": "existing-output/SDLXLIFF/response_chapter0001.html.sdlxliff"}]
    dialog._generation_streaming_active = False
    dialog._generation_stream_preserve_after_finish = False
    dialog._review_data_loaded = False
    dialog.piece_list = SimpleNamespace(currentRow=lambda: 0)
    dialog._refresh_piece_header = lambda _row: None
    dialog._hide_generation_progress = lambda: None
    dialog.loading_label = SimpleNamespace(setText=lambda _text: None)
    dialog.save_status_label = SimpleNamespace(setText=lambda _text: None)

    dialog._finish_generation_streaming("Generated 1/1 SDLXLIFF sidecar(s)")

    assert dialog._generation_stream_preserve_after_finish is False


def test_sdlxliff_review_refresh_does_not_generate_when_no_output_html_exists(tmp_path):
    (tmp_path / "SDLXLIFF").mkdir()
    (tmp_path / "translation_progress.json").write_text(
        json.dumps(
            {
                "chapters": {
                    "1": {
                        "actual_num": 1,
                        "status": "completed",
                        "output_file": "response_chapter0001.html",
                        "original_basename": "chapter0001.xhtml",
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    class FailingOwner:
        def _generate_sdlxliff_sidecars_from_completed_entries(self, *args, **kwargs):
            raise AssertionError("generator should not run when no output HTML exists")

    dialog = SDLXLIFFReviewDialog.__new__(SDLXLIFFReviewDialog)
    dialog.output_dir = str(tmp_path)
    dialog._book_entries = []
    dialog._book_index = 0
    dialog._sdlxliff_autogen_owner = FailingOwner()

    autogen_signature = dialog._current_review_autogen_signature()
    assert dialog._review_autogen_has_output_html(autogen_signature) is False
    assert dialog._regenerate_review_sidecars_for_refresh_scan(
        force=True,
        previous_signature=None,
        current_signature=autogen_signature,
    ) is None

    result = dialog._build_review_refresh_scan_result(
        force=True,
        current_path="",
        last_review_signature=None,
        last_mt_signature=None,
        last_autogen_signature=None,
    )

    assert result["error"] == ""
    assert result["sidecars_generated"] is False
    assert result["stats"] is None


def test_sdlxliff_review_opener_shows_message_when_no_sidecars_or_output_html(tmp_path):
    (tmp_path / "SDLXLIFF").mkdir()
    progress = {
        "chapters": {
            "1": {
                "actual_num": 1,
                "status": "completed",
                "output_file": "response_chapter0001.html",
            }
        }
    }

    class StubRetranslation(RetranslationMixin):
        def __init__(self):
            self.messages = []

        def _show_message(self, msg_type, title, message, parent=None):
            self.messages.append((msg_type, title, message))

    stub = StubRetranslation()

    assert stub._open_or_reuse_sdlxliff_review(
        str(tmp_path),
        autogen_progress_data=progress,
    ) is None
    assert stub.messages == [
        (
            "info",
            "Text Analysis Unavailable",
            "No SDLXLIFF sidecars were found for this output folder.",
        )
    ]


def test_qa_sdlxliff_tag_check_flags_added_output_text_units():
    issue = _missing_beautifulsoup_tags_issue({"p": 212}, {"p": 213})

    assert issue == "missing_tags: 212/213 (+1)"


def test_qa_sdlxliff_tag_check_defaults_to_configured_tolerances():
    settings = default_qa_scan_settings()

    assert settings["sdlxliff_tag_retention_threshold"] == 0.9
    assert settings["sdlxliff_tag_surplus_tolerance"] == 0.05
    assert settings["sdlxliff_min_source_paragraph_tags"] == 20


def test_qa_sdlxliff_tag_check_flags_missing_output_text_units():
    issue = _missing_beautifulsoup_tags_issue({"p": 174}, {"p": 173})

    assert issue == "missing_tags: 174/173 (-1)"


def test_qa_sdlxliff_tag_check_allows_missing_tags_within_retention_threshold():
    issue = _missing_beautifulsoup_tags_issue(
        {"p": 100},
        {"p": 90},
        retention_threshold=0.9,
    )

    assert issue is None


def test_qa_sdlxliff_tag_check_flags_below_retention_threshold():
    issue = _missing_beautifulsoup_tags_issue(
        {"p": 100},
        {"p": 89},
        retention_threshold=0.9,
    )

    assert issue == "missing_tags: 100/89 (-11)"


def test_qa_sdlxliff_tag_check_still_flags_added_tags_with_relaxed_threshold():
    issue = _missing_beautifulsoup_tags_issue(
        {"p": 100},
        {"p": 101},
        retention_threshold=0.9,
    )

    assert issue == "missing_tags: 100/101 (+1)"


def test_qa_sdlxliff_tag_check_allows_surplus_within_tolerance():
    issue = _missing_beautifulsoup_tags_issue(
        {"p": 100},
        {"p": 105},
        surplus_tolerance=0.05,
    )

    assert issue is None


def test_qa_sdlxliff_tag_check_flags_surplus_above_tolerance():
    issue = _missing_beautifulsoup_tags_issue(
        {"p": 100},
        {"p": 106},
        surplus_tolerance=0.05,
    )

    assert issue == "missing_tags: 100/106 (+6)"


def test_qa_sdlxliff_tag_check_ignores_files_below_minimum_source_paragraphs():
    issue = _missing_beautifulsoup_tags_issue(
        {"p": 19, "h1": 100},
        {"p": 20, "h1": 100},
        min_source_paragraph_tags=20,
    )

    assert issue is None


def test_qa_sdlxliff_tag_check_checks_files_at_minimum_source_paragraphs():
    issue = _missing_beautifulsoup_tags_issue(
        {"p": 20, "h1": 100},
        {"p": 21, "h1": 100},
        min_source_paragraph_tags=20,
    )

    assert issue == "missing_tags: 120/121 (+1)"


def _quick_scan_sdlxliff_tag_issues(
    tmp_path,
    source_count,
    output_count,
    retention_threshold=0.9,
    surplus_tolerance=0.05,
    min_source_paragraph_tags=20,
):
    filename = "response_chapter0001.html"
    source_markup = "".join(f"<p>Source {index}</p>" for index in range(source_count))
    output_markup = "".join(f"<p>Output {index}</p>" for index in range(output_count))
    (tmp_path / filename).write_text(
        f"<html><body>{output_markup}</body></html>",
        encoding="utf-8",
    )
    sidecar_dir = tmp_path / "SDLXLIFF"
    sidecar_dir.mkdir()
    (sidecar_dir / f"{filename}.sdlxliff").write_text(
        f"""<?xml version="1.0" encoding="utf-8"?>
<xliff xmlns="urn:oasis:names:tc:xliff:document:1.2" version="1.2">
  <file original="chapter0001.xhtml" source-language="ko-KR" target-language="en-US">
    <body>
      <trans-unit id="html">
        <source><![CDATA[<html><body>{source_markup}</body></html>]]></source>
        <target><![CDATA[<html><body>{output_markup}</body></html>]]></target>
      </trans-unit>
    </body>
  </file>
</xliff>
""",
        encoding="utf-8",
    )
    settings = default_qa_scan_settings()
    settings.update(
        {
            "check_missing_beautifulsoup_tags": True,
            "sdlxliff_tag_retention_threshold": retention_threshold,
            "sdlxliff_tag_surplus_tolerance": surplus_tolerance,
            "sdlxliff_min_source_paragraph_tags": min_source_paragraph_tags,
        }
    )

    results = process_html_file_batch(
        (
            [(0, filename)],
            str(tmp_path),
            settings,
            "quick-scan",
            {},
            {},
            False,
            {},
            {},
            {},
        )
    )

    return results[0]["issues"]


def test_quick_scan_sdlxliff_tag_check_allows_ten_percent_missing(tmp_path):
    issues = _quick_scan_sdlxliff_tag_issues(tmp_path, 100, 90)

    assert not any(issue.startswith("missing_tags:") for issue in issues)


def test_quick_scan_sdlxliff_tag_check_flags_more_than_ten_percent_missing(tmp_path):
    issues = _quick_scan_sdlxliff_tag_issues(tmp_path, 100, 89)

    assert "missing_tags: 100/89 (-11)" in issues


def test_quick_scan_sdlxliff_tag_check_allows_five_percent_surplus(tmp_path):
    issues = _quick_scan_sdlxliff_tag_issues(tmp_path, 100, 105)

    assert not any(issue.startswith("missing_tags:") for issue in issues)


def test_quick_scan_sdlxliff_tag_check_flags_more_than_five_percent_surplus(tmp_path):
    issues = _quick_scan_sdlxliff_tag_issues(tmp_path, 100, 106)

    assert "missing_tags: 100/106 (+6)" in issues


def test_quick_scan_sdlxliff_tag_check_ignores_small_source_paragraph_files(tmp_path):
    issues = _quick_scan_sdlxliff_tag_issues(tmp_path, 19, 20)

    assert not any(issue.startswith("missing_tags:") for issue in issues)


def test_quick_scan_sdlxliff_tag_check_checks_minimum_source_paragraph_boundary(tmp_path):
    issues = _quick_scan_sdlxliff_tag_issues(tmp_path, 20, 22)

    assert "missing_tags: 20/22 (+2)" in issues


def test_qa_sdlxliff_tag_check_ignores_empty_text_units(tmp_path):
    assert _count_beautifulsoup_review_tags("<p></p><p>Source</p><h1>Title</h1><h2> </h2>") == {
        "h1": 1,
        "p": 1,
    }

    sidecar_dir = tmp_path / "SDLXLIFF"
    sidecar_dir.mkdir()
    (sidecar_dir / "response_chapter0001.html.sdlxliff").write_text(
        """<?xml version="1.0" encoding="utf-8"?>
<xliff xmlns="urn:oasis:names:tc:xliff:document:1.2" version="1.2">
  <file original="chapter0001.xhtml" source-language="ko-KR" target-language="en-US">
    <body>
      <trans-unit id="html">
        <source><![CDATA[<html><body><h1>Title</h1><p></p><p>Body</p></body></html>]]></source>
        <target><![CDATA[<html><body><h1>Title</h1><p>Body</p><p>Extra</p></body></html>]]></target>
      </trans-unit>
    </body>
  </file>
</xliff>
""",
        encoding="utf-8",
    )

    source_counts, output_counts = _sdlxliff_review_tag_counts(str(tmp_path), "response_chapter0001.html")

    assert source_counts == {"h1": 1, "p": 1}
    assert output_counts == {"h1": 1, "p": 2}
    assert _missing_beautifulsoup_tags_issue(source_counts, output_counts) == "missing_tags: 2/3 (+1)"


def _norm_windows_rename_test_path(path):
    return os.path.normpath(os.path.abspath(path))


def _windows_rename_test_gui(tmp_path):
    import translator_gui
    from translator_gui import TranslatorGUI

    gui = TranslatorGUI.__new__(TranslatorGUI)
    gui.config = {}
    gui.base_dir = str(tmp_path / "app")
    gui.manual_glossary_map = {}
    gui.manual_glossary_path = None
    gui.auto_loaded_glossary_path = None
    gui.auto_loaded_glossary_for_file = None
    gui.manual_glossary_manually_loaded = False
    gui._glossary_dir_candidate_cache = {"stale": object()}
    gui.logs = []
    gui.append_log = gui.logs.append
    gui.save_config = lambda show_message=True: None
    gui._update_manual_glossary_status = lambda: None
    Path(gui.base_dir, "Glossary").mkdir(parents=True)
    return translator_gui, gui


def test_windows_epub_rename_moves_auto_glossary_and_updates_state(tmp_path, monkeypatch):
    translator_gui, gui = _windows_rename_test_gui(tmp_path)
    monkeypatch.setattr(translator_gui.sys, "platform", "win32")

    old_epub = tmp_path / "Book .epub"
    old_epub.write_text("epub", encoding="utf-8")
    glossary_dir = Path(gui.base_dir) / "Glossary" / "Book"
    glossary_dir.mkdir(parents=True)

    old_glossary = glossary_dir / "Book _glossary.csv"
    old_glossary.write_text("term,translation\n", encoding="utf-8")
    for name in (
        "Book _glossary_progress.json",
        "Book _gender_tracker.json",
        "Book _glossary_history.json",
    ):
        (glossary_dir / name).write_text("{}", encoding="utf-8")

    old_epub_abs = _norm_windows_rename_test_path(old_epub)
    old_glossary_abs = _norm_windows_rename_test_path(old_glossary)
    gui.manual_glossary_map = {old_epub_abs: old_glossary_abs}
    gui.config["manual_glossary_map"] = dict(gui.manual_glossary_map)
    gui.manual_glossary_path = old_glossary_abs
    gui.config["manual_glossary_path"] = old_glossary_abs
    gui.auto_loaded_glossary_path = old_glossary_abs
    gui.auto_loaded_glossary_for_file = old_epub_abs
    monkeypatch.setenv("MANUAL_GLOSSARY", old_glossary_abs)

    new_path = gui._windows_supported_input_path(str(old_epub))

    new_epub = tmp_path / "Book.epub"
    new_glossary = glossary_dir / "Book_glossary.csv"
    new_epub_abs = _norm_windows_rename_test_path(new_epub)
    new_glossary_abs = _norm_windows_rename_test_path(new_glossary)

    assert _norm_windows_rename_test_path(new_path) == new_epub_abs
    assert new_epub.is_file()
    assert not old_epub.exists()
    assert new_glossary.read_text(encoding="utf-8") == "term,translation\n"
    assert not old_glossary.exists()
    assert (glossary_dir / "Book_glossary_progress.json").is_file()
    assert (glossary_dir / "Book_gender_tracker.json").is_file()
    assert (glossary_dir / "Book_glossary_history.json").is_file()
    assert gui.manual_glossary_map == {new_epub_abs: new_glossary_abs}
    assert gui.config["manual_glossary_map"] == {new_epub_abs: new_glossary_abs}
    assert gui.manual_glossary_path == new_glossary_abs
    assert gui.config["manual_glossary_path"] == new_glossary_abs
    assert gui.auto_loaded_glossary_path == new_glossary_abs
    assert gui.auto_loaded_glossary_for_file == new_epub_abs
    assert os.environ["MANUAL_GLOSSARY"] == new_glossary_abs
    assert gui._glossary_dir_candidate_cache == {}


def test_windows_epub_rename_does_not_overwrite_existing_glossary(tmp_path, monkeypatch):
    translator_gui, gui = _windows_rename_test_gui(tmp_path)
    monkeypatch.setattr(translator_gui.sys, "platform", "win32")

    old_epub = tmp_path / "Novel .epub"
    old_epub.write_text("epub", encoding="utf-8")
    glossary_dir = Path(gui.base_dir) / "Glossary" / "Novel"
    glossary_dir.mkdir(parents=True)
    old_glossary = glossary_dir / "Novel _glossary.json"
    new_glossary = glossary_dir / "Novel_glossary.json"
    old_glossary.write_text('{"old": true}', encoding="utf-8")
    new_glossary.write_text('{"existing": true}', encoding="utf-8")

    gui.manual_glossary_map = {
        _norm_windows_rename_test_path(old_epub): _norm_windows_rename_test_path(old_glossary)
    }
    gui.config["manual_glossary_map"] = dict(gui.manual_glossary_map)

    new_path = gui._windows_supported_input_path(str(old_epub))

    assert _norm_windows_rename_test_path(new_path) == _norm_windows_rename_test_path(tmp_path / "Novel.epub")
    assert old_glossary.read_text(encoding="utf-8") == '{"old": true}'
    assert new_glossary.read_text(encoding="utf-8") == '{"existing": true}'
    assert gui.manual_glossary_map == {
        _norm_windows_rename_test_path(tmp_path / "Novel.epub"): _norm_windows_rename_test_path(old_glossary)
    }
    assert gui.config["manual_glossary_map"] == gui.manual_glossary_map


def test_windows_non_epub_rename_does_not_remap_glossary_state(tmp_path, monkeypatch):
    translator_gui, gui = _windows_rename_test_gui(tmp_path)
    monkeypatch.setattr(translator_gui.sys, "platform", "win32")

    old_text = tmp_path / "Notes .txt"
    old_text.write_text("text", encoding="utf-8")
    glossary = Path(gui.base_dir) / "Glossary" / "Notes" / "Notes _glossary.csv"
    glossary.parent.mkdir(parents=True)
    glossary.write_text("term,translation\n", encoding="utf-8")

    original_map = {
        _norm_windows_rename_test_path(old_text): _norm_windows_rename_test_path(glossary)
    }
    gui.manual_glossary_map = dict(original_map)
    gui.config["manual_glossary_map"] = dict(original_map)

    new_path = gui._windows_supported_input_path(str(old_text))

    assert _norm_windows_rename_test_path(new_path) == _norm_windows_rename_test_path(tmp_path / "Notes.txt")
    assert (tmp_path / "Notes.txt").is_file()
    assert gui.manual_glossary_map == original_map
    assert gui.config["manual_glossary_map"] == original_map
