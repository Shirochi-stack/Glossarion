import json
import os
import re
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from lxml import etree

from sdlxliff_converter import convert_sdlxliff
from sdlxliff_extractor import extract_sdlxliff_to_chapters
from TransateKRtoEN import _write_html_sdlxliff_sidecar
from Retranslation_GUI import RetranslationMixin, SDLXLIFFReviewDialog
from scan_html_folder import (
    _count_beautifulsoup_review_tags,
    _missing_beautifulsoup_tags_issue,
    _sdlxliff_review_tag_counts,
)


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


def test_sdlxliff_review_heading_to_paragraph_mismatch_stays_red(tmp_path):
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

    assert piece["mismatch"] is True
    assert piece["red_count"] == 1
    assert piece["yellow_count"] == 0
    assert piece["rows"][0]["status"] == "red"
    assert piece["rows"][0]["reason"] == "tag mismatch"


def test_sdlxliff_review_translate_tooltips_uses_google_translate_free():
    source = (SRC / "Retranslation_GUI.py").read_text(encoding="utf-8")

    assert "🌐 Generate Google Translate Preview" in source
    assert "google-translate-free" in source
    assert "from google_free_translate import GoogleFreeTranslateNew" in source
    assert 'name="sdlxliff-tooltip-google-translate-free"' in source
    assert "batch_html = self._tooltip_batch_html(work)" in source
    assert "result = translator.translate(batch_html)" in source
    assert 'data-sdl-tip="' in source
    assert "_start_tooltip_translation" in source
    assert "_translate_single_row_tooltip" in source
    assert "_refresh_visible_review_row_source_preview" in source
    assert "_refresh_visible_review_row_source_previews" in source
    assert "visible_only=True" in source
    assert "Google Translate \\u2192" in source
    assert "_open_google_translate" not in source
    assert "Translate tooltip" not in source
    assert "Retranslate tooltip" not in source
    assert "Inject Machine Translation" in source
    assert "inject_machine_translation_callback" in source
    assert "tooltip_translation_pending" in source
    assert "⏳ Translating with Google Translate..." in source
    assert "SdlReviewSourceText" in source
    assert "SdlReviewMachineTranslation" in source
    assert "SdlReviewMachineTranslationPending" in source
    assert "QListWidget#SdlReviewPieceList::item:hover:!selected" in source
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
    assert "_refresh_visible_review_row_source_preview" in apply_body
    assert "_discard_piece_page" not in apply_body
    assert "_render_piece" not in apply_body


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
    assert piece["yellow_count"] == 0
    assert piece["mismatch"] is True
    assert piece["count_ratio"] == 0.5


def test_retranslation_show_model_info_defaults_on_but_respects_saved_false():
    mixin = RetranslationMixin.__new__(RetranslationMixin)
    mixin.config = {}
    mixin._retranslation_dialog_cache = {}

    assert mixin._get_retranslation_show_model_info_state() is True

    mixin.config = {mixin._RETRANSLATION_SHOW_MODEL_INFO_CONFIG_KEY: False}

    assert mixin._get_retranslation_show_model_info_state() is False


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

    stats = mixin._generate_sdlxliff_sidecars_from_completed_entries(
        str(tmp_path),
        progress_data=progress,
    )

    sidecar = tmp_path / "SDLXLIFF" / "response_chapter0001.html.sdlxliff"
    assert stats["created"] == 1
    assert sidecar.is_file()
    assert os.environ["OUTPUT_SDLXLIFF"] == "0"

    dialog = SDLXLIFFReviewDialog.__new__(SDLXLIFFReviewDialog)
    source_html, target_html = dialog._read_sdlxliff_html_pair(str(sidecar))

    assert "Source Title" in source_html
    assert "Source body." in source_html
    assert "Target Title" in target_html
    assert "Target body." in target_html


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
    assert [row["source"] for row in piece["rows"]] == [
        "The Girl With the Magitek Armor",
        "Final Fantasy 6- The Novel",
        "Written by me: Celes Chere",
    ]
    assert piece["red_count"] == 0


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
    dialog = SDLXLIFFReviewDialog.__new__(SDLXLIFFReviewDialog)
    dialog.output_dir = str(tmp_path)
    dialog.current_path = ""

    pieces = dialog._load_pieces()

    assert [piece["name"] for piece in pieces] == ["response_chapter0001.html.sdlxliff"]
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


def test_qa_sdlxliff_tag_check_flags_added_output_text_units():
    issue = _missing_beautifulsoup_tags_issue({"p": 212}, {"p": 213})

    assert issue == "missing_tags: 212/213 (+1)"


def test_qa_sdlxliff_tag_check_flags_missing_output_text_units():
    issue = _missing_beautifulsoup_tags_issue({"p": 174}, {"p": 173})

    assert issue == "missing_tags: 174/173 (-1)"


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
