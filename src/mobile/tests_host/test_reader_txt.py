"""Host tests for TXT books in the Reader (device report issue 2; UI_SPEC §3.11 "Modes").

* ``ui/reader/text_book.py``: text -> reader HTML (paragraphs, line breaks, escaping, CRLF/BOM),
  the decode order (BOM, UTF-8, CJK code pages, never BOM-less UTF-16), standalone sections
  (~20k characters cut at paragraph ends; compiled ``_translated.txt`` split on the separator the
  real ``txt_processor.create_output_structure`` writes), TXT translation workspaces (the real
  split cache written by ``TextFileProcessor._save_split_cache``, sections paired with their
  translation by ``content_hash`` -- chunk 11 of chapter 1 is numbered 2.0 --, with
  ``RETAIN_SOURCE_EXTENSION`` names, and by the pipeline's output name for old progress files);
* ``ui/reader/session.py``: ``plan_open`` for a Library TXT, an imported Not-started TXT
  (``library_core.import_paths`` into a scratch Library), a TXT workspace and a compiled
  ``_translated.txt``; ``plan_for_file`` for ``.txt``; text sessions (load, search, the Translate
  reason), TXT workspace sessions (Original / Translated / Bilingual, no PDF raw extraction) and
  the saved-position fallback between the two TXT section layouts;
* ``ui/reader/document.py`` builds a TXT page (the shared ``ReaderDocument`` with a ``.txt`` path).

Every test runs with HOME / USERPROFILE / APPDATA / GLOSSARION_LIBRARY_DIR / OUTPUT_DIRECTORY in
``tmp_path`` and the Library pinned there (``install_library_env``).

Run from src/mobile with the mobile venv:
    .venv/Scripts/python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic tests_host/test_reader_txt.py
"""

from __future__ import annotations

import codecs
import html
import importlib.util
import json
import os
import re
import sys
import types
from pathlib import Path

import pytest

MOBILE_DIR = Path(__file__).resolve().parents[1]
APP_DIR = MOBILE_DIR / "app"
SRC_DIR = MOBILE_DIR.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from glossarion_mobile.ui.reader import model as rm  # noqa: E402
from glossarion_mobile.ui.reader import text_book as tb  # noqa: E402


def _has(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


needs_cores = pytest.mark.skipif(
    not all(_has(m) for m in ("reader_doc", "library_core", "workspace_reader", "txt_processor", "bs4")),
    reason="the shared reader cores are not importable here")

THEME = {"name": "Dark", "bg": "#1e1e1e", "fg": "#d4d4d4", "heading": "#c8c8f0", "link": "#6c9bd2",
         "code_bg": "#252530", "border": "#333333"}


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    """Nothing reads or writes the developer's Library, output folders, home or config."""
    for key in ("HOME", "USERPROFILE", "APPDATA"):
        (tmp_path / "env" / key.lower()).mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv(key, str(tmp_path / "env" / key.lower()))
    (tmp_path / "Library").mkdir(exist_ok=True)
    (tmp_path / "Output").mkdir(exist_ok=True)
    monkeypatch.setenv("GLOSSARION_LIBRARY_DIR", str(tmp_path / "Library"))
    monkeypatch.setenv("OUTPUT_DIRECTORY", str(tmp_path / "Output"))
    monkeypatch.setenv("GLOSSARION_HTTP_LOG", "0")
    monkeypatch.delenv("RETAIN_SOURCE_EXTENSION", raising=False)
    monkeypatch.delenv("retain_source_extension", raising=False)
    library_core = None
    if _has("library_core"):
        import library_core
        import reader_doc

        monkeypatch.setattr(reader_doc, "_EPUB_CACHE_DIR_OVERRIDE", str(tmp_path / "epubcache"))
        # The default output root follows the pinned env (never the app folder, i.e. src/).
        library_core.install_library_env(library_core.LibraryEnv(
            tmp_path / "Library", (tmp_path / "Output",), tmp_path / "cache", {}))
    try:
        yield tmp_path
    finally:
        if library_core is not None:
            library_core.uninstall_library_env()


def _paragraphs(markup: str) -> list:
    """The text of each ``<p>`` (``<br/>`` back to newlines)."""
    return [html.unescape(p.replace("<br/>", "\n")) for p in re.findall(r"<p>(.*?)</p>", markup, re.S)]


def _novel_paragraphs(count: int) -> list:
    words = ("바람이", "불었다", "그는", "검을", "들었다", "하늘은", "맑았다", "the", "road", "went", "on")
    out = []
    for i in range(count):
        size = 40 + (i * 37) % 160
        text = " ".join(words[(i + k) % len(words)] for k in range(size // 4))
        out.append(f"{i + 1}. {text}")
    return out


# =====================================================================================
# text -> reader HTML
# =====================================================================================


def test_blank_line_blocks_become_paragraphs_with_line_breaks():
    markup = tb.text_to_reader_html("First line\nsame paragraph\n\n\nSecond paragraph\n")
    assert markup == "<p>First line<br/>same paragraph</p>\n<p>Second paragraph</p>"
    # a line holding only (full-width) spaces separates paragraphs too
    assert _paragraphs(tb.text_to_reader_html("a\n　\nb")) == ["a", "b"]


def test_text_without_blank_lines_gets_one_paragraph_per_line():
    markup = tb.text_to_reader_html("첫째 줄입니다.\n둘째 줄입니다.\n셋째.\n")
    assert markup.count("<p>") == 3 and "<br/>" not in markup
    assert _paragraphs(markup) == ["첫째 줄입니다.", "둘째 줄입니다.", "셋째."]


def test_markup_in_text_is_escaped_and_stays_inert():
    from glossarion_mobile.ui.reader.document import sanitize_book_html

    markup = tb.text_to_reader_html('<script>alert(1)</script>\n\n<img src=x onerror="alert(2)">\n\n'
                                    'see javascript:alert(3) & more')
    assert "<script" not in markup and "<img" not in markup and "&amp; more" in markup
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in markup
    assert sanitize_book_html(markup) == markup  # nothing active left for the sanitiser to remove
    assert tb.text_to_reader_html("x", title="<b>T</b>").startswith("<h2>&lt;b&gt;T&lt;/b&gt;</h2>")


def test_crlf_and_bom_give_the_same_html():
    plain = tb.text_to_reader_html("one\ntwo\n\nthree")
    assert tb.text_to_reader_html("﻿one\r\ntwo\r\n\r\nthree") == plain
    assert tb.text_to_reader_html("one\rtwo\r\rthree") == plain
    data = codecs.BOM_UTF8 + "one\r\ntwo\r\n\r\nthree".encode("utf-8")
    assert tb.text_to_reader_html(tb.decode_text_bytes(data)) == plain


# =====================================================================================
# decoding
# =====================================================================================


def test_cp949_decodes_as_korean_never_through_utf16():
    text = "안녕하세요. 이것은 한국어 소설입니다."
    assert tb.decode_text_bytes(text.encode("cp949")) == text
    # Hangul whose CP949 bytes also read as BOM-less UTF-16 (no surrogate code units): that decode
    # "succeeds" with mojibake, so UTF-16 must never be tried without a BOM.
    syllables = [ch for ch in "가각간갈감강개거건걸검게겨결경계고곡공과관광교구국군궁권귀규그근글금기길김"
                 if not 0xD8 <= ch.encode("cp949")[1] <= 0xDF]
    text = "".join(syllables[:16])
    data = text.encode("cp949")
    assert data.decode("utf-16-le") != text
    with pytest.raises(UnicodeDecodeError):
        data.decode("utf-8")
    assert tb.decode_text_bytes(data) == text


def test_bom_files_decode_with_their_encoding():
    text = "第一章 天地 — 한국어"
    assert tb.decode_text_bytes(text.encode("utf-16")) == text  # BOM + native order
    assert tb.decode_text_bytes(codecs.BOM_UTF16_BE + text.encode("utf-16-be")) == text
    assert tb.decode_text_bytes(codecs.BOM_UTF16_LE + text.encode("utf-16-le")) == text
    assert tb.decode_text_bytes(codecs.BOM_UTF32_LE + text.encode("utf-32-le")) == text
    assert tb.decode_text_bytes(codecs.BOM_UTF8 + text.encode("utf-8")) == text
    assert tb.decode_text_bytes(text.encode("utf-8")) == text


def test_invalid_bytes_fall_back_to_replacement_characters():
    decoded = tb.decode_text_bytes(b"abc\x80\xffdef")
    assert decoded.startswith("abc") and decoded.endswith("def") and "�" in decoded
    assert tb.decode_text_bytes(b"") == ""


@pytest.mark.parametrize("text,codec", [
    ("「おはよう」と彼女は言った。\n「今日はいい天気だね」\n", "shift_jis"),
    ("「ねえ、あなたの名前は？」\n「ぼくはタロウ。きみは？」\n", "shift_jis"),
    ("　僕は朝起きて、学校へ行った。\n　その日は雨だった。\n", "shift_jis"),
    ("①吾輩は猫である。名前はまだ無い。\n", "cp932"),
    ("第一章　転生\n\n目を覚ますと、そこは見知らぬ森の中だった。\n", "euc_jp"),
    ("他说：“你好，世界。”\n她笑了笑，没有回答。\n", "gbk"),
    ("他說：「你好，世界。」\n她笑了笑，沒有回答。\n", "big5"),
    ("第一章 重生\n\n林凡睜開眼睛，發現自己回到了十年前。窗外的陽光照在床上。\n", "big5"),
    ("제1장 회귀\n\n눈을 떠보니 십 년 전이었다. 똠방각하 쀍 햏\n", "cp949"),
], ids=["sjis-short", "sjis-dialogue", "sjis-indented", "cp932-nec", "euc-jp", "gbk-short",
        "big5-short", "big5-para", "cp949-uhc"])
def test_cjk_code_pages_are_detected_not_first_that_decodes(text, codec):
    """GB18030 reads almost any double-byte text and CP949 many Shift-JIS pairs, so the code page
    is detected (chardet / charset_normalizer) instead of taking the first that decodes."""
    assert tb.decode_text_bytes(text.encode(codec)) == text


def test_a_large_cjk_file_is_detected_from_a_sample():
    text = "第一章　転生\n\n目を覚ますと、そこは見知らぬ森の中だった。\n\n" * 20000
    data = text.encode("cp932")
    assert len(data) > 10 * tb._DETECT_SAMPLE
    assert tb.decode_text_bytes(data) == text
    # a stray byte past the sample keeps the detected code page (one replacement character)
    broken = tb.decode_text_bytes(data[: -4] + b"\x81\x20" + data[-4:])  # lead byte, bad trail
    assert broken.startswith("第一章　転生") and broken.count("�") >= 1


# =====================================================================================
# standalone sections
# =====================================================================================


def test_load_text_chapters_has_the_epub_loader_contract(tmp_path):
    path = tmp_path / "Library" / "short.txt"
    path.write_text("Hello there.\n\nSecond paragraph.", encoding="utf-8")
    result = tb.load_text_chapters(str(path))
    assert isinstance(result, tuple) and len(result) == 3
    chapters, images, filenames = result
    assert images == {} and filenames == ["section_0001.txt"]
    assert chapters == [("Section 1", "<p>Hello there.</p>\n<p>Second paragraph.</p>")]
    assert tb.load_text_chapters(str(path), should_stop=lambda: True) is None
    with pytest.raises(OSError):
        tb.load_text_chapters(str(tmp_path / "missing.txt"))


@needs_cores
def test_compiled_output_of_the_real_txt_processor_splits_back_into_its_sections(tmp_path):
    from txt_processor import TextFileProcessor

    workspace = tmp_path / "Output" / "novel"
    workspace.mkdir(parents=True)
    processor = TextFileProcessor.__new__(TextFileProcessor)  # no tiktoken splitter needed
    processor.file_path = str(tmp_path / "Library" / "novel.txt")
    processor.output_dir = str(workspace)
    processor.file_base = "novel"
    translated = [("response_section_1_0.txt", "The first section.\n\nIt has two paragraphs."),
                  ("response_section_1_1.txt", "The second section."),
                  ("response_section_1_2.txt", "The third section.\nWith a line break.")]
    compiled = processor.create_output_structure(translated)
    assert compiled.endswith("novel_translated.txt")
    chapters, _images, filenames = tb.load_text_chapters(compiled)
    assert filenames == ["section_0001.txt", "section_0002.txt", "section_0003.txt"]
    assert [t for t, _h in chapters] == ["Section 1", "Section 2", "Section 3"]
    assert [_paragraphs(h) for _t, h in chapters] == [
        ["The first section.", "It has two paragraphs."], ["The second section."],
        ["The third section.", "With a line break."]]  # a section without blank lines: a <p> per line
    assert tb.text_section_count(compiled) == 3


def test_separator_splits_only_compiled_output(tmp_path):
    body = "Part A.\n\n" + "=" * 50 + "\n\nPart B."
    raw = tmp_path / "Library" / "raw_novel.txt"
    raw.write_text(body, encoding="utf-8")
    chapters, _i, _f = tb.load_text_chapters(str(raw))
    assert len(chapters) == 1 and "Part A." in chapters[0][1] and "Part B." in chapters[0][1]
    named = tmp_path / "Library" / "raw_novel_translated.txt"
    named.write_text(body, encoding="utf-8")
    assert len(tb.load_text_chapters(str(named))[0]) == 2
    beside = tmp_path / "Output" / "ws"
    beside.mkdir(parents=True)
    (beside / "translation_progress.json").write_text('{"chapters": {}}', encoding="utf-8")
    (beside / "anything.txt").write_text(body.replace("\n", "\r\n"), encoding="utf-8")
    assert len(tb.load_text_chapters(str(beside / "anything.txt"))[0]) == 2


@pytest.mark.parametrize("joiner", ["\n\n", "\n"])
def test_large_text_splits_into_paragraph_bounded_sections(tmp_path, joiner):
    paragraphs = _novel_paragraphs(900)
    text = joiner.join(paragraphs)
    assert len(text) >= 100_000
    path = tmp_path / "Library" / "big.txt"
    path.write_bytes(text.encode("utf-8"))
    chapters, _images, filenames = tb.load_text_chapters(str(path))
    assert len(chapters) >= 5
    sections = [_paragraphs(h) for _t, h in chapters]
    assert all(sum(len(p) for p in section) <= tb.READER_TXT_SECTION_CHARS for section in sections)
    assert [p for section in sections for p in section] == paragraphs  # every paragraph whole, in order
    again = tb.load_text_chapters(str(path))
    assert again[2] == filenames == [f"section_{n:04d}.txt" for n in range(1, len(chapters) + 1)]
    assert again[0] == chapters
    assert tb.text_section_count(str(path)) == len(chapters)


def test_a_file_without_line_breaks_is_still_cut_into_sections(tmp_path):
    words = " ".join(f"word{i}" for i in range(12_000))  # one ~100k-character paragraph
    path = tmp_path / "Library" / "wall.txt"
    path.write_text(words, encoding="utf-8")
    chapters, _images, _names = tb.load_text_chapters(str(path))
    texts = [p for _t, h in chapters for p in _paragraphs(h)]
    assert len(chapters) >= 5 and all(len(t) <= tb.READER_TXT_SECTION_CHARS for t in texts)
    assert " ".join(texts).split() == words.split()


def test_an_empty_file_is_one_empty_section(tmp_path):
    from glossarion_mobile.ui.reader.document import chapter_body_empty

    path = tmp_path / "Library" / "empty.txt"
    path.write_bytes(b"")
    chapters, images, filenames = tb.load_text_chapters(str(path))
    assert chapters == [("Section 1", "")] and images == {} and filenames == ["section_0001.txt"]
    assert chapter_body_empty(chapters[0][1])  # the Reader shows its empty-chapter state


# =====================================================================================
# TXT translation workspaces
# =====================================================================================


def _txt_workspace(tmp_path, *, chunks: int = 12, untranslated=(5,), retain: bool = False,
                   with_hash: bool = True) -> types.SimpleNamespace:
    """A TXT workspace the way txt_processor + TransateKRtoEN leave it: word_count/ chunks, the
    split cache (written by the real ``_save_split_cache``), response files and progress entries."""
    from txt_processor import TextFileProcessor

    raw = tmp_path / "Library" / "Raw" / "novel.txt"
    raw.parent.mkdir(parents=True, exist_ok=True)
    workspace = tmp_path / "Output" / "novel"
    (workspace / "word_count").mkdir(parents=True)
    (workspace / ".cache").mkdir()
    processor = TextFileProcessor.__new__(TextFileProcessor)
    processor.file_path = str(raw)
    processor.output_dir = str(workspace)
    processor.file_base = "novel"
    sections, progress, bodies = [], {}, []
    for k in range(1, chunks + 1):
        num = round(1 + (k - 1) * 0.1, 1)  # txt_processor's chunk numbering: chunk 11 -> 2.0
        filename = f"section_1_{k - 1}.txt"
        body = f"원문 {k}번째 조각입니다.\n\n둘째 문단 {k}."
        bodies.append(body)
        with open(workspace / "word_count" / filename, "w", encoding="utf-8") as stream:
            stream.write(body)
        sections.append({"num": num, "title": f"novel (Part {k}/{chunks})", "filename": filename,
                         "is_chunk": True, "source_file": str(raw),
                         "chunk_info": {"chunk_idx": k, "total_chunks": chunks, "original_chapter": 1}})
        major, minor = int(num), int(round((num - int(num)) * 10))
        output = f"section_{major}_{minor}.txt" if retain else f"response_section_{major}_{minor}.txt"
        if k - 1 in untranslated:
            continue
        (workspace / output).write_text(f"Translated chunk {k}.\n\nSecond paragraph {k}.", encoding="utf-8")
        entry = {"actual_num": num, "output_file": output, "status": "completed"}
        if with_hash:
            entry["content_hash"] = processor._generate_hash(body)
        progress[str(num)] = entry
    raw.write_text("\n\n".join(bodies), encoding="utf-8")
    processor._save_split_cache(str(workspace / ".cache" / "split.cache"), processor._generate_hash("x"), sections)
    (workspace / "translation_progress.json").write_text(
        json.dumps({"chapters": progress, "chapter_chunks": {}, "version": "2.1"}), encoding="utf-8")
    (workspace / "source_epub.txt").write_text(str(raw), encoding="utf-8")
    book = {"name": "novel", "type": "in_progress", "path": str(workspace), "output_folder": str(workspace),
            "raw_source_path": str(raw), "is_in_progress": True}
    return types.SimpleNamespace(raw=raw, workspace=workspace, book=book, sections=sections, bodies=bodies)


@needs_cores
def test_workspace_manifest_pairs_sections_by_content_hash(tmp_path):
    ws = _txt_workspace(tmp_path)
    manifest = tb.build_text_workspace_manifest(str(ws.workspace))
    assert manifest["source_format"] == "txt" and manifest["source_path"] == str(ws.raw)
    entries = manifest["entries"]
    assert [e["filename"] for e in entries] == [s["filename"] for s in ws.sections]  # split.cache order
    assert all(Path(e["raw_path"]).is_file() for e in entries)
    assert [e["title"] for e in entries[:2]] == ["Section 1", "Section 2"]
    # chunk 11 of chapter 1 is numbered 2.0: section_1_10.txt <-> response_section_2_0.txt
    assert Path(entries[10]["translated_path"]).name == "response_section_2_0.txt"
    assert Path(entries[11]["translated_path"]).name == "response_section_2_1.txt"
    assert entries[5]["translated_path"] == "" and entries[5]["status"] == ""
    assert entries[0]["status"] == "completed"
    assert tb.has_text_workspace(str(ws.workspace)) and tb.workspace_section_count(str(ws.workspace)) == 12


@needs_cores
def test_retained_source_extension_names_pair_too(tmp_path):
    ws = _txt_workspace(tmp_path, retain=True)
    entries = tb.build_text_workspace_manifest(str(ws.workspace))["entries"]
    assert Path(entries[10]["translated_path"]).name == "section_2_0.txt"
    assert Path(entries[10]["translated_path"]).parent == ws.workspace  # never the word_count raw
    assert entries[5]["translated_path"] == ""


@needs_cores
def test_progress_without_hashes_pairs_by_the_pipeline_output_name(tmp_path):
    if not _has("TransateKRtoEN"):
        pytest.skip("TransateKRtoEN is not importable here")
    from TransateKRtoEN import FileUtilities

    ws = _txt_workspace(tmp_path, with_hash=False)
    assert FileUtilities.create_chapter_filename(dict(ws.sections[10])) == "response_section_2_0.txt"
    entries = tb.build_text_workspace_manifest(str(ws.workspace))["entries"]
    assert Path(entries[10]["translated_path"]).name == "response_section_2_0.txt"
    assert Path(entries[0]["translated_path"]).name == "response_section_1_0.txt"
    assert entries[5]["translated_path"] == ""


@needs_cores
def test_no_split_yet_means_no_entries(tmp_path):
    workspace = tmp_path / "Output" / "fresh"
    workspace.mkdir(parents=True)
    (workspace / "translation_progress.json").write_text('{"chapters": {}}', encoding="utf-8")
    manifest = tb.build_text_workspace_manifest(str(workspace))
    assert manifest["entries"] == [] and manifest["source_format"] == "txt"
    assert not tb.has_text_workspace(str(workspace)) and tb.workspace_section_count(str(workspace)) == 0


@needs_cores
def test_workspace_chapters_render_both_sides(tmp_path):
    ws = _txt_workspace(tmp_path)
    manifest = tb.build_text_workspace_manifest(str(ws.workspace))
    raw, translated, filenames = tb.load_text_workspace_chapters(manifest)
    assert len(raw) == len(translated) == len(filenames) == 12
    assert _paragraphs(raw[10][1]) == ["원문 11번째 조각입니다.", "둘째 문단 11."]
    assert _paragraphs(translated[10][1]) == ["Translated chunk 11.", "Second paragraph 11."]
    assert tb.UNTRANSLATED_TEXT in translated[5][1] and "원문 6번째 조각입니다." in translated[5][1]
    assert tb.load_text_workspace_chapters(manifest, should_stop=lambda: True) is None


# =====================================================================================
# Reader session
# =====================================================================================


def _novel_txt(tmp_path, name: str = "Novel.txt", *, marker: str = "SAPPHIRE") -> Path:
    path = tmp_path / "Library" / "Raw" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    paragraphs = _novel_paragraphs(240)
    paragraphs[-1] += f" {marker}"  # in the last section, past the first ~20k characters
    path.write_text("\n\n".join(paragraphs), encoding="utf-8")
    return path


@needs_cores
def test_library_txt_opens_in_text_mode_and_searches(tmp_path):
    from glossarion_mobile.ui.reader import session as rs

    novel = _novel_txt(tmp_path)
    engine = rs.DocEngine()
    plan = rs.plan_open({"type": "txt", "name": "Novel", "path": str(novel)}, engine=engine)
    assert plan.mode == rs.MODE_PLAIN and plan.source_kind == rs.SOURCE_TXT and not plan.error
    assert plan.epub_path == str(novel) and plan.title == "Novel"
    session = rs.ReaderSession(plan, engine=engine)
    session.load()
    assert session.count >= 2 and "<p>" in session.chapter_html(0)
    assert session.filenames[:2] == ["section_0001.txt", "section_0002.txt"]
    assert session.display_numbers == list(range(1, session.count + 1)) and session.native_toc == []
    assert not session.has_alternate and session.available_modes(0)[rm.TRANSLATED]
    rows = session.search("SAPPHIRE")
    assert [r["chapter_idx"] for r in rows] == [session.count - 1]
    assert session.translate_target(0) == ("", "", rs.TXT_TRANSLATE_REASON) and not session.translate_visible(0)
    assert session.adopt_output_folder(str(tmp_path / "Output")) == set()
    assert session.image_bytes("x.png") is None  # no EPUB zip behind a text book
    session.close()


@needs_cores
def test_plan_for_file_accepts_txt_and_other_types_keep_the_reason(tmp_path):
    from glossarion_mobile.ui.reader import session as rs

    novel = _novel_txt(tmp_path, "shared.txt")
    plan = rs.plan_for_file(str(novel))
    assert plan.mode == rs.MODE_PLAIN and plan.source_kind == rs.SOURCE_TXT and plan.title == "shared"
    session = rs.ReaderSession(plan, engine=rs.DocEngine())
    session.load()
    assert session.count >= 2
    epub = rs.plan_for_file(str(tmp_path / "Book.epub"))
    assert epub.source_kind == rs.SOURCE_EPUB and epub.epub_path.endswith("Book.epub")
    pdf = tmp_path / "Doc.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    other = rs.plan_open({"name": "Doc", "path": str(pdf)}, engine=rs.DocEngine())
    assert other.error and "Reader opens EPUB" in other.error and "TXT" in other.error


@needs_cores
def test_imported_not_started_txt_opens_in_text_mode(tmp_path):
    import library_core

    from glossarion_mobile.ui.reader import session as rs

    source = _novel_txt(tmp_path, "incoming.txt")
    picked = tmp_path / "picked.txt"
    picked.write_bytes(source.read_bytes())
    result = library_core.import_paths([str(picked)], target="raw", copy_into_library=True)
    assert result["imported"] and not result["errors"]
    workspace = tmp_path / "Output" / "picked"
    assert (workspace / "source_epub.txt").is_file() and (workspace / "translation_progress.json").is_file()
    in_progress, _completed = library_core.scan_library({})
    rows = [r for r in in_progress if os.path.normcase(str(r.get("output_folder") or "")) ==
            os.path.normcase(str(workspace))]
    assert rows and rows[0]["translation_state"] == "not_started"  # the "🆕 Not started" card
    book = rows[0]
    engine = rs.DocEngine()
    plan = rs.plan_open(book, engine=engine)
    assert plan.mode == rs.MODE_PLAIN and plan.source_kind == rs.SOURCE_TXT and not plan.error
    assert Path(plan.epub_path).name == "picked.txt" and plan.output_folder == str(workspace)
    session = rs.ReaderSession(plan, engine=engine)
    session.load()
    assert session.count >= 2 and "<p>" in session.chapter_html(0)


@needs_cores
def test_txt_workspace_session_has_original_translated_and_bilingual(tmp_path):
    from glossarion_mobile.ui.reader import session as rs

    ws = _txt_workspace(tmp_path)
    engine = rs.DocEngine()
    plan = rs.plan_open(ws.book, engine=engine)
    assert plan.mode == rs.MODE_WORKSPACE and plan.source_kind == rs.SOURCE_TXT
    assert plan.workspace_dir == str(ws.workspace) and plan.source_path == str(ws.raw) and not plan.initial_raw
    assert rs.plan_open(ws.book, raw_only=True, engine=engine).initial_raw
    session = rs.ReaderSession(plan, engine=engine)
    session.load()
    assert session.count == 12 and session.has_alternate and session.overlay_applied
    assert session.filenames[10] == "section_1_10.txt" and session.display_numbers == list(range(1, 13))
    info = session.chapter_info(10)
    assert info.has_raw and info.has_translation and info.status == "completed"
    assert session.available_modes(10) == {"original": True, "translated": True, "bilingual": True}
    assert session.available_modes(5)["bilingual"] is False and not session.chapter_info(5).has_translation
    assert "Translated chunk 11." in session.chapter_html(10)
    assert "원문 11번째" in session.chapter_html(10, rm.ORIGINAL)
    bilingual = session.chapter_html(10, rm.BILINGUAL)
    assert bilingual.count('<div class="glr-bi-pair">') == 2 and "원문 11번째" in bilingual and "Translated chunk 11." in bilingual
    assert tb.UNTRANSLATED_TEXT in session.chapter_html(5) and "원문 6번째" in session.chapter_html(5)
    before = session.chapter_html(0, rm.ORIGINAL)
    assert session.ensure_workspace_raw(0) is False and session.chapter_html(0, rm.ORIGINAL) == before
    assert session.translate_target(0)[2] and not session.translate_visible(0)
    session.set_flavor(rm.ORIGINAL)
    assert session.search("둘째 문단 12")[0]["chapter_idx"] == 11


@needs_cores
def test_completed_txt_card_opens_the_workspace_or_the_compiled_file(tmp_path):
    from glossarion_mobile.ui.reader import session as rs

    ws = _txt_workspace(tmp_path, untranslated=())
    compiled = ws.workspace / "novel_translated.txt"
    compiled.write_text("One.\n\n" + "=" * 50 + "\n\nTwo.", encoding="utf-8")
    card = {"name": "novel", "type": "txt", "path": str(compiled), "output_folder": str(ws.workspace),
            "raw_source_path": str(ws.raw), "compiled_output_path": str(compiled), "is_in_progress": False}
    engine = rs.DocEngine()
    plan = rs.plan_open(card, engine=engine)
    assert plan.mode == rs.MODE_WORKSPACE and plan.source_kind == rs.SOURCE_TXT
    (ws.workspace / ".cache" / "split.cache").unlink()  # no split: the compiled file as text
    plan = rs.plan_open(card, engine=engine)
    assert plan.mode == rs.MODE_PLAIN and plan.source_kind == rs.SOURCE_TXT and plan.epub_path == str(compiled)
    session = rs.ReaderSession(plan, engine=engine)
    session.load()
    assert session.count == 2 and _paragraphs(session.chapter_html(1)) == ["Two."]


@needs_cores
def test_saved_position_of_the_other_txt_layout_restores_by_book_percent(tmp_path):
    from glossarion_mobile.ui.reader import session as rs

    ws = _txt_workspace(tmp_path, chunks=12)
    big = "\n\n".join(_novel_paragraphs(400))  # the raw file read standalone: several sections
    ws.raw.write_text(big, encoding="utf-8")
    standalone = tb.text_section_count(str(ws.raw))
    assert standalone >= 3
    engine = rs.DocEngine()
    session = rs.ReaderSession(rs.plan_open(ws.book, engine=engine), engine=engine)
    session.load()
    saved = {"href": "section_0002.txt", "chapter": 1, "fraction": 0.5, "mode": "translated", "page": 3}
    position = session.position_from_pref(saved)
    expected = rm.book_percent(1, 0.5, standalone)
    assert position is not None and position.href == session.filenames[position.chapter]
    assert abs(rm.book_percent(position.chapter, position.fraction, session.count) - expected) <= 1
    same = session.position_from_pref({"href": "section_1_3.txt", "chapter": 3, "fraction": 0.25})
    assert (same.chapter, same.fraction) == (3, 0.25)  # its own layout: unchanged
    assert session.position_from_pref(None) is None
    # text mode on the raw file with a position saved in the translation's layout
    text = rs.ReaderSession(rs._text_plan(str(ws.raw), output_folder=str(ws.workspace)), engine=engine)
    text.load()
    back = text.position_from_pref({"href": "section_1_9.txt", "chapter": 9, "fraction": 0.0})
    assert abs(rm.book_percent(back.chapter, back.fraction, text.count) - rm.book_percent(9, 0.0, 12)) <= 1
    missing_own = text.position_from_pref({"href": "section_9999.txt", "chapter": 1, "fraction": 0.0})
    assert missing_own is not None and missing_own.chapter == 1  # a gone name of the open layout: index rule


@needs_cores
def test_document_builder_builds_txt_pages(tmp_path):
    from glossarion_mobile.ui.reader import session as rs
    from glossarion_mobile.ui.reader.document import EMPTY_CHAPTER_TEXT, DocumentBuilder, build_native_blocks

    novel = _novel_txt(tmp_path)
    session = rs.ReaderSession(rs.plan_for_file(str(novel)), engine=rs.DocEngine())
    session.load()
    builder = DocumentBuilder(session, lambda key, source=None: "/img/x")
    built = builder.build(0, settings=rm.ReaderSettings(), layout=rm.LAYOUT_SINGLE, theme=THEME, doc_id="t1",
                          event_url="/ev")
    assert built.has_page_bridge and "1. " in built.html and "<p>" in built.html
    blocks = build_native_blocks(session, 0)
    assert blocks and blocks[0].kind == "paragraph"
    builder.close()
    empty = tmp_path / "Library" / "empty.txt"
    empty.write_bytes(b"")
    session = rs.ReaderSession(rs.plan_for_file(str(empty)), engine=rs.DocEngine())
    session.load()
    page = DocumentBuilder(session, lambda key, source=None: "").build(
        0, settings=rm.ReaderSettings(), layout=rm.LAYOUT_SINGLE, theme=THEME, doc_id="t2", event_url="/ev")
    assert session.count == 1 and EMPTY_CHAPTER_TEXT in page.html
