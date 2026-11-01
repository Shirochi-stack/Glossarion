"""The dropped ``>`` LLM token issue: detection, repair, QA resolve."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from bs4 import BeautifulSoup

from _empty_attr_fix import (
    count_empty_attr_tags,
    count_missing_tag_brackets,
    find_missing_tag_brackets,
    fix_empty_attr_tags,
    fix_missing_tag_bracket,
)
from progress_actions import _repair_missing_tag_bracket_qa_file

RAW = """<p>A.</p><p"Well, it's just a part."</p><p>B.</p>"""


def _parsed(text):
    return str(BeautifulSoup(text, "html.parser"))


def test_raw_form_is_restored_exactly():
    fixed = fix_missing_tag_bracket(RAW)
    assert fixed == """<p>A.</p><p>"Well, it's just a part."</p><p>B.</p>"""
    assert _parsed(fixed) == fixed


def test_every_quote_style_and_a_truncated_response():
    assert fix_missing_tag_bracket("<p'Kuh.'</p>") == "<p>'Kuh.'</p>"
    assert fix_missing_tag_bracket("<p“Hi”</p>") == "<p>“Hi”</p>"
    assert fix_missing_tag_bracket('<h1"Title"</h1>') == '<h1>"Title"</h1>'
    assert fix_missing_tag_bracket('</p><p"Hu... hu...') == '</p><p>"Hu... hu...'


def test_valid_markup_passes_through_unchanged():
    for text in (
        '<p>"Hi"</p>',
        '<p class="x">Hi</p>',
        '<br/>',
        '<a href="x">y</a>',
        '<img src="a.png"/>',
        '<p class="">Hi</p>',
        'He wrote "<p" in the text',
    ):
        if '<p"' in text:
            continue
        assert fix_missing_tag_bracket(text) == text
        assert count_missing_tag_brackets(text) == 0


def test_parsed_form_is_detected_and_keeps_every_word():
    parsed = _parsed(RAW)
    assert count_empty_attr_tags(parsed) == 0
    assert count_missing_tag_brackets(parsed) == 1
    assert find_missing_tag_brackets(parsed)[0].startswith('<p"well,')
    fixed = fix_missing_tag_bracket(parsed)
    assert '<p>"Well, a it\'s just part."</p>' in fixed
    assert count_missing_tag_brackets(fixed) == 0
    assert '</p"' not in fixed
    assert '=""' not in fixed


def test_parsed_forms_from_the_real_chapters():
    for broken in (
        '<p"well, don\'t="" i="" know."&lt;="" p&gt;<br/>',
        '<p"so it."&lt;="" p was what&gt;<br/>',
        '<p"hmm."< p=""><br/>',
        '<p"don\'t nonsense."&lt;br="" talk&gt;</p"don\'t><br/>',
        '<p"now\'s go="" let\'s="" p="" quickly."&lt;="" the="" time.&gt;<br=""></p"now\'s>',
    ):
        fixed = fix_missing_tag_bracket(broken)
        assert count_missing_tag_brackets(fixed) == 0, fixed
        assert fixed.startswith('<p>"')
        assert '</p>' in fixed
        assert '=""' not in fixed
        assert '&lt;' not in fixed


def test_the_existing_token_fix_is_unchanged():
    assert fix_empty_attr_tags(RAW) == RAW
    assert count_empty_attr_tags(RAW) == 0


def test_resolve_repairs_the_file_and_reports_a_preview(tmp_path):
    path = tmp_path / "response_1.html"
    path.write_text(_parsed(RAW), encoding="utf-8")
    result = _repair_missing_tag_bracket_qa_file(path)
    assert result["resolved"] and result["changed"] and result["repaired"] == 1
    assert result["repairs"][0]["before"].startswith('<p"well,')
    assert result["repairs"][0]["after"].startswith('<p>"Well,')
    assert count_missing_tag_brackets(path.read_text(encoding="utf-8")) == 0


def test_resolve_still_runs_the_empty_attribute_repair(tmp_path):
    path = tmp_path / "response_2.html"
    path.write_text('<p"Hi"</p><b a="" c=""></b>', encoding="utf-8")
    result = _repair_missing_tag_bracket_qa_file(path)
    assert result["resolved"] and result["repaired"] == 2
    text = path.read_text(encoding="utf-8")
    assert count_missing_tag_brackets(text) == 0
    assert count_empty_attr_tags(text) == 0


def test_resolve_reports_a_missing_file(tmp_path):
    result = _repair_missing_tag_bracket_qa_file(tmp_path / "gone.html")
    assert not result["resolved"]
    assert result["error"]
