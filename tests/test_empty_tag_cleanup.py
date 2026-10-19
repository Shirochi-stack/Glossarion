"""'Remove excessive empty p and br tags' (html_duplicate_cleanup.collapse_empty_paragraphs and the
On-device / all-models toggles that decide when chapter text sent to the model gets it)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import html_duplicate_cleanup as cleanup  # noqa: E402
from html_duplicate_cleanup import collapse_empty_paragraphs  # noqa: E402

OWNER_EXAMPLE = ("…없었다.<br/></p><p><br/></p><p><br/></p><p>- 메카물! 좋아합니다!<br/></p>"
                 "<p><br/></p><p><br/></p><p><br/></p><p>- 오 정말요?…")


def test_owner_example_keeps_two_breaks_per_run():
    assert collapse_empty_paragraphs(OWNER_EXAMPLE) == (
        "…없었다.<br/></p><br/><br/><p>- 메카물! 좋아합니다!<br/></p><br/><br/><p>- 오 정말요?…")


def test_one_blank_line_stays_one_break():
    assert collapse_empty_paragraphs("<p>a</p><p><br/></p><p>b</p>") == "<p>a</p><br/><p>b</p>"
    # an empty paragraph without a <br> still stands for one blank line
    assert collapse_empty_paragraphs("<p>a</p>\n<p></p>\n<p>b</p>") == "<p>a</p>\n<br/>\n<p>b</p>"


def test_attributes_case_and_self_closing_forms():
    html = ('<p class="txt">a</p>\n<p class="blank" style="margin:0"> </p>\n'
            '<P id="x"><BR class="calibre" /></P>\n<p/>\n<p class="txt">b</p>')
    assert collapse_empty_paragraphs(html) == '<p class="txt">a</p>\n<br/><br/>\n<p class="txt">b</p>'


@pytest.mark.parametrize("blank", ["&nbsp;", "&#160;", "&#xA0;", "\xa0", " \n\t "])
def test_nbsp_and_whitespace_only_paragraphs_are_empty(blank):
    html = f"<p>a</p><p>{blank}</p><p>{blank}<br/>{blank}</p><p>{blank}</p><p>b</p>"
    assert collapse_empty_paragraphs(html) == "<p>a</p><br/><br/><p>b</p>"


def test_break_runs_inside_a_paragraph_keep_two():
    assert collapse_empty_paragraphs("<p>a<br><br><br><br>b</p>") == "<p>a<br/><br/>b</p>"
    assert collapse_empty_paragraphs("<p>a<br/>\n<br />\n<br/></p>") == "<p>a<br/><br/></p>"


def test_paragraphs_with_text_or_elements_are_kept():
    html = ('<p><img src="a.png" alt=""/></p><p><br/></p><p><br/></p><p><br/></p>'
            '<p><span></span></p><p><br/><img src="b.png"/></p><p>&nbsp;x</p>')
    assert collapse_empty_paragraphs(html) == (
        '<p><img src="a.png" alt=""/></p><br/><br/>'
        '<p><span></span></p><p><br/><img src="b.png"/></p><p>&nbsp;x</p>')


def test_clean_html_is_unchanged():
    clean = ('<html><head><title>T</title></head><body>\n<h1>Chapter 1</h1>\n'
             '<p>line one<br/>line two</p>\n<p>para<br><br>two breaks</p>\n<pre>x</pre><param name="a"/>'
             '<p class="x">text</p>\n</body></html>')
    assert collapse_empty_paragraphs(clean) == clean
    for value in ("", "plain text\n\n\nmore", "# Markdown\n\nbody", None):
        assert collapse_empty_paragraphs(value) == value


def test_idempotent():
    once = collapse_empty_paragraphs(OWNER_EXAMPLE + "<p>a<br/><br/><br/></p><p> </p><p></p>")
    assert collapse_empty_paragraphs(once) == once


@pytest.fixture
def toggles(monkeypatch):
    for name in (cleanup.EMPTY_TAGS_ON_DEVICE_ENV, cleanup.EMPTY_TAGS_ALL_MODELS_ENV, "MODEL",
                 "GLOSSARION_LOCAL_LLM_URL"):
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def test_on_device_toggle_is_on_by_default_and_only_for_the_on_device_engine(toggles):
    toggles.setenv("MODEL", "local/qwen3.5-0.8b")
    # desktop: nothing publishes the on-device endpoint, so a local/ llama-server is left alone
    assert cleanup.prepare_html_for_model(OWNER_EXAMPLE) == OWNER_EXAMPLE
    toggles.setenv("GLOSSARION_LOCAL_LLM_URL", "http://127.0.0.1:5123/tok/v1")
    assert cleanup.should_collapse_empty_paragraphs() is True
    assert cleanup.prepare_html_for_model(OWNER_EXAMPLE) == collapse_empty_paragraphs(OWNER_EXAMPLE)
    assert cleanup.should_collapse_empty_paragraphs("gpt-4o") is False
    toggles.setenv(cleanup.EMPTY_TAGS_ON_DEVICE_ENV, "0")
    assert cleanup.prepare_html_for_model(OWNER_EXAMPLE) == OWNER_EXAMPLE


def test_all_models_toggle_is_off_by_default(toggles):
    toggles.setenv("MODEL", "gemini-2.5-pro")
    assert cleanup.should_collapse_empty_paragraphs() is False
    toggles.setenv(cleanup.EMPTY_TAGS_ALL_MODELS_ENV, "1")
    assert cleanup.should_collapse_empty_paragraphs() is True
    assert cleanup.prepare_html_for_model(OWNER_EXAMPLE) == collapse_empty_paragraphs(OWNER_EXAMPLE)
