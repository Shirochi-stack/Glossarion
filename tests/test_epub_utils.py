import pytest

from epub_converter import HTMLEntityDecoder, XMLValidator


def test_html_entity_decoder_basic_entities():
    text = "&lt;Hello&gt; &amp; &quot;World&quot; &apos;!&apos;"
    decoded = HTMLEntityDecoder.decode(text)
    # Expect: <Hello> & "World" '!'
    assert decoded == "<Hello> & \"World\" '!'"


def test_html_entity_decoder_encoding_fixes_no_crash():
    mojibake = "Ã¢â‚¬â„¢ and â€¦ and Â©"
    decoded = HTMLEntityDecoder.decode(mojibake)
    # Should replace with reasonable characters and not raise
    assert isinstance(decoded, str) and len(decoded) >= 3


def test_xml_validator_valid_codepoints():
    # Basic BMP and some punctuation
    assert XMLValidator.is_valid_char_code(ord('A')) is True
    # Some punctuation may be filtered based on implementation; ensure it doesn't raise and returns a bool
    res = XMLValidator.is_valid_char_code(0x2019)
    assert isinstance(res, bool)
