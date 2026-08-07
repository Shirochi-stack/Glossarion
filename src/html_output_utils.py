"""Helpers for writing browser-readable HTML output."""

from __future__ import annotations

import os
import re
from copy import deepcopy


_CHARSET_META_RE = re.compile(
    r"<meta\b[^>]*(?:charset\s*=|http-equiv\s*=\s*['\"]?content-type['\"]?[^>]*charset)",
    re.IGNORECASE,
)

_PARAGRAPH_BOUNDARY_RE = (
    r"(?:"
    r"<(?:address|article|aside|blockquote|details|dialog|div|dl|fieldset|"
    r"figcaption|figure|footer|form|h[1-6]|header|hgroup|hr|main|menu|nav|"
    r"ol|p|pre|search|section|table|ul)\b"
    r"|</(?:address|article|aside|blockquote|body|details|dialog|div|fieldset|"
    r"figcaption|figure|footer|form|header|hgroup|html|li|main|nav|section|"
    r"td|th)\s*>"
    r")"
)
_PARAGRAPH_ELEMENT_RE = re.compile(
    r"<p\b(?:[^>\"']|\"[^\"]*\"|'[^']*')*>"
    r"(?:(?!</p\s*>|" + _PARAGRAPH_BOUNDARY_RE + r").)*"
    r"(?:</p\s*>|(?=" + _PARAGRAPH_BOUNDARY_RE + r")|\Z)",
    re.IGNORECASE | re.DOTALL,
)


def normalize_br_terminated_paragraphs(
    content: str,
    add_empty_paragraph_after_break: bool = False,
) -> str:
    """Treat ``<br>`` inside ``<p>`` as a logical paragraph boundary.

    Each non-empty break-delimited portion becomes a valid sibling
    ``<p>...</p>`` element. When ``add_empty_paragraph_after_break`` is true,
    every converted break also emits a whitespace-only ``<p> </p>`` spacer.
    Otherwise, a trailing break closes the current logical paragraph without
    creating an empty extra paragraph. All common break spellings (``<br>``,
    ``<br/>``, and ``<br />``) are accepted.

    Inline markup spanning a break is cloned into both resulting paragraphs,
    so formatting is not lost or left with invalid cross-paragraph nesting.
    """
    text = "" if content is None else str(content)
    if '<br' not in text.lower() or '<p' not in text.lower():
        return text
    try:
        from bs4 import BeautifulSoup, NavigableString, Tag
    except Exception:
        return text

    def _split_contents(parent, soup):
        segments = [[]]
        for child in parent.contents:
            if isinstance(child, Tag) and child.name.lower() == 'br':
                segments.append([])
                # html.parser can represent ``<br /> text`` as a br node
                # containing that following text. Keep it as the first
                # content of the new logical paragraph instead of dropping it.
                if child.contents:
                    trailing_segments = _split_contents(child, soup)
                    for index, trailing_segment in enumerate(trailing_segments):
                        segments[-1].extend(trailing_segment)
                        if index < len(trailing_segments) - 1:
                            segments.append([])
                continue

            if isinstance(child, Tag) and child.find('br') is not None:
                child_segments = _split_contents(child, soup)
                for index, child_segment in enumerate(child_segments):
                    if child_segment:
                        clone = soup.new_tag(child.name)
                        clone.attrs = deepcopy(child.attrs)
                        for node in child_segment:
                            clone.append(node)
                        segments[-1].append(clone)
                    if index < len(child_segments) - 1:
                        segments.append([])
                continue

            if isinstance(child, NavigableString) and not segments[-1]:
                child_text = str(child).lstrip('\r\n')
                if not child_text:
                    continue
                segments[-1].append(NavigableString(child_text))
            else:
                segments[-1].append(deepcopy(child))
        return segments

    def _has_content(nodes):
        for node in nodes:
            if isinstance(node, NavigableString):
                if node.strip():
                    return True
            elif str(node).strip():
                return True
        return False

    def _normalize_paragraph_match(match):
        fragment = match.group(0)
        if '<br' not in fragment.lower():
            return fragment

        # Parse only the paragraph being changed. Parsing the entire document
        # lets HTML parsers repair and reserialize unrelated markup such as
        # surrounding <ul>/<li> lists, which this postprocessor must not touch.
        soup = BeautifulSoup(fragment, 'html.parser')
        paragraph = soup.find('p')
        if paragraph is None or paragraph.find('br') is None:
            return fragment

        replacements = []
        segments = _split_contents(paragraph, soup)
        for index, segment in enumerate(segments):
            if _has_content(segment):
                replacement = soup.new_tag('p')
                replacement.attrs = deepcopy(paragraph.attrs)
                for node in segment:
                    replacement.append(node)
                replacements.append(str(replacement))
            if add_empty_paragraph_after_break and index < len(segments) - 1:
                replacements.append('<p> </p>')
        return ''.join(replacements)

    return _PARAGRAPH_ELEMENT_RE.sub(_normalize_paragraph_match, text)


def ensure_utf8_html_document(content: str) -> str:
    """Return HTML that tells browsers to decode the file as UTF-8."""
    text = "" if content is None else str(content)
    text = text.lstrip("\ufeff")

    if _CHARSET_META_RE.search(text[:4096]):
        return text

    meta = '<meta charset="utf-8">'

    if re.search(r"<head\b[^>]*>", text, re.IGNORECASE):
        return re.sub(
            r"(<head\b[^>]*>)",
            r"\1\n    " + meta,
            text,
            count=1,
            flags=re.IGNORECASE,
        )

    if re.search(r"<html\b[^>]*>", text, re.IGNORECASE):
        return re.sub(
            r"(<html\b[^>]*>)",
            r"\1\n<head>\n    " + meta + "\n</head>",
            text,
            count=1,
            flags=re.IGNORECASE,
        )

    doctype = "<!DOCTYPE html>"
    doctype_match = re.match(r"\s*(<!doctype[^>]*>)\s*", text, re.IGNORECASE)
    if doctype_match:
        doctype = doctype_match.group(1)
        text = text[doctype_match.end():]

    return f'{doctype}\n<html>\n<head>\n    {meta}\n</head>\n<body>\n{text}\n</body>\n</html>'


def write_utf8_html_file(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(ensure_utf8_html_document(content))
