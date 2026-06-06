"""Helpers for safely rehydrating escaped HTML tag entities."""

from __future__ import annotations

import html
import re


VALID_ENTITY_TAGS = frozenset({
    'html', 'head', 'body', 'title', 'meta', 'link', 'style', 'noscript',
    'p', 'div', 'span', 'br', 'hr', 'img', 'a', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
    'ul', 'ol', 'li', 'dl', 'dt', 'dd',
    'pre', 'code', 'em', 'strong', 'b', 'i', 'u', 's', 'strike', 'del', 'ins', 'mark',
    'small', 'sub', 'sup',
    'table', 'thead', 'tbody', 'tr', 'td', 'th', 'caption', 'col', 'colgroup',
    'blockquote', 'q', 'cite',
    'section', 'article', 'header', 'footer', 'nav', 'main', 'aside', 'details', 'summary',
    'figure', 'figcaption',
    'form', 'input', 'button', 'select', 'option', 'textarea', 'label', 'fieldset', 'legend',
    'iframe', 'canvas', 'svg', 'math',
    'video', 'audio', 'source', 'track', 'embed', 'object', 'param',
    'map', 'area',
    'ruby', 'rt', 'rp', 'rb', 'rtc',
    'center', 'font', 'base',
})

_TAG_ENTITY_RE = re.compile(
    r'(&lt;|&LT;|&#0*60;|&#x0*3[cC];)(.*?)(&gt;|&GT;|&#0*62;|&#x0*3[eE];)',
    re.DOTALL,
)
_GT_ENTITY = r'(?:&gt;|&GT;|&#0*62;|&#x0*3[eE];|>)'
_STRAY_P_GT_TEXT = rf'(?<![\w])/?p\s*{_GT_ENTITY}'
_STRAY_P_GT_AT_P_START_RE = re.compile(
    rf'(<p\b[^>]*>\s*){_STRAY_P_GT_TEXT}\s*',
    re.IGNORECASE,
)
_STRAY_P_GT_AT_P_END_RE = re.compile(
    rf'\s*{_STRAY_P_GT_TEXT}(\s*</p>)',
    re.IGNORECASE,
)
_STRAY_P_GT_BETWEEN_P_RE = re.compile(
    rf'(</p>)\s*{_STRAY_P_GT_TEXT}\s*(?=<p\b)',
    re.IGNORECASE,
)


def unescape_valid_html_tag_entities(text: str) -> str:
    """Turn encoded known HTML tags into real tags while preserving narrative angle text."""
    if not isinstance(text, str) or '&' not in text:
        return text

    def repl(match: re.Match) -> str:
        inner = html.unescape(match.group(2))
        stripped = inner.strip()
        if not stripped:
            return match.group(0)
        tag_bits = stripped[1:].lstrip() if stripped.startswith('/') else stripped
        if tag_bits.startswith(('!', '?')):
            return match.group(0)
        tag_name = re.split(r'[\s/>]', tag_bits, 1)[0].lower()
        if tag_name.endswith('/'):
            tag_name = tag_name[:-1]
        if tag_name in VALID_ENTITY_TAGS:
            return f'<{inner}>'
        return match.group(0)

    return _TAG_ENTITY_RE.sub(repl, text)


def fix_stray_p_gt_artifacts(text: str) -> str:
    """Remove stray paragraph-tag crumbs like p&gt; emitted as visible text."""
    if not isinstance(text, str):
        return text
    lowered = text.lower()
    if 'p' not in lowered or ('&gt;' not in lowered and '&#' not in lowered and '>' not in text):
        return text

    text = _STRAY_P_GT_BETWEEN_P_RE.sub(r'\1', text)
    text = _STRAY_P_GT_AT_P_START_RE.sub(r'\1', text)
    text = _STRAY_P_GT_AT_P_END_RE.sub(r'\1', text)
    return text
