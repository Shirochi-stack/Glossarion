"""Helpers for writing browser-readable HTML output."""

from __future__ import annotations

import os
import re
import html as html_lib
import shutil
import tempfile


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
_PARAGRAPH_OPEN_TAG_RE = re.compile(
    r"<(?P<name>p)\b(?:[^>\"']|\"[^\"]*\"|'[^']*')*>",
    re.IGNORECASE | re.DOTALL,
)
_PARAGRAPH_CLOSE_TAG_RE = re.compile(r"</p\s*>\Z", re.IGNORECASE)
_HTML_TOKEN_RE = re.compile(
    r"<!--.*?-->|<!\[CDATA\[.*?\]\]>|<\?.*?\?>|<![^>]*>"
    r"|</?[A-Za-z][\w:.-]*\b(?:[^>\"']|\"[^\"]*\"|'[^']*')*/?\s*>",
    re.IGNORECASE | re.DOTALL,
)
_HTML_TAG_NAME_RE = re.compile(
    r"<\s*(?P<closing>/)?\s*(?P<name>[A-Za-z][\w:.-]*)\b",
    re.IGNORECASE,
)
_HTML_VOID_TAGS = frozenset({
    'area', 'base', 'br', 'col', 'embed', 'hr', 'img', 'input', 'link',
    'meta', 'param', 'source', 'track', 'wbr',
})
_CONTENT_ONLY_TAG_RE = re.compile(
    r"<(?:audio|canvas|embed|figure|iframe|img|math|object|picture|svg|video)\b",
    re.IGNORECASE,
)
_BR_ELEMENT_RE = re.compile(
    r"<br\b(?:[^>\"']|\"[^\"]*\"|'[^']*')*/?\s*>",
    re.IGNORECASE | re.DOTALL,
)
_HTML_OUTPUT_EXTENSIONS = ('.html', '.htm', '.xhtml')


def convert_br_to_paragraphs(content: str) -> str:
    """Split ``<br>``-delimited content inside ``<p>`` into sibling paragraphs.

    This is a lexical transformation: it never reparses or serializes the
    surrounding document, so list markup and other unrelated HTML remain
    byte-for-byte unchanged. Inline tags spanning a break are closed and
    reopened in the new paragraph.
    """
    text = "" if content is None else str(content)
    if '<br' not in text.lower() or '<p' not in text.lower():
        return text

    def _split_inner_html(inner_html):
        segments = []
        current = []
        open_inline_tags = []
        cursor = 0
        strip_leading_linebreaks = False

        for token_match in _HTML_TOKEN_RE.finditer(inner_html):
            raw_text = inner_html[cursor:token_match.start()]
            if strip_leading_linebreaks:
                raw_text = raw_text.lstrip('\r\n')
                if raw_text:
                    strip_leading_linebreaks = False
            current.append(raw_text)
            token = token_match.group(0)
            cursor = token_match.end()
            tag_match = _HTML_TAG_NAME_RE.match(token)
            if tag_match is None:
                current.append(token)
                continue

            tag_name = tag_match.group('name').lower()
            is_closing = bool(tag_match.group('closing'))
            is_self_closing = token.rstrip().endswith('/>')

            if tag_name == 'br' and not is_closing:
                current.extend(
                    f'</{open_name}>'
                    for open_name, _open_token in reversed(open_inline_tags)
                )
                segments.append(''.join(current))
                current = [
                    open_token for _open_name, open_token in open_inline_tags
                ]
                strip_leading_linebreaks = True
                continue

            current.append(token)
            if is_closing:
                for index in range(len(open_inline_tags) - 1, -1, -1):
                    if open_inline_tags[index][0] == tag_name:
                        del open_inline_tags[index:]
                        break
            elif tag_name not in _HTML_VOID_TAGS and not is_self_closing:
                open_inline_tags.append((tag_name, token))

        remaining_text = inner_html[cursor:]
        if strip_leading_linebreaks:
            remaining_text = remaining_text.lstrip('\r\n')
        current.append(remaining_text)
        current.extend(
            f'</{open_name}>'
            for open_name, _open_token in reversed(open_inline_tags)
        )
        segments.append(''.join(current))
        return segments

    def _has_content(segment):
        if _CONTENT_ONLY_TAG_RE.search(segment):
            return True
        visible_text = _HTML_TOKEN_RE.sub('', segment)
        visible_text = html_lib.unescape(visible_text).replace('\xa0', ' ')
        return bool(visible_text.strip())

    def _convert_paragraph(match):
        fragment = match.group(0)
        if '<br' not in fragment.lower():
            return fragment
        opening_match = _PARAGRAPH_OPEN_TAG_RE.match(fragment)
        if opening_match is None:
            return fragment
        closing_match = _PARAGRAPH_CLOSE_TAG_RE.search(fragment)
        closing_tag = (
            closing_match.group(0)
            if closing_match is not None
            else f"</{opening_match.group('name')}>"
        )
        inner_end = (
            closing_match.start() if closing_match is not None else len(fragment)
        )
        inner_html = fragment[opening_match.end():inner_end]
        opening_tag = opening_match.group(0)
        return ''.join(
            f'{opening_tag}{segment}{closing_tag}'
            for segment in _split_inner_html(inner_html)
            if _has_content(segment)
        )

    return _PARAGRAPH_ELEMENT_RE.sub(_convert_paragraph, text)


def convert_br_in_output_folder(output_dir: str) -> dict:
    """Apply :func:`convert_br_to_paragraphs` to root HTML output files.

    Translation outputs live at the output-folder root. Deliberately avoiding
    recursion keeps extracted EPUB trees, backups, reports, and sidecars
    untouched. Files are replaced atomically and an existing UTF-8 BOM is
    preserved. The returned dictionary contains a full per-file audit suitable
    for a GUI summary and a detailed text log.
    """
    root = os.path.abspath(os.fspath(output_dir))
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Output folder not found: {root}")

    html_files = []
    with os.scandir(root) as entries:
        for entry in entries:
            try:
                is_file = entry.is_file()
            except OSError:
                is_file = False
            if is_file and entry.name.lower().endswith(_HTML_OUTPUT_EXTENSIONS):
                html_files.append(entry.path)
    html_files.sort(key=lambda path: os.path.basename(path).casefold())

    audit = {
        'output_dir': root,
        'scanned': len(html_files),
        'changed': 0,
        'unchanged': 0,
        'failed': 0,
        'files': [],
    }
    utf8_bom = b'\xef\xbb\xbf'

    for path in html_files:
        record = {
            'path': path,
            'status': 'unchanged',
            'breaks_before': 0,
            'breaks_after': 0,
        }
        temp_path = None
        try:
            with open(path, 'rb') as handle:
                original_bytes = handle.read()
            had_bom = original_bytes.startswith(utf8_bom)
            original = original_bytes[len(utf8_bom):].decode('utf-8') if had_bom else original_bytes.decode('utf-8')
            record['breaks_before'] = len(_BR_ELEMENT_RE.findall(original))
            converted = convert_br_to_paragraphs(original)
            record['breaks_after'] = len(_BR_ELEMENT_RE.findall(converted))

            if converted == original:
                audit['unchanged'] += 1
                audit['files'].append(record)
                continue

            encoded = converted.encode('utf-8')
            if had_bom:
                encoded = utf8_bom + encoded
            descriptor, temp_path = tempfile.mkstemp(
                prefix=f'.{os.path.basename(path)}.',
                suffix='.tmp',
                dir=os.path.dirname(path),
            )
            with os.fdopen(descriptor, 'wb') as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                shutil.copymode(path, temp_path)
            except OSError:
                pass
            os.replace(temp_path, path)
            temp_path = None

            record['status'] = 'converted'
            audit['changed'] += 1
        except Exception as exc:
            record['status'] = 'failed'
            record['error'] = str(exc)
            audit['failed'] += 1
        finally:
            if temp_path:
                try:
                    os.remove(temp_path)
                except OSError:
                    pass
        audit['files'].append(record)

    return audit


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
