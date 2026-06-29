"""Helpers for writing browser-readable HTML output."""

from __future__ import annotations

import os
import re


_CHARSET_META_RE = re.compile(
    r"<meta\b[^>]*(?:charset\s*=|http-equiv\s*=\s*['\"]?content-type['\"]?[^>]*charset)",
    re.IGNORECASE,
)


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
