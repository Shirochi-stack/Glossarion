"""Shared policy and helpers for translating HTML/XHTML ``<title>`` tags."""

from __future__ import annotations

import html
import os
import re

from bs4 import BeautifulSoup


_TRUE_VALUES = {"1", "true", "yes", "on"}
_TITLE_TAG_RE = re.compile(
    r"(<title\b[^>]*>)(.*?)(</title\s*>)",
    flags=re.IGNORECASE | re.DOTALL,
)


def _env_true(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in _TRUE_VALUES


def should_translate_title_tags() -> bool:
    """Return whether document title tags should be sent for translation.

    ``SKIP_TITLE_TAG_TRANSLATION`` is authoritative and defaults to false,
    which means title tags are translated.  ``USE_TITLE`` remains a fallback
    for old launchers/configurations that have not learned the new setting.
    """
    if "SKIP_TITLE_TAG_TRANSLATION" in os.environ:
        return not _env_true("SKIP_TITLE_TAG_TRANSLATION")
    if "USE_TITLE" in os.environ:
        return _env_true("USE_TITLE")
    return True


def title_tag_translation_payload(markup: str) -> str:
    """Return only non-empty title tags from *markup*, preserving their tags."""
    if not isinstance(markup, str) or not markup.strip():
        return ""
    soup = BeautifulSoup(markup, "html.parser")
    return "\n".join(
        str(tag)
        for tag in soup.find_all("title")
        if tag.get_text(" ", strip=True)
    )


def restore_translated_title_tags(original_markup: str, translated: str):
    """Put translated title text back into the untouched source document.

    The original markup is changed only inside matching ``<title>`` elements,
    so image references, namespaces, declarations, and body formatting survive
    image-only copy paths byte-for-byte.  ``None`` signals an unusable response.
    """
    if not isinstance(original_markup, str) or not original_markup.strip():
        return None

    original_matches = list(_TITLE_TAG_RE.finditer(original_markup))
    if not original_matches:
        return None

    value = str(translated or "").strip()
    value = re.sub(
        r"^```(?:x?html)?\s*(?:\r?\n)?",
        "",
        value,
        count=1,
        flags=re.IGNORECASE,
    )
    value = re.sub(r"(?:\r?\n)?```\s*$", "", value, count=1).strip()

    translated_soup = BeautifulSoup(value, "html.parser")
    translated_titles = [
        tag.get_text(" ", strip=True)
        for tag in translated_soup.find_all("title")
        if tag.get_text(" ", strip=True)
    ]
    if not translated_titles and len(original_matches) == 1:
        # Some providers obey the translation request but return only the title
        # text. Accept that compact form, but never treat arbitrary HTML as it.
        if value and not re.search(r"<[A-Za-z!/][^>]*>", value):
            plain = BeautifulSoup(value, "html.parser").get_text(" ", strip=True)
            if plain:
                translated_titles = [plain]

    if len(translated_titles) != len(original_matches):
        return None

    title_iter = iter(translated_titles)

    def _replace(match):
        title_text = html.escape(next(title_iter), quote=False)
        return f"{match.group(1)}{title_text}{match.group(3)}"

    return _TITLE_TAG_RE.sub(_replace, original_markup)
