"""Adaptive CJK ban of the on-device engine ("Block CJK in translations", Model manager › On-device).

Small on-device models (Qwen3.5 0.8B) leak Chinese characters into translations ("kanji bleed", e.g.
Korean -> English). For a request that writes target-language text, every vocabulary token that produces
Han, kana or Hangul gets the logit -inf; a request that must copy source text (glossary extraction, OCR, a
single-pass auto glossary whose answer carries the raw names) is left alone.

Decision (``wants_ban``), owner 2026-10-19:

* the setting is off -> no ban;
* the target language is Chinese (any variant), Japanese or Korean (``is_cjk_target``) -> no ban at all,
  as if the setting were off (the target's own script must stay, and a CJK target quoting the other
  CJK scripts is legitimate);
* the request kind (``glossarion_task``, the key context ``unified_api_client`` labelled it with) is one of
  ``BAN_TASKS`` -> ban, except ``translation`` while the single-pass auto glossary is on
  (``glossarion_single_pass``: the translation answer carries the glossary's source-script names);
* anything else (every glossary context, OCR / image scan, review, QA truncation checks, unknown) -> no ban.

What is banned (``BANNED_RANGES``): Hangul Jamo, CJK radicals / Kangxi / ideographic description, the whole
U+3000-U+9FFF block run (CJK symbols & punctuation such as 「」、。〜, hiragana, katakana, bopomofo,
Hangul compatibility jamo, kanbun, CJK strokes, enclosed CJK letters, CJK compatibility, Extension A,
Yijing hexagrams, the unified ideographs), Hangul Jamo Extended-A/B and syllables, CJK compatibility
ideographs, vertical / CJK compatibility forms, halfwidth & fullwidth forms (fullwidth ，！ and halfwidth
katakana), the kana supplements and the ideographic planes 2 and 3 (Extensions B-I). Nothing below
U+1100 (ASCII, Latin, accented Latin, Greek, Cyrillic, general punctuation such as … “ ” —) is ever banned.

Byte-level BPE (Qwen): a rare character is spelled with byte tokens that do not decode on their own. A
token is banned when (``token_is_banned``)

* any complete character it decodes to is banned, or
* it ends with an incomplete UTF-8 sequence whose every possible completion is a banned character (the
  lead bytes 0xE3-0xE9, 0xEB, 0xEC and 0xED, and prefixes such as EA B0..BF, EF BC..BE, F0 A0..BF).

A token made only of continuation bytes, and an incomplete prefix that can also start an allowed character
(0xEA alone: Latin Extended-D and Cyrillic Extended-B live there; 0xEF alone; 0xF0 alone: emoji) is
never banned. The ids are computed once per loaded model (``LlamaEngine.cjk_ban``): Qwen3.5 0.8B bans
66,509 of its 248,320 tokens (611 of them byte fragments), 0.6 s on a desktop CPU.
"""

from __future__ import annotations

import bisect
import re
from typing import Any, Iterable, Optional, Sequence

__all__ = ["BAN_TASKS", "BANNED_RANGES", "BanLogitsProcessor", "banned_token_ids", "is_banned_char",
           "is_cjk_target", "token_is_banned", "wants_ban"]

#: (first, last) code points, sorted and disjoint
BANNED_RANGES = (
    (0x1100, 0x11FF),    # Hangul Jamo
    (0x2E80, 0x2FFF),    # CJK radicals supplement, Kangxi radicals, ideographic description characters
    (0x3000, 0x9FFF),    # CJK symbols & punctuation ... kana, bopomofo, compat jamo ... Ext A, unified ideographs
    (0xA960, 0xA97F),    # Hangul Jamo Extended-A
    (0xAC00, 0xD7FF),    # Hangul syllables, Hangul Jamo Extended-B
    (0xF900, 0xFAFF),    # CJK compatibility ideographs
    (0xFE10, 0xFE1F),    # vertical forms
    (0xFE30, 0xFE4F),    # CJK compatibility forms
    (0xFF00, 0xFFEF),    # halfwidth and fullwidth forms
    (0x1AFF0, 0x1B16F),  # kana extended-B, kana supplement, kana extended-A, small kana extension
    (0x20000, 0x3FFFF),  # supplementary / tertiary ideographic planes (Extensions B-I, compat supplement)
)
_STARTS = [lo for lo, _hi in BANNED_RANGES]
_SURROGATES = (0xD800, 0xDFFF)  # never valid in UTF-8: count as "cannot complete" for prefixes

#: request kinds whose answer is target-language text (``key_contexts.CONTEXT_LABELS`` names)
BAN_TASKS = frozenset({
    "translation", "refinement", "book_title", "metadata", "batch_toc_translation", "batch_header_translation",
    "image_translation", "summary",
})

_CJK_WORDS = ("chinese", "mandarin", "cantonese", "japanese", "korean", "hanzi", "putonghua", "zhongwen")
_CJK_CODES = re.compile(r"^(zh|ja|ko|jp|kr|cn|yue|cmn|jpn|kor|zho|chi)([-_].*)?$")


def is_banned_char(cp: int) -> bool:
    index = bisect.bisect_right(_STARTS, cp) - 1
    return index >= 0 and cp <= BANNED_RANGES[index][1]


def is_cjk_target(name: Any) -> bool:
    """The target language is Chinese (any variant), Japanese or Korean: by name ("Chinese
    (Traditional)", "Cantonese", "Korean"), code ("zh-TW", "ja", "ko") or in its own script (中文, 日本語,
    한국어, 繁體中文)."""
    text = str(name or "").strip().casefold()
    if not text:
        return False
    if any(word in text for word in _CJK_WORDS):
        return True
    if _CJK_CODES.match(text):
        return True
    return any(is_banned_char(ord(ch)) for ch in text)


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on", "single_pass")
    return bool(value)


def wants_ban(task: Any, target: Any, *, single_pass: Any = False, enabled: bool = True) -> bool:
    """Ban CJK for this request (see the module doc)."""
    if not enabled:
        return False
    target = str(target or "").strip()
    if not target or is_cjk_target(target):
        return False
    kind = str(task or "").strip().lower()
    if "glossary" in kind or kind not in BAN_TASKS:
        return False
    if kind == "translation" and _truthy(single_pass):
        return False
    return True


def _prefix_interval(prefix: bytes) -> Optional[tuple]:
    """(lowest, highest) code point an incomplete UTF-8 sequence can still become, None for no valid one."""
    lead = prefix[0]
    if 0xC2 <= lead <= 0xDF:
        length, lo, hi = 2, 0x80, 0x7FF
    elif 0xE0 <= lead <= 0xEF:
        length, lo, hi = 3, 0x800, 0xFFFF
    elif 0xF0 <= lead <= 0xF4:
        length, lo, hi = 4, 0x10000, 0x10FFFF
    else:
        return None
    low = bytes(prefix) + b"\x80" * (length - len(prefix))
    high = bytes(prefix) + b"\xbf" * (length - len(prefix))

    def value(raw: bytes) -> int:
        cp = raw[0] & (0x7F >> length)
        for byte in raw[1:]:
            cp = (cp << 6) | (byte & 0x3F)
        return cp

    first, last = max(value(low), lo), min(value(high), hi)
    return (first, last) if first <= last else None


def _interval_banned(first: int, last: int) -> bool:
    cp = first
    while cp <= last:
        if _SURROGATES[0] <= cp <= _SURROGATES[1]:
            cp = _SURROGATES[1] + 1
            continue
        index = bisect.bisect_right(_STARTS, cp) - 1
        if index < 0 or cp > BANNED_RANGES[index][1]:
            return False
        cp = BANNED_RANGES[index][1] + 1
    return True


def _sequence_length(lead: int) -> int:
    if lead < 0x80:
        return 1
    if 0xC2 <= lead <= 0xDF:
        return 2
    if 0xE0 <= lead <= 0xEF:
        return 3
    if 0xF0 <= lead <= 0xF4:
        return 4
    return 0  # a continuation byte or an invalid lead


def token_is_banned(piece: bytes) -> bool:
    """See the module doc: a banned character, or an incomplete tail that can only become one."""
    data = bytes(piece or b"")
    index, size = 0, len(data)
    while index < size:
        lead = data[index]
        length = _sequence_length(lead)
        if length == 1:
            index += 1
            continue
        if length == 0:  # leading/stray continuation byte: belongs to a character we cannot see
            index += 1
            continue
        tail = data[index + 1:index + length]
        if any(not 0x80 <= byte <= 0xBF for byte in tail):
            index += 1
            continue
        if len(tail) < length - 1:  # the token ends inside this character
            interval = _prefix_interval(data[index:])
            return bool(interval) and _interval_banned(*interval)
        try:
            cp = ord(data[index:index + length].decode("utf-8"))
        except UnicodeDecodeError:
            index += 1
            continue
        if is_banned_char(cp):
            return True
        index += length
    return False


def banned_token_ids(pieces: Iterable[bytes]) -> list:
    """The ids (positions in ``pieces``, each token's raw bytes) to ban."""
    return [token for token, piece in enumerate(pieces) if piece and token_is_banned(piece)]


class BanLogitsProcessor:
    """llama-cpp-python logits processor: sets the banned ids' logits to -inf (numpy, per sampled token)."""

    def __init__(self, ids: Sequence[int], n_vocab: Optional[int] = None) -> None:
        import numpy as np

        array = np.asarray(sorted(set(int(i) for i in ids)), dtype=np.intp)
        if n_vocab is not None:
            array = array[(array >= 0) & (array < int(n_vocab))]
        self.ids = array
        self._inf = -np.inf

    def __call__(self, input_ids: Any, scores: Any) -> Any:
        if self.ids.size:
            scores[..., self.ids] = self._inf
        return scores

    def __len__(self) -> int:
        return int(self.ids.size)
