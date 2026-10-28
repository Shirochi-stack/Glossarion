"""Adaptive script ban: which writing systems a request may not answer in (shared, GUI-free).

One table for desktop and Glossarion Mobile (owner 2026-10-28: "this feature is too cool to not be on
desktop"). Two users:

* the on-device engine (mobile ``services.local_llm.script_ban``, Model manager › On-device switch "Block
  other scripts in translations (adaptive)", Prefs key ``local_llm_block_cjk``): bans the tokens of the
  loaded llama.cpp vocabulary (``token_masks`` here, the -inf logits processor there);
* every API model (Other Settings › Anti-Duplicate Parameters › Logit Bias › "Block other scripts in
  translations (adaptive, all models)", config ``block_other_scripts`` / env ``BLOCK_OTHER_SCRIPTS``,
  only while Enable Anti-Duplicate Parameters is on; ``unified_api_client``): official OpenAI chat models
  get ``openai_logit_bias`` (the provider caps the map, so it can only push down the most frequent
  tokens), and every route gets the output guard ``foreign_script_leak``: a translation whose text has
  characters of a banned script the source does not have is sent again, within the client's own retry
  budget, with ``retry_instruction`` added.

Decision (``banned_scripts``; ``wants_ban`` is its bool), owner 2026-10-19, widened 2026-10-28:

* the setting is off -> no ban;
* the request kind (``glossarion_task``, the key context ``unified_api_client`` labelled it with) is one of
  ``BAN_TASKS`` -> ban, except ``translation`` while the single-pass auto glossary is on
  (``glossarion_single_pass``: the translation answer carries the glossary's source-script names);
  anything else (every glossary context, OCR / image scan, review, QA truncation checks, unknown) and a
  request without a target -> no ban;
* what is banned: every script of ``SCRIPTS`` the target does not use (``target_scripts``: by name
  "Hindi", "Chinese (Traditional)", "Serbian", by code "hi", "zh-TW", "sr-Latn", or written in its own
  script "हिन्दी", "中文", "Русский"). English bans CJK, Devanagari, Thai, Arabic, Cyrillic, Hebrew and the
  rest; Russian keeps Cyrillic; Hindi keeps Devanagari; Punjabi keeps Gurmukhi and Shahmukhi (Arabic). A
  Chinese, Japanese or Korean target keeps all of Han, kana and Hangul (a CJK target quoting the other CJK
  scripts is legitimate) and bans the other scripts. The target language's own script is never banned;
* a target this module does not know ("Other", a custom name in Latin letters) bans CJK only: its own
  script is unknown, and banning it would leave the model nothing to write with.

Never banned, for any target: ASCII, Latin (accented, extended, fullwidth excepted), Greek (also used as
symbols: α, Ω, μ), IPA, combining marks, general punctuation (… “ ” —), letterlike symbols, arrows, maths,
box drawing, emoji, private use. ``SCRIPTS`` lists each script's Unicode blocks; the CJK ranges are those
of the original ban (Hangul Jamo, CJK radicals / Kangxi, the U+3000-U+9FFF run with CJK punctuation such as
「」、。〜, kana, bopomofo and the unified ideographs, Hangul syllables, CJK compatibility and vertical forms,
halfwidth & fullwidth forms, the kana supplements and the ideographic planes 2 and 3).

Byte-level BPE (Qwen, OpenAI's tiktoken): a rare character is spelled with byte tokens that do not decode
on their own. A token is banned when (``token_is_banned``)

* any complete character it decodes to belongs to a banned script, or
* it ends with an incomplete UTF-8 sequence whose every possible completion belongs to a banned script
  (for English: the lead bytes 0xD0-0xD7 and 0xD8-0xDF (Cyrillic, Armenian, Hebrew, Arabic), 0xE0
  (U+0800-U+0FFF: Arabic extensions, the Indic scripts, Thai, Lao, Tibetan), 0xE3-0xE9, 0xEB-0xED (CJK),
  and prefixes such as E0 A4 (Devanagari), EA B0..BF, EF BC..BE, F0 A0..BF).

A token made only of continuation bytes, and an incomplete prefix that can also start an allowed
character (0xCE: Greek; 0xE1: Latin Extended Additional and Greek Extended live there; 0xE2: punctuation;
0xEA, 0xEF; 0xF0: emoji) is never banned.
"""

from __future__ import annotations

import bisect
import functools
import re
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Tuple

__all__ = ["BAN_TASKS", "BANNED_RANGES", "CJK", "LEAK_MIN_CHARS", "LEAK_MIN_SHARE", "OPENAI_LOGIT_BIAS_LIMIT",
           "SCRIPTS", "banned_scripts", "banned_token_ids", "bias_token_ids", "foreign_script_leak",
           "is_banned_char", "is_cjk_target", "openai_encoding_name", "openai_logit_bias", "retry_instruction",
           "script_mask", "script_of", "target_scripts", "token_is_banned", "token_masks", "wants_ban"]

CJK = "cjk"

#: the original CJK ban's (first, last) code points, sorted and disjoint
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

#: script -> its (first, last) Unicode blocks; no two ranges overlap. Neighbouring minor scripts share a
#: group (Syriac, Thaana, NKo, Samaritan and Mandaic go with Arabic); "other" holds scripts no target uses.
SCRIPTS: Dict[str, Tuple[Tuple[int, int], ...]] = {
    CJK: BANNED_RANGES,
    "cyrillic": ((0x0400, 0x052F), (0x1C80, 0x1C8F), (0x2DE0, 0x2DFF), (0xA640, 0xA69F)),
    "armenian": ((0x0530, 0x058F),),
    "hebrew": ((0x0590, 0x05FF), (0xFB1D, 0xFB4F)),
    "arabic": ((0x0600, 0x08FF), (0xFB50, 0xFDFF), (0xFE70, 0xFEFF)),
    "devanagari": ((0x0900, 0x097F), (0x1CD0, 0x1CFF), (0xA8E0, 0xA8FF)),
    "bengali": ((0x0980, 0x09FF),),
    "gurmukhi": ((0x0A00, 0x0A7F),),
    "gujarati": ((0x0A80, 0x0AFF),),
    "oriya": ((0x0B00, 0x0B7F),),
    "tamil": ((0x0B80, 0x0BFF),),
    "telugu": ((0x0C00, 0x0C7F),),
    "kannada": ((0x0C80, 0x0CFF),),
    "malayalam": ((0x0D00, 0x0D7F),),
    "sinhala": ((0x0D80, 0x0DFF),),
    "thai": ((0x0E00, 0x0E7F),),
    "lao": ((0x0E80, 0x0EFF),),
    "tibetan": ((0x0F00, 0x0FFF),),
    "myanmar": ((0x1000, 0x109F), (0xA9E0, 0xA9FF), (0xAA60, 0xAA7F)),
    "georgian": ((0x10A0, 0x10FF), (0x1C90, 0x1CBF), (0x2D00, 0x2D2F)),
    "ethiopic": ((0x1200, 0x139F), (0x2D80, 0x2DDF), (0xAB00, 0xAB2F)),
    "khmer": ((0x1780, 0x17FF), (0x19E0, 0x19FF)),
    # Cherokee, Canadian syllabics, Ogham, Runic, Philippine scripts, Mongolian, Limbu, Tai Le, Buginese,
    # Tai Tham, Balinese, Sundanese, Batak, Lepcha, Ol Chiki, Tifinagh, Yi, Lisu, Vai, Bamum, Syloti Nagri,
    # Phags-pa, Saurashtra, Kayah Li, Rejang, Javanese, Cham, Tai Viet, Meetei Mayek
    "other": ((0x13A0, 0x177F), (0x1800, 0x19DF), (0x1A00, 0x1C7F), (0x1CC0, 0x1CCF), (0x2D30, 0x2D7F),
              (0xA000, 0xA63F), (0xA6A0, 0xA6FF), (0xA800, 0xA8DF), (0xA900, 0xA95F), (0xA980, 0xA9DF),
              (0xAA00, 0xAA5F), (0xAA80, 0xAAFF), (0xAB70, 0xABFF)),
}
_BITS = {name: 1 << index for index, name in enumerate(SCRIPTS)}
_TABLE = sorted((lo, hi, _BITS[name]) for name, ranges in SCRIPTS.items() for lo, hi in ranges)
_STARTS = [lo for lo, _hi, _bit in _TABLE]
_SURROGATES = (0xD800, 0xDFFF)  # never valid in UTF-8: count as "cannot complete" for prefixes

#: request kinds whose answer is target-language text (``key_contexts.CONTEXT_LABELS`` names)
BAN_TASKS = frozenset({
    "translation", "refinement", "book_title", "metadata", "batch_toc_translation", "batch_header_translation",
    "image_translation", "summary",
})

_CJK_WORDS = ("chinese", "mandarin", "cantonese", "japanese", "korean", "hanzi", "putonghua", "zhongwen")
_CJK_CODES = re.compile(r"^(zh|ja|ko|jp|kr|cn|yue|cmn|jpn|kor|zho|chi)([-_].*)?$")

#: target language names (lower-case words) and codes -> the scripts of ``SCRIPTS`` they write in; ""
#: is Latin (or Greek): nothing of ``SCRIPTS`` is theirs. Covers every ``language_options.TARGET_LANGUAGES``.
_LANGUAGES = {
    "": ("english spanish castilian french german italian portuguese dutch flemish polish czech slovak slovenian "
         "slovene croatian bosnian romanian moldovan hungarian swedish danish norwegian bokmal nynorsk finnish "
         "estonian latvian lithuanian icelandic faroese irish gaelic welsh breton catalan valencian galician "
         "basque occitan turkish azerbaijani azeri turkmen vietnamese indonesian indonesia malay melayu filipino "
         "tagalog cebuano ilocano javanese sundanese swahili kiswahili afrikaans hausa yoruba igbo zulu xhosa "
         "somali shona kinyarwanda malagasy maltese albanian esperanto latin maori hawaiian samoan tongan "
         "fijian haitian luxembourgish frisian quechua guarani aymara greek",
         "en es fr de it pt nl pl cs sk sl hr bs ro hu sv da no nb nn fi et lv lt is fo ga gd cy br ca gl eu oc "
         "tr az tk vi id in ms fil tl ceb jv su sw af ha yo ig zu xh so sn rw mg mt sq eo la mi haw sm to fj ht "
         "lb fy qu gn el"),
    "cyrillic": ("russian ukrainian belarusian byelorussian bulgarian macedonian serbian montenegrin kazakh kyrgyz "
                 "kirghiz tajik mongolian uzbek tatar bashkir chechen chuvash yakut buryat",
                 "ru uk be bg mk sr cnr kk ky tg mn uz tt ba ce cv"),
    "armenian": ("armenian", "hy"),
    "georgian": ("georgian", "ka"),
    "hebrew": ("hebrew yiddish", "he iw yi"),
    "arabic": ("arabic persian farsi dari pashto pushto urdu sindhi uyghur uighur kurdish sorani kashmiri balochi "
               "dhivehi maldivian punjabi panjabi shahmukhi saraiki hindustani",
               "ar fa prs ps ur sd ug ku ckb ks dv pa"),
    "devanagari": ("hindi marathi nepali sanskrit maithili bhojpuri konkani dogri bodo awadhi magahi rajasthani "
                   "hindustani", "hi mr ne sa mai bho kok doi"),
    "bengali": ("bengali bangla assamese manipuri meitei sylheti", "bn as mni"),
    "gurmukhi": ("punjabi panjabi gurmukhi", "pa"),
    "gujarati": ("gujarati", "gu"),
    "oriya": ("odia oriya", "or"),
    "tamil": ("tamil", "ta"),
    "telugu": ("telugu", "te"),
    "kannada": ("kannada", "kn"),
    "malayalam": ("malayalam", "ml"),
    "sinhala": ("sinhala sinhalese", "si"),
    "thai": ("thai", "th"),
    "lao": ("lao laotian", "lo"),
    "tibetan": ("tibetan dzongkha", "bo dz"),
    "myanmar": ("burmese myanmar", "my"),
    "khmer": ("khmer cambodian", "km"),
    "ethiopic": ("amharic tigrinya tigre geez", "am ti"),
    CJK: (" ".join(_CJK_WORDS), "zh ja ko jp kr cn yue cmn jpn kor zho chi"),
}


def _invert(column: int) -> Dict[str, FrozenSet[str]]:
    table: Dict[str, set] = {}
    for script, columns in _LANGUAGES.items():
        for word in columns[column].split():
            table.setdefault(word, set())
            if script:
                table[word].add(script)
    return {word: frozenset(scripts) for word, scripts in table.items()}


_NAME_SCRIPTS = _invert(0)
_CODE_SCRIPTS = _invert(1)
_CODE = re.compile(r"^([a-z]{2,3})(?:[-_][a-z0-9]+)*$")


def _lookup(cp: int) -> int:
    index = bisect.bisect_right(_STARTS, cp) - 1
    if index >= 0 and cp <= _TABLE[index][1]:
        return _TABLE[index][2]
    return 0


def script_mask(scripts: Iterable[str]) -> int:
    mask = 0
    for name in scripts or ():
        mask |= _BITS.get(str(name), 0)
    return mask


def script_of(cp: int) -> Optional[str]:
    """The ``SCRIPTS`` name of a code point, None for one never banned (Latin, Greek, punctuation, emoji)."""
    bit = _lookup(cp)
    return next((name for name, value in _BITS.items() if value == bit), None) if bit else None


def is_banned_char(cp: int, scripts: Iterable[str] = (CJK,)) -> bool:
    return bool(_lookup(cp) & script_mask(scripts))


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


def target_scripts(name: Any) -> Optional[FrozenSet[str]]:
    """The ``SCRIPTS`` the target language writes in (empty for a Latin or Greek one), None when this
    module does not recognise it."""
    text = str(name or "").strip().casefold()
    if not text:
        return None
    found: set = set()
    known = False
    code = _CODE.match(text)
    if code and code.group(1) in _CODE_SCRIPTS:
        known = True
        found |= _CODE_SCRIPTS[code.group(1)]
    for word in re.findall(r"[a-z]+", text):
        if word in _NAME_SCRIPTS:
            known = True
            found |= _NAME_SCRIPTS[word]
    for ch in text:  # written in its own script: 中文, हिन्दी, Русский
        own = script_of(ord(ch))
        if own:
            known = True
            found.add(own)
    if is_cjk_target(text):
        known = True
        found.add(CJK)
    return frozenset(found) if known else None


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on", "single_pass")
    return bool(value)


def banned_scripts(task: Any, target: Any, *, single_pass: Any = False, enabled: bool = True) -> Tuple[str, ...]:
    """The ``SCRIPTS`` to ban for this request, () for none (see the module doc)."""
    if not enabled:
        return ()
    target = str(target or "").strip()
    if not target:
        return ()
    kind = str(task or "").strip().lower()
    if "glossary" in kind or kind not in BAN_TASKS:
        return ()
    if kind == "translation" and _truthy(single_pass):
        return ()
    own = target_scripts(target)
    if own is None:
        return (CJK,)  # an unknown target: the original CJK-only ban
    return tuple(name for name in SCRIPTS if name not in own)


def wants_ban(task: Any, target: Any, *, single_pass: Any = False, enabled: bool = True) -> bool:
    """Ban some script for this request (see the module doc)."""
    return bool(banned_scripts(task, target, single_pass=single_pass, enabled=enabled))


# ---------------------------------------------------------------------------------------- tokens

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


def _interval_bits(first: int, last: int) -> int:
    """The script bits of every code point in [first, last] (surrogates skipped); 0 when some code point
    there belongs to no script (it can always become an allowed character)."""
    bits, cp = 0, first
    while cp <= last:
        if _SURROGATES[0] <= cp <= _SURROGATES[1]:
            cp = _SURROGATES[1] + 1
            continue
        index = bisect.bisect_right(_STARTS, cp) - 1
        if index < 0 or cp > _TABLE[index][1]:
            return 0
        bits |= _TABLE[index][2]
        cp = _TABLE[index][1] + 1
    return bits


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


def token_masks(piece: bytes) -> Tuple[int, int]:
    """(the script bits of the token's complete characters, the script bits every completion of its
    incomplete UTF-8 tail can have, 0 when it has none or one completion is never banned)."""
    data = bytes(piece or b"")
    if data.isascii():
        return 0, 0
    chars, index, size = 0, 0, len(data)
    while index < size:
        lead = data[index]
        length = _sequence_length(lead)
        if length <= 1:  # ASCII, or a leading/stray continuation byte (a character we cannot see)
            index += 1
            continue
        tail = data[index + 1:index + length]
        if any(not 0x80 <= byte <= 0xBF for byte in tail):
            index += 1
            continue
        if len(tail) < length - 1:  # the token ends inside this character
            interval = _prefix_interval(data[index:])
            return chars, (_interval_bits(*interval) if interval else 0)
        try:
            cp = ord(data[index:index + length].decode("utf-8"))
        except UnicodeDecodeError:
            index += 1
            continue
        chars |= _lookup(cp)
        index += length
    return chars, 0


def _banned(chars: int, tail: int, mask: int) -> bool:
    return bool(chars & mask) or (bool(tail) and not tail & ~mask)


def token_is_banned(piece: bytes, scripts: Iterable[str] = (CJK,)) -> bool:
    """See the module doc: a character of a banned script, or an incomplete tail that can only become one."""
    return _banned(*token_masks(piece), script_mask(scripts))


def banned_token_ids(pieces: Iterable[bytes], scripts: Iterable[str] = (CJK,)) -> list:
    """The ids (positions in ``pieces``, each token's raw bytes) to ban."""
    mask = script_mask(scripts)
    return [token for token, piece in enumerate(pieces) if piece and _banned(*token_masks(piece), mask)]


def bias_token_ids(pieces: Iterable[Tuple[int, bytes]], scripts: Iterable[str], limit: int) -> List[int]:
    """The first ``limit`` banned ids of ``(id, raw bytes)`` pairs given lowest id first: in a BPE
    vocabulary a lower id is an earlier merge, so these are the most frequent banned tokens (the single
    lead bytes such as 0xE4 come first and stop the byte-spelled rare characters)."""
    mask = script_mask(scripts)
    found: List[int] = []
    if not mask or limit <= 0:
        return found
    for token, piece in pieces:
        if piece and _banned(*token_masks(piece), mask):
            found.append(int(token))
            if len(found) >= limit:
                break
    return found


# ---------------------------------------------------------------------- OpenAI logit bias (API)

#: entries OpenAI accepts in ``logit_bias`` (documented as 300 for the completions API; kept for chat)
OPENAI_LOGIT_BIAS_LIMIT = 300
#: the highest token id every OpenAI chat model accepted as a ``logit_bias`` key (cl100k_base's last id:
#: gpt-4o requests were reported to reject larger o200k ids), so a key is never refused
_OPENAI_MAX_BIAS_ID = 100256


def openai_encoding_name(model: Any) -> Optional[str]:
    """The tiktoken encoding of an official OpenAI chat model that takes ``logit_bias``; None for the
    reasoning models (o1 / o3 / o4, GPT-5 and later reject it) and for every other model (a guessed
    tokenizer would bias the wrong tokens)."""
    leaf = str(model or "").strip().lower().rsplit("/", 1)[-1]
    if leaf.startswith(("gpt-4o", "chatgpt-4o", "gpt-4.1", "gpt-4.5")):
        return "o200k_base"
    if leaf.startswith(("gpt-4", "gpt-3.5")):
        return "cl100k_base"
    return None


@functools.lru_cache(maxsize=32)
def _openai_bias_ids(encoding_name: str, scripts: Tuple[str, ...], limit: int) -> Tuple[int, ...]:
    import tiktoken  # the token counter's tokenizer; its files are cached on first use

    encoding = tiktoken.get_encoding(encoding_name)
    last = min(int(getattr(encoding, "n_vocab", 0) or 0), _OPENAI_MAX_BIAS_ID + 1)

    def pieces():
        for token in range(last):
            try:
                yield token, encoding.decode_single_token_bytes(token)
            except Exception:
                continue

    return tuple(bias_token_ids(pieces(), scripts, limit))


def openai_logit_bias(model: Any, scripts: Iterable[str], limit: int = OPENAI_LOGIT_BIAS_LIMIT) -> Dict[str, int]:
    """``{"<token id>": -100}`` for the most frequent tokens of ``scripts`` in ``model``'s tokenizer, {}
    when the model takes no logit bias, nothing is banned or tiktoken cannot load the encoding. A
    whole script does not fit in the provider's cap; the output guard (``foreign_script_leak``) covers
    the rest."""
    name = openai_encoding_name(model)
    wanted = tuple(sorted(set(str(s) for s in scripts or ()) & set(SCRIPTS)))
    if not name or not wanted:
        return {}
    try:
        ids = _openai_bias_ids(name, wanted, int(limit))
    except Exception:
        return {}
    return {str(token): -100 for token in ids}


# ------------------------------------------------------------------------------ output guard (API)

#: a response counts as leaking once its foreign-script characters reach both of these
LEAK_MIN_CHARS = 3
LEAK_MIN_SHARE = 0.001  # of the response's non-space characters

_SCRIPT_NAMES = {CJK: "Chinese, Japanese or Korean characters", "other": "other scripts"}


def _source_text(source: Any) -> str:
    """The text of a request: a string, or chat messages (string contents and text parts)."""
    if isinstance(source, str):
        return source
    parts: List[str] = []
    for message in source or ():
        content = message.get("content") if isinstance(message, dict) else message
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    parts.append(part["text"])
    return "\n".join(parts)


def foreign_script_leak(text: Any, scripts: Iterable[str], source: Any = "") -> Optional[Dict[str, Any]]:
    """The banned-script characters of a response, None while below the threshold (``LEAK_MIN_CHARS`` and
    ``LEAK_MIN_SHARE``). A run of banned-script characters that also occurs in ``source`` (the request's
    text or messages: a kept original name, a quoted line) is not counted.

    Returns ``{"chars": n, "scripts": {name: n}, "sample": "<first runs>"}``."""
    body = str(text or "")
    mask = script_mask(scripts)
    if not body or not mask:
        return None
    request = _source_text(source)
    counts: Dict[str, int] = {}
    samples: List[str] = []
    total = 0
    run: List[str] = []

    def close() -> None:
        nonlocal total
        if not run:
            return
        piece = "".join(run)
        run.clear()
        if piece in request:
            return
        total += len(piece)
        for ch in piece:
            name = script_of(ord(ch)) or "other"
            counts[name] = counts.get(name, 0) + 1
        if len(samples) < 5:
            samples.append(piece[:20])

    for ch in body:
        if _lookup(ord(ch)) & mask:
            run.append(ch)
        else:
            close()
    close()
    if not total:
        return None
    visible = sum(1 for ch in body if not ch.isspace())
    if total < LEAK_MIN_CHARS or total < LEAK_MIN_SHARE * visible:
        return None
    return {"chars": total, "scripts": counts, "sample": " ".join(samples)}


def retry_instruction(target: Any, scripts: Iterable[str] = ()) -> str:
    """The line added to the request when a response leaked another script."""
    names = []
    for name in scripts or ():
        label = _SCRIPT_NAMES.get(name, str(name).title() + " script")
        if label not in names:
            names.append(label)
    shown = ", ".join(names[:4]) + (", ..." if len(names) > 4 else "")
    language = str(target or "").strip() or "the target language"
    return (f"[Write the whole answer in {language} only. The previous answer contained text in another "
            f"writing system ({shown}); do not use it, except for names kept exactly as in the source.]")
