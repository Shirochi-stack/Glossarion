# -*- coding: utf-8 -*-
"""Is one part of a source-language name TRANSLITERATED in its translation?

Used by glossary matching to tell a person's name from a descriptive title
when only one word of a multi-word, gender-enabled entry appears in a chapter:

    카인 에렌하이츠 = Kain Erenheitz      카인   -> "Kain"   transliterated: a name
    3서클 마법사   = Third-Circle Mage    마법사 -> "Mage"   translated: a title

Names are carried across languages by sound, ordinary words by meaning, so
the glossary's own translated_name column answers the question without a
dictionary. Measured on six novels this kept the given names and surnames
(카인, 에비스, 리즈, 아르젠트, スズキ-style 스즈키 …) and rejected the generic
head nouns of epithets (마법사, 기사, 탐험가, 엄마, 제자 …) that every other
rule tried let through.

Pure functions, no I/O, no third-party imports. Hangul and kana can be
romanized algorithmically; Han characters cannot (the reading is not in the
character), so for those -- and for a non-Latin translation -- the answer is
``None`` ("cannot tell") and the caller falls back to a structural rule.
"""

import difflib
import re
import unicodedata
from functools import lru_cache

# Revised Romanization letter tables, indexed by the jamo arithmetic of a
# precomposed syllable. No sound-change rules: loanword spellings are being
# compared, not pronunciations.
_INITIALS = ("g", "kk", "n", "d", "tt", "r", "m", "b", "pp", "s", "ss", "", "j", "jj", "ch", "k", "t", "p", "h")
_MEDIALS = ("a", "ae", "ya", "yae", "eo", "e", "yeo", "ye", "o", "wa", "wae", "oe", "yo", "u", "wo", "we",
            "wi", "yu", "eu", "ui", "i")
_FINALS = ("", "k", "k", "k", "n", "n", "n", "t", "l", "k", "m", "l", "l", "l", "p", "l", "m", "p", "p",
           "t", "t", "ng", "t", "t", "k", "t", "p", "t")

_KANA = {
    "きゃ": "kya", "きゅ": "kyu", "きょ": "kyo", "しゃ": "sha", "しゅ": "shu", "しょ": "sho", "ちゃ": "cha",
    "ちゅ": "chu", "ちょ": "cho", "にゃ": "nya", "にゅ": "nyu", "にょ": "nyo", "ひゃ": "hya", "ひゅ": "hyu",
    "ひょ": "hyo", "みゃ": "mya", "みゅ": "myu", "みょ": "myo", "りゃ": "rya", "りゅ": "ryu", "りょ": "ryo",
    "ぎゃ": "gya", "ぎゅ": "gyu", "ぎょ": "gyo", "じゃ": "ja", "じゅ": "ju", "じょ": "jo", "びゃ": "bya",
    "びゅ": "byu", "びょ": "byo", "ぴゃ": "pya", "ぴゅ": "pyu", "ぴょ": "pyo", "ふぁ": "fa", "ふぃ": "fi",
    "ふぇ": "fe", "ふぉ": "fo", "てぃ": "ti", "でぃ": "di", "うぃ": "wi", "うぇ": "we", "うぉ": "wo",
    "ゔぁ": "va", "ゔぃ": "vi", "ゔぇ": "ve", "ゔぉ": "vo", "しぇ": "she", "じぇ": "je", "ちぇ": "che",
    "あ": "a", "い": "i", "う": "u", "え": "e", "お": "o", "か": "ka", "き": "ki", "く": "ku", "け": "ke",
    "こ": "ko", "さ": "sa", "し": "shi", "す": "su", "せ": "se", "そ": "so", "た": "ta", "ち": "chi",
    "つ": "tsu", "て": "te", "と": "to", "な": "na", "に": "ni", "ぬ": "nu", "ね": "ne", "の": "no",
    "は": "ha", "ひ": "hi", "ふ": "fu", "へ": "he", "ほ": "ho", "ま": "ma", "み": "mi", "む": "mu",
    "め": "me", "も": "mo", "や": "ya", "ゆ": "yu", "よ": "yo", "ら": "ra", "り": "ri", "る": "ru",
    "れ": "re", "ろ": "ro", "わ": "wa", "を": "o", "ん": "n", "が": "ga", "ぎ": "gi", "ぐ": "gu",
    "げ": "ge", "ご": "go", "ざ": "za", "じ": "ji", "ず": "zu", "ぜ": "ze", "ぞ": "zo", "だ": "da",
    "ぢ": "ji", "づ": "zu", "で": "de", "ど": "do", "ば": "ba", "び": "bi", "ぶ": "bu", "べ": "be",
    "ぼ": "bo", "ぱ": "pa", "ぴ": "pi", "ぷ": "pu", "ぺ": "pe", "ぽ": "po", "ゔ": "vu",
    "ぁ": "a", "ぃ": "i", "ぅ": "u", "ぇ": "e", "ぉ": "o",
}


def _to_hiragana(text):
    return "".join(chr(ord(ch) - 0x60) if 0x30A1 <= ord(ch) <= 0x30F6 else ch for ch in text)


def romanize(part):
    """Hangul or kana (or Latin) as lowercase Latin letters; ``""`` when any
    character has no algorithmic reading (Han, digits, symbols)."""
    part = unicodedata.normalize("NFKC", str(part or ""))
    hira = _to_hiragana(part)
    out = []
    i = 0
    while i < len(part):
        ch = part[i]
        code = ord(ch)
        if 0xAC00 <= code <= 0xD7A3:
            offset = code - 0xAC00
            out.append(_INITIALS[offset // 588] + _MEDIALS[(offset % 588) // 28] + _FINALS[offset % 28])
            i += 1
        elif hira[i:i + 2] in _KANA:
            out.append(_KANA[hira[i:i + 2]])
            i += 2
        elif hira[i] in _KANA:
            out.append(_KANA[hira[i]])
            i += 1
        elif hira[i] == "っ" or ch == "ー":
            i += 1  # gemination and long-vowel marks carry no letter of their own
        elif ch.isalpha() and unicodedata.normalize("NFKD", ch)[0].isascii():
            out.append(unicodedata.normalize("NFKD", ch)[0].lower())
            i += 1
        else:
            return ""
    return "".join(out)


# Consonants that loanword spelling treats as interchangeable: 리즈 is "rijeu"
# and Liz is "liz" -- r/l and j/z -- and 에비스 / Evis differ only by b/v.
_CONSONANT_CLASS = {
    "b": "B", "p": "B", "f": "B", "v": "B",
    "d": "D", "t": "D",
    "g": "K", "k": "K", "q": "K", "x": "K",
    "j": "S", "z": "S", "s": "S",
    "l": "L", "r": "L",
    "m": "M", "n": "N",
}
_VOWELISH = frozenset("aeiouyw")


def consonant_skeleton(latin, hard_ch=False, hard_g=True):
    """Consonant classes in order, vowels and doubled consonants dropped.

    English spells two sounds each with ``ch`` (Charles / Chloe) and ``g``
    (Gerik / Eugene); the flags pick the reading. A romanized part needs no
    guessing -- its ch is soft and its g is hard, the defaults.
    """
    latin = latin.lower().replace("ch", "k" if hard_ch else "s")
    for digraph, single in (("sh", "s"), ("th", "t"), ("ph", "f"), ("ng", "n")):
        latin = latin.replace(digraph, single)
    out = []
    for i, ch in enumerate(latin):
        before_front_vowel = latin[i + 1:i + 2] in ("e", "i", "y")
        if ch == "c":
            cls = "S" if before_front_vowel else "K"
        elif ch == "g":
            cls = "K" if hard_g or not before_front_vowel else "S"
        else:
            cls = _CONSONANT_CLASS.get(ch)
        if cls and (not out or out[-1] != cls):
            out.append(cls)
    return "".join(out)


_NON_PREVOCALIC_R_RE = re.compile(r"r(?![aeiouy])")


def _word_readings(word):
    """The word, and the word as a non-rhotic speaker says it: Korean and
    Japanese borrow George as 조지 / ジョージ and Edward as 에드워드, no r."""
    return {word, _NON_PREVOCALIC_R_RE.sub("", word)}


def _word_skeletons(word):
    """Every consonant reading of an English-spelled word."""
    return {
        consonant_skeleton(reading, hard_ch, hard_g)
        for reading in _word_readings(word)
        for hard_ch in (False, True) for hard_g in (False, True)
    }


def _spelling_form(roman):
    """Romanization minus what Korean adds to fit its syllables: the filler
    vowel 으 (알렉스 al-rek-seu) and the ㄹㄹ that RR writes as ``lr``."""
    return roman.replace("eu", "").replace("lr", "ll") or roman


_LATIN_WORD_RE = re.compile(r"[A-Za-zÀ-ɏ]+")


def _same_onset(roman, word):
    if roman[0] == word[0]:
        return True
    if roman[0] in _VOWELISH and word[0] in _VOWELISH:
        return True
    if {roman[0], word[0]} == {"j", "y"}:
        return True  # 요한 / Johan, 율리아 / Julia
    first = consonant_skeleton(roman[:2])[:1]
    return bool(first) and any(sk[:1] == first for sk in _word_skeletons(word[:2]))


def transliteration_score(roman, word):
    """0..1: how much `word` reads like the romanized part `roman`.

    Same onset required, then a blend of consonant-skeleton similarity
    (weighted higher -- vowels are where loanword spellings wander) and plain
    spelling similarity.
    """
    if not roman or not word or not _same_onset(roman, word):
        return 0.0
    plain = _spelling_form(roman)
    spelling = max(difflib.SequenceMatcher(None, plain, reading).ratio() for reading in _word_readings(word))
    skeleton = consonant_skeleton(roman)
    word_skeletons = {sk for sk in _word_skeletons(word) if sk}
    if not skeleton or not word_skeletons:
        score = spelling  # all vowels (유이 / Yui): spelling is all there is
    else:
        sound = max(difflib.SequenceMatcher(None, skeleton, sk).ratio() for sk in word_skeletons)
        score = 0.6 * sound + 0.4 * spelling
    # A transliteration is about as long as what it transliterates; 마법사
    # (mabeopsa) sharing m-g with "Mage" is a coincidence of two short words.
    shorter, longer = sorted((len(plain), len(word)))
    if shorter < 0.6 * longer:
        score *= shorter / (0.6 * longer)
    return score


@lru_cache(maxsize=65536)
def is_transliterated(part, translated_name, siblings=(), threshold=0.66):
    """True / False, or None when it cannot be judged.

    ``part`` is one part of the raw name, ``translated_name`` the whole
    translated name, ``siblings`` the raw name's other parts (a tuple). True
    when some word of the translation reads like the part AND no sibling
    reads more like that word: in 시에네 선배 = Senior Siene, "Siene" belongs
    to 시에네, so it cannot also vouch for 선배.
    """
    roman = romanize(part)
    if not roman:
        return None
    words = []
    for word in _LATIN_WORD_RE.findall(str(translated_name or "")):
        word = unicodedata.normalize("NFKD", word).encode("ascii", "ignore").decode().lower()
        if len(word) >= 2:
            words.append(word)
    if not words:
        return None
    rivals = [romanize(sibling) for sibling in siblings if sibling != part]
    for word in words:
        score = transliteration_score(roman, word)
        if score < threshold:
            continue
        if any(transliteration_score(rival, word) > score for rival in rivals if rival):
            continue
        return True
    return False
