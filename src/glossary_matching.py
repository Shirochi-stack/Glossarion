# -*- coding: utf-8 -*-
"""Shared term-matching primitives for glossary compression and usage.

This module is deliberately a *leaf*: apart from ``glossary_translit`` (itself
pure functions with no imports of its own) it imports nothing from the rest of
the project.  Both ``glossary_compressor`` and ``glossary_usage`` need the same
matching logic, but ``glossary_compressor`` pulls in ``gender_tracking``,
``extract_glossary_from_epub`` and ``GlossaryManager``.  That import weight is
why ``glossary_usage`` used to carry a hand-copied duplicate of the matcher as
an ImportError fallback — a copy that could silently drift from the original.
Keeping the primitives here means both sides import the same code.

For the same reason this module never reads settings itself.  Callers build a
``MatchConfig`` from whatever getter they use (``_setting`` in the compressor,
``os.getenv`` in usage) and pass it in.
"""

import json
import os
import re
import threading
import unicodedata
from collections import Counter, OrderedDict
from functools import lru_cache

from glossary_translit import is_transliterated


# ─── Text primitives ─────────────────────────────────────────────────────────
# Moved here verbatim from glossary_usage so the translated-output matchers and
# the source-text matchers cannot disagree about what a token or a fold is.

_WHITESPACE_RE = re.compile(r"\s+")
# "Despacing" also drops the middle dot that CJK text puts between the parts
# of a transliterated name: 哈利·波特 and 哈利波特, アリス・リデル and アリスリデル
# are the same name. Applied to the term and to the chapter alike.
_DESPACE_RE = re.compile(r"[\s\u30fb\u00b7\u2022\u2027]+")
# Maximal runs of ASCII "word" characters, matching the boundary class used by
# the matching regexes: (?<![A-Za-z0-9_]) ... (?![A-Za-z0-9_])
_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
# Translation table for folding ASCII text: non-alphanumeric -> space. Produces
# byte-for-byte the same result as the char-by-char generator for ASCII input,
# but runs ~3x faster via str.translate (used on whole output files).
_ASCII_FOLD_TABLE = {i: (chr(i) if chr(i).isalnum() else " ") for i in range(128)}


def norm_text(value):
    return str(value or "").strip()


def fold_match_text(value):
    value = unicodedata.normalize("NFKC", str(value or "")).casefold()
    if value.isascii():
        # Fast path for ASCII (the common case for translated English output):
        # str.translate is far faster than a Python-level char generator and
        # yields an identical result.
        value = value.translate(_ASCII_FOLD_TABLE)
    else:
        value = "".join(ch if ch.isalnum() else " " for ch in value)
    return _WHITESPACE_RE.sub(" ", value).strip()


def boundary_match(term, text):
    return re.search(r"(?<![A-Za-z0-9_])" + re.escape(term) + r"(?![A-Za-z0-9_])", text) is not None


# ─── Entry-type classification ───────────────────────────────────────────────

# Entry types that carry a gender and are therefore matched as personal names.
# glossary_usage hard-coded this set; the compressor derives one dynamically
# from the custom entry types and degrades to {"character"} when that import
# fails.  is_gender_entry_type UNIONS the two rather than substituting, because
# substituting the degraded set would quietly narrow usage-side matching.
DEFAULT_GENDER_ENTRY_TYPES = frozenset({
    "character", "characters", "title", "titles", "nickname", "nicknames",
})


def is_gender_entry_type(entry_type, gender="", gender_types=None):
    """Return True when an entry should be matched as a personal name.

    An explicit gender value counts on its own: a row that carries a gender is
    a name regardless of how its type column is spelled.
    """
    if norm_text(gender):
        return True
    entry_type = norm_text(entry_type).lower()
    if not entry_type:
        return False
    if entry_type in DEFAULT_GENDER_ENTRY_TYPES:
        return True
    if gender_types:
        return entry_type in {str(t or "").lower() for t in gender_types}
    return False


# ─── Legacy matcher ──────────────────────────────────────────────────────────

def legacy_text_contains_term(text, term, is_character=False):
    """The pre-tiered matcher, preserved exactly.

    Substring matching in every script, plus a whitespace-token fallback whose
    minimum token length drops to 1 for character entries.  That 1-char floor
    is the main over-matching source, but it is load-bearing for recall: it is
    what rescues a glossary term spelled with a space ("미샤 랄토스") when the
    source text runs it together ("미샤랄토스").
    """
    if not term or not text:
        return False

    # Full term match first (fast path)
    if term in text:
        return True

    # Multi-word: check individual tokens
    # Character entries: accept any token length (>=1 char)
    # Non-character entries: require >=2 chars to limit false positives
    min_token_len = 1 if is_character else 2
    if ' ' in term:
        for token in term.split():
            if len(token) >= min_token_len and token in text:
                return True

    return False


# ─── Script classification ───────────────────────────────────────────────────
# Plain codepoint ranges rather than regex \p{Script=...}: this runs per
# character inside the boundary heuristic, and property lookups are measurably
# slower there. It also keeps this module free of third-party imports.

LATIN = "latin"
HANGUL = "hangul"
HAN = "han"
KANA = "kana"
DIGIT = "digit"
OTHER = "other"

_HANGUL_RANGES = ((0xAC00, 0xD7A3), (0x1100, 0x11FF), (0x3130, 0x318F), (0xA960, 0xA97F))
_HAN_RANGES = ((0x4E00, 0x9FFF), (0x3400, 0x4DBF), (0xF900, 0xFAFF), (0x20000, 0x2A6DF))
_KANA_RANGES = ((0x3040, 0x309F), (0x30A0, 0x30FF), (0x31F0, 0x31FF), (0xFF66, 0xFF9D))

_CJK_SCRIPTS = frozenset({HANGUL, HAN, KANA})


@lru_cache(maxsize=4096)
def script_of(ch):
    """Classify a single character into the coarse script buckets we care about."""
    if not ch:
        return OTHER
    cp = ord(ch)
    if cp < 128:
        if ch.isdigit():
            return DIGIT
        return LATIN if ch.isalpha() else OTHER
    for lo, hi in _HANGUL_RANGES:
        if lo <= cp <= hi:
            return HANGUL
    for lo, hi in _HAN_RANGES:
        if lo <= cp <= hi:
            return HAN
    for lo, hi in _KANA_RANGES:
        if lo <= cp <= hi:
            return KANA
    if ch.isdigit():
        return DIGIT
    return LATIN if ch.isalpha() else OTHER


def is_cjk_char(ch):
    return script_of(ch) in _CJK_SCRIPTS


@lru_cache(maxsize=8192)
def term_script(term):
    """The dominant script of a term, used to pick boundary rules."""
    counts = {}
    for ch in term:
        sc = script_of(ch)
        if sc in (OTHER, DIGIT):
            continue
        counts[sc] = counts.get(sc, 0) + 1
    if not counts:
        return OTHER
    # Kana beats han for mixed Japanese text: 太郎さん is Japanese, not Chinese.
    if KANA in counts and HAN in counts:
        return KANA
    return max(counts.items(), key=lambda kv: kv[1])[0]


# ─── Korean particles (josa) ─────────────────────────────────────────────────
# A Korean name is normally followed immediately by a particle, with no space:
# 루나는, 루나가, 루나에게. So "the next character is Hangul" does NOT mean the
# match ran into another word — it usually means the name is being used. This
# table is what makes the sub-word boundary check safe for Korean; without it
# the heuristic would reject nearly every true positive.
KOREAN_PARTICLE_PREFIXES = (
    "으로부터", "에게서", "한테서", "이라고", "께서", "부터", "까지", "에게",
    "한테", "에서", "보다", "처럼", "마저", "조차", "밖에", "으로", "이랑",
    "이나", "라고", "라며", "이며", "든지", "이든", "커녕",
    "은", "는", "이", "가", "을", "를", "와", "과", "의", "에", "도", "만",
    "로", "랑", "나", "야", "아", "님", "씨", "들", "군", "양", "여", "고",
)

# Bound nouns and delimiters that attach straight to a noun. Measured on a real
# book these were the misses: 로켓같은, 펄스때문에, 수장끼리, 기업측, 록온따윈,
# 소녀답게.
#
# This table, the copula and everything below it are only consulted when the
# WHOLE term (two syllables or more) is being matched. A part of a multi-word
# entry, or a one-syllable term, keeps the conservative particle table above:
# measured on the same book, relaxing those readmitted exactly the noise the
# precise matcher exists to remove (침묵 지대 via 침묵하던, 노스 스타 사 via
# 스타일, 유 via 유일하게) and recovered nothing.
KOREAN_BOUND_NOUN_PREFIXES = (
    "하고", "마다", "만큼", "대로", "때문", "끼리", "따위", "따윈", "다운",
    "같", "답", "뿐", "쯤", "째", "측", "쪽", "네", "께",
)

# The copula 이다 fuses onto the noun with no space, and it attaches to names
# as readily as to anything else: 아논입니다 ("it's Anon"), 드론일 것이다,
# 수장인 페데리코, 소녀였다, 친구다, 악마라 했었지, 변수겠지. None of these
# start with a josa, so a particle-only table drops every one of them.
KOREAN_COPULA_PREFIXES = (
    "입니", "이에요", "예요", "이었", "였", "이다", "이라", "이란", "이고",
    "이면", "이니", "이지", "이죠", "이네", "이냐", "이잖", "이겠", "이던",
    "이군", "이구", "구나", "군요", "니까", "지만", "지요",
    "라", "란", "인", "일", "임", "죠", "냐", "잖", "겠",
)
# Measured against 3,741 two-syllable terms over a whole novel. Left out because
# what they caught was a verb stem that happens to spell a name, not a noun
# taking the copula: bare 지 (사라지는, 지나지, 나가지), 던 (달리던, 지나던),
# 든 (나가든), 니 (해주니). 다 is kept but only as a word-final ending, so
# 라베다 / 악마다 / 친구다고 count and 구름다리 does not.
_KOREAN_DA_FOLLOWERS = frozenset("고는면니만며가")

# 하다 / 되다 and friends turn a noun into a verb or adjective: 오염된 땅,
# 록온하고, 총괄하는, 공격적인. They never attach to a person's name, and one
# of them would readmit a real false positive if they did (유리한 "favourable"
# is not the character 유리), so they only count for non-character entries.
KOREAN_VERBALIZER_PREFIXES = (
    "시키", "시킨", "시켜", "시켰", "당하", "당한", "당해", "당했",
    "스러", "스럽", "스런",
    "하", "한", "할", "함", "합", "해", "했", "된", "되", "될", "됐", "돼", "됨", "됩",
    "받", "적",
)

# Sino-Korean prefixes that build a compound on the LEFT of a term: 중장갑,
# 반기업, 핵펄스, 초진동. Only honoured for non-character entries and only when
# the prefix itself starts the word, so 자유 / 이유 / 크롬 / 스카라베 stay
# rejected.
#
# The set is small on purpose. It was measured against 2,216 two-syllable
# terms over a whole novel, and most productive-looking prefixes turned out to
# admit a different word far more often than a compound of the term:
# 주 (주기적 -> 기적, 주변이 -> 변이), 무 (무의식 -> 의식), 대 (대상의 -> 상의),
# 소 (소리치 -> 리치), 본 (본인도 -> 인도), 비, 신, 구, 총, 타. These four were
# right in 204 of 210 hits.
KOREAN_NOUN_PREFIXES = frozenset("반핵초중")

# One-syllable noun suffixes on the RIGHT: 장갑판, 완충재, 링거줄, 강화형,
# 선택권. Non-character entries only, and the suffix has to end the word.
# Measured the same way as the prefixes; dropped for admitting other words:
# 화 (무력화), 기 (나가기, 생기기), 탄 (철갑탄), 체 (투사체), 물 (부산물), 망, 진,
# 포, and never included: 장 / 북 / 전 / 회 (수라장, 노트북, 지구전).
KOREAN_NOUN_SUFFIXES = frozenset("판재줄류형용권성식")


# ─── Japanese particles ──────────────────────────────────────────────────────
# Hiragana and katakana share one "kana" script bucket, and a name is followed
# by a hiragana particle with no space: アリスは, リンが. Treating that as "glued
# to more kana" rejected every short katakana name. A switch between katakana
# and hiragana is itself a word edge; this table covers the hiragana-to-hiragana
# case (さくらは) that a script switch cannot.
JAPANESE_PARTICLE_PREFIXES = (
    "から", "まで", "より", "です", "でし", "だっ", "だけ", "など", "って",
    "たち", "さえ", "しか", "こそ", "にも", "には", "とは", "では",
    "は", "が", "を", "に", "へ", "と", "で", "も", "の", "や", "か", "ね", "よ", "な", "だ", "ら",
)
_JAPANESE_LEFT_PARTICLES = frozenset("はがをにへとでものやか")


# ─── Chinese function words ──────────────────────────────────────────────────
# Chinese is the hardest case for a boundary rule: there are no spaces at all,
# so the character after a name is almost always another Han character. What
# distinguishes 宋家 in 宋家的人 (a real reference) from 天下 in 天下第一楼 (a
# fragment of a proper noun) is that a grammatical particle cannot be part of a
# compound noun. Only function words are listed — deliberately no verbs, since
# a verb can be half of a compound and would readmit the false positives.
CHINESE_FUNCTION_WORDS = (
    # structural and aspect particles
    "的", "地", "得", "了", "着", "过", "之",
    # copula, existence, negation
    "是", "有", "没", "不", "在",
    # conjunctions
    "和", "与", "及", "或", "而", "但", "则", "也", "就", "还", "又", "却", "都",
    # prepositions / coverbs
    "把", "被", "给", "对", "向", "从", "到", "为", "由", "跟", "比", "用",
    # locatives and enclitics
    "里", "中", "上", "下", "内", "外", "前", "后", "间", "们", "者",
    # sentence-final particles
    "呢", "吗", "吧", "啊", "嘛", "呀", "么",
)

# What follows a subject. Chinese has no spaces, so a name is almost always
# followed directly by another Han character, and for a name that character is
# overwhelmingly a verb of speech / motion / perception, an adverb, or a body
# part: 小明说, 小明走了, 小明忽然, 小明脸色一变. Function words alone miss all of
# these. Numerals, 第, and size adjectives are left out on purpose because they
# build proper nouns (天下第一楼) and compounds (大地震) rather than follow a
# subject.
CHINESE_SUBJECT_FOLLOWERS = frozenset(
    "说道笑看走想问答叫喊听见来去站坐点摇皱叹忽突连忙立已正这那便将会能要可"
    "很最更太竟才刚曾只倒仍依似如像等带拿抬伸转回望盯瞪低沉怒微轻冷淡心脸眼身手"
)

# Japanese plural / collective markers written in kanji after a kanji name.
# What may stand directly left of a ONE-character Han entry: a function word
# (是白, 和白, 对白兄), or a word that introduces a name (姓白, 老白, 小白, 阿白).
_HAN_SINGLE_LEFT_OPENERS = frozenset(
    "的地得了着过是有在和与及或而但则也就还又却都把被给对向从到为由跟比用"
    "呢吗吧啊嘛呀这那说道问叫喊请让姓老小阿"
)

# What follows a short kanji / hanzi NAME without making it a different word:
# plural markers, and the kinship / rank words used as forms of address
# (萧炎哥哥, 小明师兄, 太郎先輩, 王女殿下). Longest first so 哥哥 wins over 哥.
_HAN_NAME_SUFFIXES = (
    "お嬢様", "哥哥", "姐姐", "弟弟", "妹妹", "大哥", "大姐", "师兄", "师姐", "师弟", "师妹", "师叔",
    "师伯", "師兄", "師姐", "師弟", "師妹", "前辈", "前輩", "少爷", "少爺", "殿下", "陛下", "閣下",
    "阁下", "先輩", "師匠", "隊長", "団長", "将軍", "王子", "王女",
    "達", "等", "们", "們", "哥", "姐", "兄", "弟", "妹", "叔", "伯", "爷", "爺", "嬢", "卿", "姫",
)


# ─── Honorifics that attach to a name ────────────────────────────────────────
# Deliberately NOT PatternManager.CJK_HONORIFICS. That table is a dialogue /
# speech-ending list: it contains '郎' as a Japanese court title and single
# Korean syllables like 자, 모, 시, 요. Running a name through it turns 太郎
# into 太 and 미자 into 미 — manufacturing exactly the one-character false
# positives this module exists to remove. These are the suffixes that actually
# attach to a personal name and can be safely peeled off for matching.
HONORIFIC_SUFFIXES = (
    # Korean
    "선생님", "사부님", "스님", "선배", "후배", "공주", "왕자", "아씨", "도령",
    "대감", "나리", "누나", "언니", "오빠", "형", "님", "씨",
    # Japanese
    "ちゃん", "さん", "くん", "さま", "先生", "どの", "たん", "様", "殿", "君", "氏",
    # Chinese
    "先生", "小姐", "大人", "公子", "姑娘", "师父", "老师", "夫人",
    # Romanized
    "-sensei", "-chan", "-sama", "-san", "-kun", "-nim", "-ssi",
)


def _chinese_surnames():
    """Chinese surname tables, imported lazily and tolerantly.

    PatternManager is the project's canonical source for these, but this
    module must stay importable on its own, so a failure here degrades to
    empty sets (the boundary rule simply loses its surname refinement).
    """
    global _CN_SINGLE, _CN_COMPOUND
    if _CN_SINGLE is None:
        try:
            import PatternManager as _pm
            _CN_SINGLE = frozenset(getattr(_pm, "CHINESE_SINGLE_SURNAMES", ()) or ())
            _CN_COMPOUND = frozenset(getattr(_pm, "CHINESE_COMPOUND_SURNAMES", ()) or ())
        except Exception:
            _CN_SINGLE = frozenset()
            _CN_COMPOUND = frozenset()
    return _CN_SINGLE, _CN_COMPOUND


_CN_SINGLE = None
_CN_COMPOUND = None


# ─── Tiers ───────────────────────────────────────────────────────────────────
# Higher tier == stronger evidence. Acceptance is `tier >= cfg.min_tier`, so a
# single knob slides the whole matcher between "only exact" and "legacy-loose".

TIER_WEAK = 0        # a single-character token
TIER_TOKEN = 1       # a >=2 character token of a multi-part term
TIER_HONORIFIC = 2   # the term minus a name honorific
TIER_DESPACED = 3    # the term with its internal spaces removed
TIER_NORM = 4        # NFKC + casefold
TIER_EXACT = 5       # the term as written

TIER_NAMES = {
    TIER_WEAK: "weak_token",
    TIER_TOKEN: "token",
    TIER_HONORIFIC: "honorific",
    TIER_DESPACED: "despaced",
    TIER_NORM: "normalized",
    TIER_EXACT: "exact",
}


class MatchConfig:
    """Matcher settings, resolved once per compression call.

    Built from a caller-supplied getter so this module never reads os.environ
    or the compressor's ContextVar itself.
    """

    __slots__ = (
        "min_tier", "allow_weak_token", "weak_token_max_hits", "cjk_boundary",
        "short_cjk_len", "nfkc", "despaced", "despaced_latin", "honorific",
        "honorific_min_residual", "derived_forms", "cache_key",
    )

    def __init__(self, min_tier=TIER_TOKEN, allow_weak_token=True,
                 weak_token_max_hits=3, cjk_boundary=True, short_cjk_len=2,
                 nfkc=True, despaced=True, despaced_latin=False,
                 honorific=True, honorific_min_residual=2, derived_forms=True):
        self.min_tier = int(min_tier)
        self.allow_weak_token = bool(allow_weak_token)
        self.weak_token_max_hits = int(weak_token_max_hits)
        self.cjk_boundary = bool(cjk_boundary)
        self.short_cjk_len = int(short_cjk_len)
        self.nfkc = bool(nfkc)
        self.despaced = bool(despaced)
        self.despaced_latin = bool(despaced_latin)
        self.honorific = bool(honorific)
        self.honorific_min_residual = int(honorific_min_residual)
        # Verb/adjective endings (오염된, 록온하고) and one-syllable noun affixes
        # (중장갑, 장갑판) built on a term. Right for a book's own glossary; a
        # cross-novel glossary turns them into homonym hits (고려하면 is not
        # Goryeo, 결정한 is not Crystal), so the compressor switches them off
        # for the unified glossary.
        self.derived_forms = bool(derived_forms)
        # Only the variant-generation knobs belong in the cache key; the
        # acceptance knobs (min_tier, weak-token gates, boundary) are applied
        # after the variants are built, so they must not fragment the cache.
        self.cache_key = (
            self.nfkc, self.despaced, self.despaced_latin,
            self.honorific, self.honorific_min_residual,
        )

    @classmethod
    def from_getter(cls, getter):
        def flag(name, default):
            return str(getter(name, default)).strip().lower() in ("1", "true", "yes", "on")

        def num(name, default):
            try:
                return int(str(getter(name, default)).strip())
            except (TypeError, ValueError):
                return int(default)

        return cls(
            min_tier=num("GLOSSARY_MATCH_MIN_TIER", TIER_TOKEN),
            allow_weak_token=flag("GLOSSARY_MATCH_ALLOW_WEAK_TOKEN", "1"),
            weak_token_max_hits=num("GLOSSARY_WEAK_TOKEN_MAX_HITS", 3),
            cjk_boundary=flag("GLOSSARY_MATCH_CJK_BOUNDARY", "1"),
            short_cjk_len=num("GLOSSARY_MATCH_SHORT_CJK_LEN", 2),
            nfkc=flag("GLOSSARY_MATCH_NFKC", "1"),
            despaced=flag("GLOSSARY_MATCH_DESPACED", "1"),
            despaced_latin=flag("GLOSSARY_MATCH_DESPACED_LATIN", "0"),
            honorific=flag("GLOSSARY_MATCH_HONORIFIC", "1"),
            honorific_min_residual=num("GLOSSARY_MATCH_HONORIFIC_MIN_RESIDUAL", 2),
            derived_forms=flag("GLOSSARY_MATCH_DERIVED_FORMS", "1"),
        )


DEFAULT_CONFIG = MatchConfig()


# ─── Sub-word boundary heuristic ─────────────────────────────────────────────

def _kana_kind(ch):
    """'hira' / 'kata' for a kana character, '' otherwise."""
    cp = ord(ch) if ch else 0
    if 0x3040 <= cp <= 0x309F:
        return "hira"
    if 0x30A0 <= cp <= 0x30FF or 0x31F0 <= cp <= 0x31FF or 0xFF66 <= cp <= 0xFF9D:
        return "kata"
    return ""


def _korean_tail_is_boundary(tail, derived, relaxed=True):
    """Does this text, right after a Korean noun, mark the end of that noun?

    ``relaxed`` adds bound nouns and the copula (whole terms of 2+ syllables
    only). ``derived`` further adds 하다/되다 verb endings; see
    MatchConfig.derived_forms.
    """
    if not tail or script_of(tail[0]) != HANGUL:
        return True
    if tail.startswith(KOREAN_PARTICLE_PREFIXES) or tail.startswith(HONORIFIC_SUFFIXES):
        return True
    if not relaxed:
        return False
    if tail.startswith(KOREAN_BOUND_NOUN_PREFIXES) or tail.startswith(KOREAN_COPULA_PREFIXES):
        return True
    if tail[0] == "다":
        # Word-final copula only: 라베다. / 친구다고, never 구름다리.
        after = tail[1:2]
        if not after or script_of(after) != HANGUL or after in _KOREAN_DA_FOLLOWERS:
            return True
    return derived and tail.startswith(KOREAN_VERBALIZER_PREFIXES)


def _left_is_clean(text, start, script, derived=False, term_len=2, is_name=False):
    """True when nothing to the left glues the match into a longer word."""
    if start == 0:
        return True
    prev = text[start - 1]
    prev_script = script_of(prev)
    if prev_script not in _CJK_SCRIPTS:
        return True
    if script == HANGUL:
        # Korean compounds freely, and prose is not always spaced, so a Hangul
        # neighbour is genuinely ambiguous. Rejecting is the conservative call;
        # the cost is a name glued to the preceding word with no other
        # occurrence in the chapter.
        if prev_script != HANGUL:
            return True
        # A Sino-Korean prefix that itself begins the word (중장갑, 반기업) is a
        # compound built on the term, not a different word containing it.
        if (
            derived
            and term_len >= 2
            and prev in KOREAN_NOUN_PREFIXES
            and (start < 2 or script_of(text[start - 2]) != HANGUL)
        ):
            return True
        return False
    if script == HAN:
        # The left edge carries almost no signal in Chinese. Verb-object and
        # modifier-noun sequences are written solid (擦剑 "wipe [the] sword",
        # 老槐树 "old pagoda tree"), so demanding a non-Han character to the
        # left rejects ordinary prose. Measured on the eval corpus, requiring
        # it cost four true positives and prevented no false ones — the right
        # edge and the surname rule already do the work. So: accept, unless
        # this is a Japanese-style kanji term (handled by KANA/script change).
        if term_len == 1 and is_name:
            # A one-hanzi NAME is a different matter: glued to a hanzi on
            # its left it is nearly always the second half of a two-character
            # word (明白, 空白, 表白 are not the surname 白). Only a function
            # word or a name-introducer to the left leaves it standing alone.
            # Not for things: 擦剑 "wipe the sword" really is the item 剑.
            return prev in _HAN_SINGLE_LEFT_OPENERS
        single, compound = _chinese_surnames()
        if prev in single:
            return True
        if start >= 2 and text[start - 2:start] in compound:
            return True
        return True
    if script == KANA:
        if prev_script != KANA:
            return True
        # Katakana after hiragana (or the reverse) is a word edge: …とアリス.
        first_kind = _kana_kind(text[start]) if start < len(text) else ""
        if _kana_kind(prev) != first_kind:
            return True
        return first_kind == "hira" and prev in _JAPANESE_LEFT_PARTICLES
    return True


def _right_is_clean(text, end, script, derived=False, term_len=2, relaxed=True):
    """True when nothing to the right glues the match into a longer word."""
    if end >= len(text):
        return True
    nxt = text[end]
    nxt_script = script_of(nxt)
    if nxt_script not in _CJK_SCRIPTS:
        return True
    if script == HANGUL:
        # A josa, the copula, or a name honorific immediately after the term
        # is the normal way Korean uses a noun, so it counts as a boundary
        # rather than a collision. Without this the heuristic would reject
        # almost every true positive in Korean.
        if nxt_script != HANGUL:
            return True
        tail = text[end:end + 6]
        if _korean_tail_is_boundary(tail, derived, relaxed):
            return True
        # 장갑판을, 완충재다: term + one noun suffix that itself ends the word.
        if (
            derived
            and term_len >= 2
            and nxt in KOREAN_NOUN_SUFFIXES
            and _korean_tail_is_boundary(text[end + 1:end + 7], derived, relaxed)
        ):
            return True
        return False
    if script == HAN:
        tail = text[end:end + 4]
        if tail.startswith(HONORIFIC_SUFFIXES) or tail.startswith(_HAN_NAME_SUFFIXES):
            return True
        # A grammatical particle cannot be half of a compound noun, so it marks
        # a real word edge: 宋家的人 references 宋家, while 天下第一楼 does not
        # reference 天下.
        if tail.startswith(CHINESE_FUNCTION_WORDS):
            return True
        # 小明说 / 小明走了 / 小明忽然: what follows a subject, not a compound.
        if nxt in CHINESE_SUBJECT_FOLLOWERS:
            return True
        return nxt_script != HAN
    if script == KANA:
        if text[end:end + 4].startswith(HONORIFIC_SUFFIXES):
            return True
        if nxt_script != KANA:
            return True
        # アリスは: katakana name, hiragana particle. The switch is the edge.
        last_kind = _kana_kind(text[end - 1]) if end else ""
        if _kana_kind(nxt) != last_kind:
            return True
        return last_kind == "hira" and text[end:end + 3].startswith(JAPANESE_PARTICLE_PREFIXES)
    return True


def _cjk_occurrence_is_clean(text, term, cfg, is_character=False, whole=True):
    """True when `term` occurs in `text` at least once without being glued
    inside a longer word.

    Only applied to short CJK terms: a four-character term buried inside a
    longer one is rare enough that the recall risk outweighs the precision
    gain. Scans occurrences with early exit, so the cost is proportional to
    the number of hits, not the length of the text.
    """
    script = term_script(term)
    if script not in _CJK_SCRIPTS:
        return True
    if not cfg.cjk_boundary or len(term) > cfg.short_cjk_len:
        return term in text
    single, compound = _chinese_surnames() if script == HAN else (frozenset(), frozenset())
    # Verb endings and noun affixes never attach to a person's name, and one of
    # them would readmit a real false positive (유리한 "favourable" is not the
    # character 유리), so they are for non-character entries only.
    relaxed = bool(whole) and len(term) >= 2
    derived = relaxed and bool(getattr(cfg, "derived_forms", True)) and not is_character
    start = 0
    while True:
        idx = text.find(term, start)
        if idx < 0:
            return False
        end = idx + len(term)
        left = _left_is_clean(text, idx, script, derived, len(term), is_character)
        right = _right_is_clean(text, end, script, derived, len(term), relaxed)
        if left and right:
            return True
        if script == HAN:
            # Chinese prose has no spaces, so the character after a name is
            # almost always another Han character (小明走了). Requiring a clean
            # right edge there would reject nearly every true positive — the
            # same mistake that a naive "next char is Hangul" rule makes in
            # Korean. A surname immediately to the left is independent
            # evidence that this is a name, so it stands in for the right edge.
            prev = text[idx - 1] if idx else ""
            # ...for a name of two characters. With one, the "surname" is as
            # likely the first half of an ordinary word: 明 in 明白.
            if len(term) == 1 and is_character:
                prev = ""
            if prev and (prev in single or (idx >= 2 and text[idx - 2:idx] in compound)):
                return True
        start = idx + 1


# ─── Script-aware containment ────────────────────────────────────────────────

@lru_cache(maxsize=8192)
def build_term_pattern(term):
    """Word-boundary regex for ASCII edges, plain escape for CJK edges.

    Each edge is decided independently so a mixed term such as "SS급 루나"
    gets a boundary on its ASCII side and none on its Hangul side.
    """
    left = r"(?<![A-Za-z0-9_])" if (term[0].isascii() and term[0].isalnum()) else ""
    right = r"(?![A-Za-z0-9_])" if (term[-1].isascii() and term[-1].isalnum()) else ""
    return re.compile(left + re.escape(term) + right)


def text_contains_variant(text, variant, cfg, is_character=False, whole=True):
    """Does `variant` appear in `text` under the script-appropriate rule?"""
    if not variant or not text:
        return False
    if variant not in text:
        return False
    script = term_script(variant)
    if script in _CJK_SCRIPTS:
        return _cjk_occurrence_is_clean(text, variant, cfg, is_character, whole)
    # Latin (and anything else): require word boundaries so "Al" stops
    # matching inside "Already".
    return build_term_pattern(variant).search(text) is not None


# ─── Variant generation ──────────────────────────────────────────────────────

def strip_name_honorific(term, cfg):
    """Remove ONE trailing name honorific, or return "" if none applies.

    Stripping is single-pass and length-guarded on purpose: iterating would
    chew through a name one honorific-shaped syllable at a time.
    """
    for suffix in HONORIFIC_SUFFIXES:
        if len(suffix) < len(term) and term.endswith(suffix):
            residual = term[:-len(suffix)].strip(" -")
            min_residual = cfg.honorific_min_residual
            if residual.isascii():
                min_residual = max(min_residual, 3)
            if len(residual) >= min_residual:
                return residual
            return ""
    return ""


# Sized for the cross-novel unified glossary (tens of thousands of terms). At
# 8192 every request evicted and recomputed most of them; an entry is a few
# short strings, so holding them all costs single-digit megabytes.
@lru_cache(maxsize=131072)
def _variants_cached(term, cache_key):
    nfkc, despaced, despaced_latin, honorific, honorific_min_residual = cache_key
    cfg = MatchConfig(
        nfkc=nfkc, despaced=despaced, despaced_latin=despaced_latin,
        honorific=honorific, honorific_min_residual=honorific_min_residual,
    )
    out = []
    seen = set()

    def add(value, tier):
        value = value.strip()
        # Keyed on (value, tier) rather than value alone: the same surface form
        # is searched in a different haystack at each tier, so an unchanged
        # NFKC form still has work to do — ｱﾘｽ in the source only matches the
        # term アリス once the *text* has been normalized.
        if value and (value, tier) not in seen:
            seen.add((value, tier))
            out.append((value, tier))

    term = term.strip()
    if not term:
        return ()
    add(term, TIER_EXACT)

    base = term
    if nfkc:
        folded = unicodedata.normalize("NFKC", term).casefold()
        add(folded, TIER_NORM)
        base = folded

    if despaced and _DESPACE_RE.search(base):
        # Only for CJK unless explicitly allowed: despacing "Al Gore" into
        # "algore" would cross a real word boundary.
        if despaced_latin or term_script(base) in _CJK_SCRIPTS:
            add(_DESPACE_RE.sub("", base), TIER_DESPACED)

    if honorific:
        stripped = strip_name_honorific(base, cfg)
        if stripped:
            add(stripped, TIER_HONORIFIC)

    parts = name_parts(base)
    if len(parts) > 1:
        for token in parts:
            add(token, TIER_TOKEN if len(token) >= 2 else TIER_WEAK)

    return tuple(out)


# What separates the parts of a name. Whitespace, plus the marks CJK text
# uses for transliterated names: アリス・リデル, ジャン＝ジャック (NFKC makes
# ＝ into =), 哈利·波特. A hyphen is not one: Al-Gore is a single surname.
_NAME_PART_SPLIT_RE = re.compile(r"[\s\u30fb\u00b7\u2022\u2027=]+")


def name_parts(term):
    """The separately usable parts of a (normalized) multi-part name."""
    return [part for part in _NAME_PART_SPLIT_RE.split(str(term or "")) if part]


def term_variants(term, cfg=None):
    """Surface forms of `term` to look for, paired with their evidence tier."""
    cfg = cfg or DEFAULT_CONFIG
    return _variants_cached(str(term or ""), cfg.cache_key)


# ─── Prepared per-chapter index ──────────────────────────────────────────────

def prepare_source_text(text, cfg=None):
    """Precompute the normalized forms and reject-sets for one chapter.

    Mirrors prepare_translated_output_text on the usage side. The two CJK
    additions are what make the richer matching affordable: `cjk_chars` gates
    the single-character tier, and `cjk_bigrams` rejects a short CJK term in
    O(1) before any scan is attempted.
    """
    cfg = cfg or DEFAULT_CONFIG
    text = str(text or "")
    norm = unicodedata.normalize("NFKC", text).casefold() if cfg.nfkc else text
    cjk_chars = Counter(ch for ch in norm if is_cjk_char(ch))
    bigrams = set()
    prev = ""
    for ch in norm:
        if is_cjk_char(ch):
            if prev:
                bigrams.add(prev + ch)
            prev = ch
        else:
            prev = ""
    return {
        "text": text,
        "norm": norm,
        "despaced": _DESPACE_RE.sub("", norm),
        "tokens": frozenset(_TOKEN_RE.findall(norm)),
        "cjk_chars": cjk_chars,
        "cjk_bigrams": frozenset(bigrams),
    }


def _haystacks(prepared, tier):
    """Which normalized form a variant of this tier should be searched in."""
    if tier == TIER_EXACT:
        # Raw text only: matching the normalized text is what TIER_NORM is for,
        # and reporting it as "exact" would overstate the evidence.
        return (prepared["text"],)
    if tier == TIER_DESPACED:
        # The despaced variant must be found in despaced text, but a glossary
        # term spelled with a space may also appear spaced in the source.
        return (prepared["despaced"], prepared["norm"])
    return (prepared["norm"], prepared["text"])


def _cheap_reject(prepared, variant):
    """O(1) pre-filter. Returns True when the variant certainly cannot match."""
    if len(variant) == 1:
        ch = variant
        if is_cjk_char(ch):
            return ch not in prepared["cjk_chars"]
        return False
    first, second = variant[0], variant[1]
    if is_cjk_char(first) and is_cjk_char(second):
        return (first + second) not in prepared["cjk_bigrams"]
    return False


class MatchResult:
    """The strongest evidence found for a term, and why it was accepted."""

    __slots__ = ("matched", "tier", "variant", "rule", "reject_reason")

    def __init__(self, matched=False, tier=-1, variant="", rule="", reject_reason=""):
        self.matched = matched
        self.tier = tier
        self.variant = variant
        self.rule = rule
        self.reject_reason = reject_reason

    def __bool__(self):
        return self.matched

    def __repr__(self):
        if self.matched:
            return f"<MatchResult {self.rule} tier={self.tier} variant={self.variant!r}>"
        return f"<MatchResult no-match reason={self.reject_reason!r}>"


NO_MATCH = MatchResult()


def _weak_token_allowed(prepared, variant, is_character, cfg):
    """The gates on the single-character tier.

    This is the rule that made compression 'very broad': a lone 미 or 李 kept
    a character entry. It survives only for CJK characters that are rare in
    the chapter and that stand clear of the words around them.
    """
    if not cfg.allow_weak_token or not is_character:
        return False, "weak_token_disabled"
    if variant.isascii():
        # A bare ASCII letter is noise, never a name reference.
        return False, "weak_token_ascii"
    hits = prepared["cjk_chars"].get(variant, 0)
    if hits > cfg.weak_token_max_hits:
        return False, "weak_token_too_common"
    return True, ""


def match_term(prepared, term, is_character=False, cfg=None, token_filter=None):
    """Find the strongest tier at which `term` is present in the chapter.

    ``token_filter(token) -> bool`` can veto a match on one part of a
    multi-part term (see GlossaryNameIndex.is_name_part); whole-term tiers
    never consult it.
    """
    cfg = cfg or DEFAULT_CONFIG
    term = str(term or "").strip()
    if not term or not prepared:
        return NO_MATCH

    variants = term_variants(term, cfg)
    if not variants:
        return NO_MATCH

    floor = cfg.min_tier

    best_reject = "no_occurrence"
    for variant, tier in variants:
        if tier < floor:
            best_reject = "below_min_tier"
            continue
        if _cheap_reject(prepared, variant):
            continue
        found = any(
            text_contains_variant(hay, variant, cfg, is_character, whole=tier > TIER_TOKEN)
            for hay in _haystacks(prepared, tier)
        )
        if not found:
            if len(variant) <= cfg.short_cjk_len and term_script(variant) in _CJK_SCRIPTS:
                best_reject = "cjk_boundary"
            elif variant.isascii():
                best_reject = "word_boundary"
            continue
        if tier <= TIER_TOKEN and token_filter is not None and not token_filter(variant):
            # Asked only about parts that do occur here, so the filter can
            # afford real work (see GlossaryNameIndex.is_name_part).
            best_reject = "generic_name_part"
            continue
        if tier == TIER_WEAK:
            allowed, reason = _weak_token_allowed(prepared, variant, is_character, cfg)
            if not allowed:
                best_reject = reason
                continue
        return MatchResult(True, tier, variant, TIER_NAMES.get(tier, str(tier)))

    return MatchResult(False, -1, "", "", best_reject)


def text_contains_term(text, term, is_character=False, cfg=None):
    """Convenience wrapper for callers that have no prepared index."""
    return bool(match_term(prepare_source_text(text, cfg), term, is_character, cfg))


# ─── Whole-term scope (part of Precise Term Matching) ────────────────────────
#
# Measured on a 704-entry glossary, two thirds of what the precise matcher
# still kept per chapter was kept because ONE word of a multi-word entry
# appeared: 넥서스 alone kept 모노크롬 넥서스, 넥서스 신권 and four more in
# nearly every chapter. So entries in scope must appear whole. The one place
# a part is real evidence is a person's name (미샤 for 미샤 랄토스), which is
# what gender-enabled entries are for; there a part still counts when it is
# a name rather than a title, see GlossaryNameIndex.

_PREPARED_CACHE = OrderedDict()
_PREPARED_CACHE_LOCK = threading.Lock()
_PREPARED_CACHE_SIZE = 8


def prepare_source_text_cached(text, cfg=None):
    """prepare_source_text with a small LRU, for callers that match one
    chapter against many terms through a per-term API."""
    cfg = cfg or DEFAULT_CONFIG
    text = str(text or "")
    key = (len(text), hash(text), cfg.nfkc)
    with _PREPARED_CACHE_LOCK:
        prepared = _PREPARED_CACHE.get(key)
        if prepared is not None and prepared["text"] == text:
            _PREPARED_CACHE.move_to_end(key)
            return prepared
    prepared = prepare_source_text(text, cfg)
    with _PREPARED_CACHE_LOCK:
        _PREPARED_CACHE[key] = prepared
        while len(_PREPARED_CACHE) > _PREPARED_CACHE_SIZE:
            _PREPARED_CACHE.popitem(last=False)
    return prepared


def _copy_config(base, **changes):
    base = base or DEFAULT_CONFIG
    values = {name: getattr(base, name) for name in MatchConfig.__slots__ if name != "cache_key"}
    values.update(changes)
    return MatchConfig(**values)


def whole_term_config(base=None):
    """`base`, but only whole-term forms count: exact, width/case, spacing
    (미샤 랄토스 / 미샤랄토스) and an attached honorific (루나님 / 루나)."""
    base = base or DEFAULT_CONFIG
    return _copy_config(base, min_tier=max(base.min_tier, TIER_HONORIFIC), allow_weak_token=False)


def name_part_config(base=None):
    """`base`, accepting a part of a name of two or more characters. A
    one-character part (김, 白) is never distinctive enough on its own."""
    base = base or DEFAULT_CONFIG
    # Never below the caller's own floor: GLOSSARY_MATCH_MIN_TIER above the
    # token tier switches name parts off along with every other token match.
    return _copy_config(base, min_tier=max(base.min_tier, TIER_TOKEN), allow_weak_token=False)


WHOLE_TERM_SCOPE_MODES = ("all", "gender", "custom", "none")
_SCOPE_MODE_ALIASES = {
    "characters": "gender", "character": "gender", "gendered": "gender",
    "gender_entries": "gender", "all_fields": "all", "everything": "all",
    "off": "none", "": "all",
}


def parse_strict_scope(mode, custom_types=None):
    """Normalize the whole-term scope to ``(mode, types)``.

    ``all`` (default) — every entry must appear whole. ``gender`` — only
    gender-enabled entries. ``custom`` — the listed entry types. ``none`` —
    no whole-term requirement: one word of a multi-word entry is enough.

    The type list may arrive as a list or as the JSON / comma string an
    environment variable carries; singular and plural spellings are both
    accepted, because section headers say CHARACTERS and rows say character.
    ``custom`` with nothing listed falls back to ``all``.
    """
    mode = str(mode or "all").strip().lower().replace(" ", "_")
    mode = _SCOPE_MODE_ALIASES.get(mode, mode)
    if mode not in WHOLE_TERM_SCOPE_MODES:
        mode = "all"
    if isinstance(custom_types, str):
        raw = custom_types.strip()
        try:
            custom_types = json.loads(raw) if raw.startswith("[") else raw.split(",")
        except ValueError:
            custom_types = raw.split(",")
    allowed = set()
    for item in custom_types or ():
        name = str(item or "").strip().lower()
        if not name:
            continue
        allowed.add(name)
        allowed.add(name[:-1] if name.endswith("s") else name + "s")
    if mode == "custom" and not allowed:
        mode = "all"
    return mode, frozenset(allowed)


DEFAULT_STRICT_SCOPE = parse_strict_scope("all")
NO_STRICT_SCOPE = parse_strict_scope("none")


def in_strict_scope(scope, entry_type="", is_gender_entry=False):
    """Must this entry appear as a whole term?

    ``is_gender_entry`` is "this entry's type has gender enabled (or the row
    carries a gender)" -- custom entry types included, not just `character`.
    """
    mode, allowed = scope or DEFAULT_STRICT_SCOPE
    if mode == "all":
        return True
    if mode == "none":
        return False
    if mode == "custom":
        return str(entry_type or "").strip().lower() in allowed
    return bool(is_gender_entry)


_ADDRESS_TERMS = frozenset(
    unicodedata.normalize("NFKC", suffix).casefold().lstrip("-") for suffix in HONORIFIC_SUFFIXES
)


class GlossaryNameIndex:
    """May one part of a gender-enabled entry's name stand for the entry?

    Gender-enabled entries are people, but half of them are stored as
    epithets (3서클 마법사, 유카 엄마, 마왕의 제자), and their head nouns are
    ordinary words that turn up in most chapters. Three tests, in order:

    1. The part is a glossary entry in its own right (피엘 for 피엘 메스,
       넥서스 for 넥서스 파일럿): no. That row already carries the word, so
       nothing is lost.
    2. The part is transliterated in this entry's translation (카인 -> Kain,
       리즈 -> Liz) rather than translated (마법사 -> Mage): it is a name, yes.
    3. When that cannot be judged (Han characters, a non-Latin translation):
       yes unless ``shared_limit`` or more entries share the part.
    """

    __slots__ = ("own", "share", "shared_limit")

    def __init__(self, raw_names=(), shared_limit=3):
        self.own = set()
        self.share = Counter()
        self.shared_limit = int(shared_limit)
        for raw in raw_names:
            self.add(raw)

    @staticmethod
    def _fold(value):
        return unicodedata.normalize("NFKC", str(value or "")).casefold().strip()

    def add(self, raw_name):
        folded = self._fold(raw_name)
        if not folded:
            return
        self.own.add(folded)
        parts = set(name_parts(folded))
        if len(parts) > 1:
            for part in parts:
                self.share[part] += 1

    def is_name_part(self, part, translated_name="", raw_name=""):
        part = self._fold(part)
        if not part or part in self.own:
            return False
        if any(ch.isdigit() for ch in part):
            return False  # 1호, 3서클, 9번대: a rank or a number, never a name
        if part in _ADDRESS_TERMS:
            return False  # 루나 선배, 유카 언니: the form of address is not the name
        siblings = tuple(name_parts(self._fold(raw_name))) if raw_name else ()
        verdict = is_transliterated(part, str(translated_name or ""), siblings)
        if verdict is None:
            return self.share.get(part, 0) < self.shared_limit
        return verdict


class ScopedMatcher:
    """Precise matching with the whole-term scope applied.

    One object per chapter. Shared by the compressor and the glossary
    editor's usage view so the two cannot disagree about what is sent.
    """

    __slots__ = ("prepared", "cfg", "scope", "index", "_whole", "_part")

    def __init__(self, prepared, cfg=None, scope=None, index=None):
        self.prepared = prepared
        self.cfg = cfg or DEFAULT_CONFIG
        self.scope = scope or DEFAULT_STRICT_SCOPE
        self.index = index
        self._whole = None
        self._part = None

    def match(self, term, is_gender_entry=False, entry_type="", translated_name=None):
        """``translated_name`` is what lets a name part be recognised; leave
        it None to require the whole term (e.g. when `term` IS the
        translated name)."""
        if not in_strict_scope(self.scope, entry_type, is_gender_entry):
            return match_term(self.prepared, term, is_gender_entry, self.cfg)
        if self._whole is None:
            self._whole = whole_term_config(self.cfg)
            self._part = name_part_config(self.cfg)
        result = match_term(self.prepared, term, is_gender_entry, self._whole)
        if result or not is_gender_entry or self.index is None or translated_name is None:
            return result
        # A person is usually referred to by one part of their name.
        index = self.index
        part = match_term(self.prepared, term, True, self._part,
                          token_filter=lambda piece: index.is_name_part(piece, translated_name, term))
        # Report "generic_name_part" rather than the whole-term miss, so the
        # match-differences report says why 넥서스 did not keep 넥서스 파일럿.
        return part if (part or part.reject_reason == "generic_name_part") else result


# ─── Review overrides ────────────────────────────────────────────────────────

ALLOWLIST_SUFFIX = "_match_allowlist.json"


def allowlist_path_for(glossary_path):
    """Where a glossary's reviewed match overrides live.

    Pure path arithmetic so the compressor (which reads the file) and the
    verdict tool (which writes it) cannot disagree about the location.
    Sits beside the glossary, like the gender tracker does.
    """
    glossary_path = str(glossary_path or "").strip()
    if not glossary_path:
        return ""
    directory = os.path.dirname(glossary_path)
    stem = os.path.splitext(os.path.basename(glossary_path))[0]
    return os.path.join(directory, stem + ALLOWLIST_SUFFIX)


def normalize_override_terms(values):
    """Casefolded set for matching, ignoring blanks."""
    return frozenset(
        norm_text(v).casefold() for v in (values or []) if norm_text(v)
    )
