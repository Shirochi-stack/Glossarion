# -*- coding: utf-8 -*-
"""Shared term-matching primitives for glossary compression and usage.

This module is deliberately a *leaf*: it imports nothing from the rest of the
project.  Both ``glossary_compressor`` and ``glossary_usage`` need the same
matching logic, but ``glossary_compressor`` pulls in ``gender_tracking``,
``extract_glossary_from_epub`` and ``GlossaryManager``.  That import weight is
why ``glossary_usage`` used to carry a hand-copied duplicate of the matcher as
an ImportError fallback — a copy that could silently drift from the original.
Keeping the primitives here means both sides import the same code.

For the same reason this module never reads settings itself.  Callers build a
``MatchConfig`` from whatever getter they use (``_setting`` in the compressor,
``os.getenv`` in usage) and pass it in.
"""

import re
import unicodedata
from collections import Counter
from functools import lru_cache


# ─── Text primitives ─────────────────────────────────────────────────────────
# Moved here verbatim from glossary_usage so the translated-output matchers and
# the source-text matchers cannot disagree about what a token or a fold is.

_WHITESPACE_RE = re.compile(r"\s+")
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

def legacy_text_contains_term(text, term, is_character=False, strict_gender=False):
    """The pre-tiered matcher, preserved exactly.

    Substring matching in every script, plus a whitespace-token fallback whose
    minimum token length drops to 1 for character entries.  That 1-char floor
    is the main over-matching source, but it is load-bearing for recall: it is
    what rescues a glossary term spelled with a space ("미샤 랄토스") when the
    source text runs it together ("미샤랄토스").

    ``strict_gender`` is passed in rather than read from the environment so
    this function stays pure and testable.
    """
    if not term or not text:
        return False

    # Full term match first (fast path)
    if term in text:
        return True

    if is_character and strict_gender:
        return False

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
        "honorific_min_residual", "strict_gender", "cache_key",
    )

    def __init__(self, min_tier=TIER_TOKEN, allow_weak_token=True,
                 weak_token_max_hits=3, cjk_boundary=True, short_cjk_len=2,
                 nfkc=True, despaced=True, despaced_latin=False,
                 honorific=True, honorific_min_residual=2, strict_gender=False):
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
        self.strict_gender = bool(strict_gender)
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
            strict_gender=flag("COMPRESS_GLOSSARY_STRICT_GENDER_MATCHING", "0"),
        )


DEFAULT_CONFIG = MatchConfig()


# ─── Sub-word boundary heuristic ─────────────────────────────────────────────

def _left_is_clean(text, start, script):
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
        return prev_script != HANGUL
    if script == HAN:
        # The left edge carries almost no signal in Chinese. Verb-object and
        # modifier-noun sequences are written solid (擦剑 "wipe [the] sword",
        # 老槐树 "old pagoda tree"), so demanding a non-Han character to the
        # left rejects ordinary prose. Measured on the eval corpus, requiring
        # it cost four true positives and prevented no false ones — the right
        # edge and the surname rule already do the work. So: accept, unless
        # this is a Japanese-style kanji term (handled by KANA/script change).
        single, compound = _chinese_surnames()
        if prev in single:
            return True
        if start >= 2 and text[start - 2:start] in compound:
            return True
        return True
    if script == KANA:
        return prev_script != KANA
    return True


def _right_is_clean(text, end, script):
    """True when nothing to the right glues the match into a longer word."""
    if end >= len(text):
        return True
    nxt = text[end]
    nxt_script = script_of(nxt)
    if nxt_script not in _CJK_SCRIPTS:
        return True
    if script == HANGUL:
        # A josa or a name honorific immediately after the term is the normal
        # way Korean uses a name, so it counts as a boundary rather than a
        # collision. Without this the heuristic would reject almost every
        # true positive in Korean.
        tail = text[end:end + 6]
        if tail.startswith(KOREAN_PARTICLE_PREFIXES):
            return True
        if tail.startswith(HONORIFIC_SUFFIXES):
            return True
        return nxt_script != HANGUL
    if script == HAN:
        tail = text[end:end + 4]
        if tail.startswith(HONORIFIC_SUFFIXES):
            return True
        # A grammatical particle cannot be half of a compound noun, so it marks
        # a real word edge: 宋家的人 references 宋家, while 天下第一楼 does not
        # reference 天下.
        if tail.startswith(CHINESE_FUNCTION_WORDS):
            return True
        return nxt_script != HAN
    if script == KANA:
        if text[end:end + 4].startswith(HONORIFIC_SUFFIXES):
            return True
        return nxt_script != KANA
    return True


def _cjk_occurrence_is_clean(text, term, cfg):
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
    start = 0
    while True:
        idx = text.find(term, start)
        if idx < 0:
            return False
        end = idx + len(term)
        left = _left_is_clean(text, idx, script)
        right = _right_is_clean(text, end, script)
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


def text_contains_variant(text, variant, cfg):
    """Does `variant` appear in `text` under the script-appropriate rule?"""
    if not variant or not text:
        return False
    if variant not in text:
        return False
    script = term_script(variant)
    if script in _CJK_SCRIPTS:
        return _cjk_occurrence_is_clean(text, variant, cfg)
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


@lru_cache(maxsize=8192)
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

    if despaced and _WHITESPACE_RE.search(base):
        # Only for CJK unless explicitly allowed: despacing "Al Gore" into
        # "algore" would cross a real word boundary.
        if despaced_latin or term_script(base) in _CJK_SCRIPTS:
            add(_WHITESPACE_RE.sub("", base), TIER_DESPACED)

    if honorific:
        stripped = strip_name_honorific(base, cfg)
        if stripped:
            add(stripped, TIER_HONORIFIC)

    if " " in base:
        for token in base.split():
            add(token, TIER_TOKEN if len(token) >= 2 else TIER_WEAK)

    return tuple(out)


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
        "despaced": _WHITESPACE_RE.sub("", norm),
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


def match_term(prepared, term, is_character=False, cfg=None):
    """Find the strongest tier at which `term` is present in the chapter."""
    cfg = cfg or DEFAULT_CONFIG
    term = str(term or "").strip()
    if not term or not prepared:
        return NO_MATCH

    variants = term_variants(term, cfg)
    if not variants:
        return NO_MATCH

    floor = cfg.min_tier
    if is_character and cfg.strict_gender:
        # Preserve the existing knob's meaning: gendered entries must match a
        # whole name, not a part of one.
        floor = max(floor, TIER_NORM)

    best_reject = "no_occurrence"
    for variant, tier in variants:
        if tier < floor:
            best_reject = "below_min_tier"
            continue
        if _cheap_reject(prepared, variant):
            continue
        found = any(
            text_contains_variant(hay, variant, cfg)
            for hay in _haystacks(prepared, tier)
        )
        if not found:
            if len(variant) <= cfg.short_cjk_len and term_script(variant) in _CJK_SCRIPTS:
                best_reject = "cjk_boundary"
            elif variant.isascii():
                best_reject = "word_boundary"
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
