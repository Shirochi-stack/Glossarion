"""Glossary name check for translated chapters (experimental).

For every glossary entry whose raw name occurs in a source chapter, the
translated chapter should contain the entry's translated name. When it does
not, the output is searched for a close spelling the model used instead
(``Walter Argent`` for glossary ``Valter Argent``, ``Gang Si-hyeon`` for
``Kang Si-hyeon``). Name-like text is found with a capitalization heuristic,
so no NER model or extra dependency is needed.

Measured on the local validation corpus (63 books, 11k chapters), the rules
below keep plurals, honorific suffixes, possessives and sentence words glued
to a name ("But Yoo So-jeong") from being reported as drift. Only character
entries with exactly one close variant are fixed automatically; everything
else is reported.

GUI-free: used by the translation pipeline (``GLOSSARY_NAME_CHECK``), the QA
scanner (``check_glossary_names``) and the QA scanner's fix dialog.
"""

import difflib
import html as html_lib
import json
import os
import re
import threading
from dataclasses import dataclass, field

# Sentence words a capitalized run can start with ("But Yoo So-jeong").
_LEADING_WORDS = frozenset((
    "a", "after", "ah", "although", "an", "and", "as", "at", "because", "before", "but",
    "by", "even", "for", "from", "he", "her", "his", "how", "i", "if", "in", "is", "it",
    "just", "my", "no", "now", "of", "oh", "on", "or", "our", "put", "she", "since", "so",
    "still", "that", "the", "their", "then", "they", "this", "though", "to", "until",
    "was", "we", "well", "what", "when", "where", "while", "who", "why", "with", "yes",
    "yet", "you", "your", "i'm", "it's", "that's", "there", "here", "i'd", "i'll", "i've",
))
_ROMAN_OR_NUMBER = re.compile(r"^(?:[IVXLCDM]+|\d+)$")
# Honorific suffixes joined with a hyphen ("So-jeong-ssi", "Minho-gong").
_HONORIFIC_SUFFIX = re.compile(
    r"-(?:ssi|nim|gong|eonni|unnie|onni|oppa|hyung|hyeong|noona|nuna|sunbae|seonbae|"
    r"sama|san|kun|chan|sensei|senpai|dono|jie|jiejie|ge|gege|xiong|shixiong|shijie|"
    r"shifu|shizun|ah|ya)$",
    re.IGNORECASE,
)
_POSSESSIVE = re.compile(r"(?:'s|’s|'|’)$")
_STUTTER = re.compile(r"^([A-Za-z])-(?=\1)", re.IGNORECASE)
_TAG_SPLIT = re.compile(r"(<[^>]+>)")
_DROP_BLOCKS = re.compile(r"(?is)<(script|style)\b[^>]*>.*?</\1\s*>")
_WORD = r"[A-Z][\w'’\-]*"
_CAPITALIZED_RUN = re.compile(
    r"(?<![\w'’\-])(" + _WORD + r"(?:\s+(?:(?:von|van|de|del|di|la|le|of|du)\s+)?" + _WORD + r")*)"
)
_LOWER_WORD = re.compile(r"(?<![\w'’\-])[a-z][a-z'’\-]+")
_SENTENCE_START = re.compile(r"(?:^|[.!?。！？…:;\"“”'‘’「」『』—\-\(\[]\s*)$")
_APOSTROPHE_FORMS = "(?:'|’|&#39;|&#x27;|&apos;|&rsquo;|&#8217;)"

KIND_VARIANT = "variant"
#: Raw names shorter than this are only fixed automatically when the variant differs in spacing / hyphens
SHORT_RAW = 3
KIND_AMBIGUOUS = "ambiguous"
# "Auto-fix strength" (glossary_name_check_autofix_level)
AUTOFIX_CONSERVATIVE, AUTOFIX_BALANCED, AUTOFIX_AGGRESSIVE = 1, 2, 3
AUTOFIX_LEVEL_NAMES = {1: "Conservative", 2: "Balanced", 3: "Aggressive"}
KIND_MISSING = "missing"


@dataclass(frozen=True)
class NameEntry:
    raw: str
    translated: str
    entry_type: str
    origin: str = ""  # "" = the book's glossary, else LAYER_EXTENSION / LAYER_UNIFIED


@dataclass
class Finding:
    raw: str
    expected: str
    entry_type: str
    kind: str
    variants: tuple = ()
    occurrences: int = 0
    origin: str = ""
    autofix_level: int = AUTOFIX_CONSERVATIVE

    @property
    def fixable(self):
        """The automatic-fix case, by "Auto-fix strength" (``autofix_level``):

        1 Conservative: one close spelling of a character name. A raw name shorter than
          ``SHORT_RAW`` characters (surnames like 박 / 강, words like 하나) also matches unrelated
          source text, and its fuzzy matches were wrong on the corpus ("Sparks" for Park, "Hanja"
          for Hana), so only a spacing / hyphen respelling of it ("Soyoon" for So-yoon) is fixed.
        2 Balanced: also the other entry types in scope (places, terms...), same rules otherwise.
        3 Aggressive: every close spelling found, several per name and short raw names included."""
        if self.kind == KIND_MISSING or not self.variants:
            return False
        if self.autofix_level >= AUTOFIX_AGGRESSIVE:
            return True
        if self.kind != KIND_VARIANT:
            return False
        if self.autofix_level < AUTOFIX_BALANCED and self.entry_type != "character":
            return False
        return len(self.raw) >= SHORT_RAW or all(_norm(v) == _norm(self.expected) for v in self.variants)

    def describe(self):
        if self.kind == KIND_MISSING:
            return f"{self.expected} missing"
        return f"{' / '.join(self.variants)} → {self.expected}"


@dataclass
class ChapterResult:
    output_file: str
    findings: list = field(default_factory=list)

    @property
    def fixable(self):
        return [f for f in self.findings if f.fixable]

    @property
    def reportable(self):
        """Findings worth showing: drift, never plain absences (pronouns, rewording)."""
        return [f for f in self.findings if f.kind != KIND_MISSING]


def entry_type_allowed(entry_type, mode, custom_types=()):
    """The Emergency Glossary Compliance entry-type filter (Characters / All / Custom)."""
    entry_type = str(entry_type or "").strip().lower()
    mode = str(mode or "characters").strip().lower()
    if mode == "all":
        return True
    if mode == "custom" and custom_types:
        # "terms" matches type "term" and the other way round
        allowed = set()
        for t in custom_types:
            t_lower = str(t).lower().strip()
            allowed.add(t_lower)
            allowed.add(t_lower[:-1] if t_lower.endswith("s") else t_lower + "s")
        return entry_type in allowed
    return entry_type == "character"


def html_to_text(markup):
    text = _DROP_BLOCKS.sub(" ", markup or "")
    return html_lib.unescape(_TAG_SPLIT.sub(" ", text))


def _norm(text):
    return re.sub(r"[^\w]|_|\d", "", str(text or "").casefold())


def _clean_translation(name):
    """Glossary translations may carry notes: "Yeomra Turyong (Minho Namgung)"."""
    return re.sub(r"\s*[\(\[][^\)\]]*[\)\]]", "", str(name or "")).strip()


def _clean_candidate(run):
    words = run.split()
    while len(words) > 1 and words[0].casefold().replace("’", "'") in _LEADING_WORDS:
        words = words[1:]
    # ...and end with one ("Sister-in-law In", "Al'Shel I'd")
    while len(words) > 1 and words[-1].casefold().replace("’", "'") in _LEADING_WORDS:
        words = words[:-1]
    if not words:
        return ""
    words[-1] = _POSSESSIVE.sub("", words[-1])
    words[-1] = _HONORIFIC_SUFFIX.sub("", words[-1])
    words[0] = _STUTTER.sub("", words[0])
    return " ".join(w for w in words if w)


def _is_inflection(a, b):
    """True when two normalized names differ only by an English plural ending."""
    if a == b:
        return True
    short, long_ = sorted((a, b), key=len)
    return (long_ in (short + "s", short + "es")
            or (short.endswith("y") and long_ == short[:-1] + "ies"))


def _fold(text):
    """Case and apostrophe style do not make a different name (Yuka’s / Yuka's)."""
    return str(text or "").casefold().replace("’", "'").replace("‘", "'")


def _expected_present(expected, text):
    pattern = r"(?<![\w])" + re.escape(_fold(expected)) + r"(?:s|es)?(?![\w])"
    return re.search(pattern, _fold(text)) is not None


def _name_candidates(text):
    """Capitalized runs in the output that are not ordinary words, keyed by ``_norm``."""
    lower_words = {w.replace("’", "'") for w in _LOWER_WORD.findall(text)}
    found = {}
    for match in _CAPITALIZED_RUN.finditer(text):
        run = match.group(1).strip(" -'’")
        if " " not in run:
            before = text[max(0, match.start() - 4):match.start()]
            if _SENTENCE_START.search(before):
                continue  # capitalization at a sentence start says nothing
            if run.casefold().replace("’", "'") in lower_words:
                continue  # the chapter also uses it as an ordinary word
        core = _clean_candidate(run)
        key = _norm(core)
        if not core or len(key) < 2:
            continue
        found.setdefault(key, core)
    return found


def _title_candidates(text):
    """Title Case puts a name in one capitalized run with its neighbours ("Walter Argent's Sword"), so
    in a chapter title / TOC label / book title every run of up to four capitalized words is a candidate."""
    found = {}
    for match in _CAPITALIZED_RUN.finditer(text):
        words = match.group(1).split()
        for i in range(len(words)):
            for j in range(i + 1, min(len(words), i + 4) + 1):
                core = _clean_candidate(" ".join(words[i:j]))
                key = _norm(core)
                if core and len(key) >= 2:
                    found.setdefault(key, core)
    return found


def _is_variant(expected, expected_key, candidate, candidate_key):
    if expected_key == candidate_key:
        # spacing / hyphenation drift ("Yoo So-jeong" vs "Yoo Sojeong"), not a case change
        return _fold(expected) != _fold(candidate)
    if _is_inflection(expected_key, candidate_key):
        return False  # "Black Iron Knights" is the same name
    if min(len(expected_key), len(candidate_key)) < 4 or len(expected.split()) != len(candidate.split()):
        return False  # "Son" is not "Sion"
    if expected_key.startswith(candidate_key):
        return False  # a cut-off name ("Shen Yin" for "Shen Yinxue") may be someone else
    if len(candidate_key) - len(expected_key) >= 3 and candidate_key.startswith(expected_key):
        return False  # letters glued on ("Mihailtal" for "Mihail") are another word, not a respelling
    if len(expected_key) - len(candidate_key) >= 3 and expected_key.endswith(candidate_key):
        return False  # cut off in front ("Lian Sect" for "Singlian Sect")
    if len(candidate_key) > len(expected_key) and candidate_key.endswith(expected_key):
        return False  # letters in front make another name ("Asakura" is not "Sakura")
    exp_last = re.split(r"[\s\-]+", expected)[-1]
    cand_last = re.split(r"[\s\-]+", candidate)[-1]
    if exp_last != cand_last and (_ROMAN_OR_NUMBER.match(exp_last) or _ROMAN_OR_NUMBER.match(cand_last)):
        return False  # "Milano-VI" is another unit than "Milano-V"
    return difflib.SequenceMatcher(None, expected_key, candidate_key).ratio() >= 0.8


class GlossaryNameEntries:
    """Entries to check plus every translated name in the glossary (never a variant)."""

    def __init__(self, entries, known_names, autofix_level=AUTOFIX_CONSERVATIVE):
        self.entries = tuple(entries)
        self.known = frozenset(known_names)
        self.autofix_level = autofix_level

    def __bool__(self):
        return bool(self.entries)


def build_entries(glossary_entries, *, mode="characters", custom_types=(), min_chars=0,
                  autofix_level=AUTOFIX_CONSERVATIVE):
    """Pick the entries to check from ``glossary_usage.parse_glossary_file`` rows."""
    known = set()
    picked = {}
    for row in glossary_entries or ():
        raw = str(row.get("raw_name") or "").strip()
        translated = _clean_translation(row.get("translated_name"))
        if not raw or not translated:
            continue
        known.add(_norm(translated))
        for part in re.split(r"[\s\-]+", translated):
            if part:
                known.add(_norm(part))
        entry_type = str(row.get("type") or "term").strip().lower()
        if not entry_type_allowed(entry_type, mode, custom_types):
            continue
        if min_chars and len(raw) < min_chars:
            continue
        if raw == translated or not re.search(r"[A-Za-z]", translated):
            continue  # untranslated names and non-Latin targets have nothing to check
        picked.setdefault(raw, NameEntry(raw, translated, entry_type, str(row.get("_origin") or "")))
    entries = sorted(picked.values(), key=lambda e: -len(e.raw))
    return GlossaryNameEntries(entries, known, autofix_level)


_load_lock = threading.Lock()
_load_cache = {}

LAYER_EXTENSION = "Glossary Extension"
LAYER_UNIFIED = "Unified Glossary"


def glossary_layers(glossary_path, settings=None):
    """``[(path, origin)]``: the book's glossary, then the glossaries a translation request adds when
    they are switched on (``TransateKRtoEN.secondary_glossary_paths``: Add Additional Glossary's
    extension, Enable Unified Glossary's shared file). ``settings`` as for the request builder
    (a dict of its env names; None = the environment)."""
    layers = [(glossary_path, "")]
    try:
        from TransateKRtoEN import secondary_glossary_paths
        extension, unified = secondary_glossary_paths(glossary_path, settings)
    except Exception:
        extension = unified = None
    layers += [(path, origin) for path, origin in ((extension, LAYER_EXTENSION), (unified, LAYER_UNIFIED)) if path]
    return layers


def layer_settings_from_config(config):
    """The request-builder settings ``glossary_layers`` reads, from a saved config (the GUI process,
    where no translation env is set)."""
    config = config if isinstance(config, dict) else {}
    on = lambda key, default=False: "1" if _truthy(config.get(key, default), default) else "0"
    return {
        "ADD_ADDITIONAL_GLOSSARY": on("add_additional_glossary"),
        "ENABLE_UNIFIED_GLOSSARY": on("enable_unified_glossary"),
        "UNIFIED_GLOSSARY_SOURCE_LANGUAGE": str(config.get("unified_glossary_source_language") or "auto").lower(),
        "UNIFIED_GLOSSARY_COMBINE_ALL_LANGUAGES": on("unified_glossary_combine_all_languages"),
        "OUTPUT_LANGUAGE": str(config.get("output_language") or "English"),
    }


def _layer_rows(layers):
    """Rows of every layer; a raw name is taken from the first layer that has it (book glossary >
    extension > unified glossary), whatever its type there."""
    import glossary_usage
    rows, seen = [], set()
    for path, origin in layers:
        layer_raws = set()
        for row in glossary_usage.parse_glossary_file(path):
            raw = str(row.get("raw_name") or "").strip()
            if not raw or raw in seen:
                continue
            layer_raws.add(raw)
            rows.append(dict(row, _origin=origin) if origin else row)
        seen |= layer_raws
    return rows


def load_entries(glossary_path, *, mode="characters", custom_types=(), min_chars=0, settings=None,
                 autofix_level=AUTOFIX_CONSERVATIVE):
    """The names to check from the book's glossary and the enabled extension / unified glossary
    (``glossary_layers``), parsed once per (file versions, options)."""
    if not glossary_path or not os.path.isfile(glossary_path):
        return GlossaryNameEntries((), ())
    layers = [(p, o) for p, o in glossary_layers(glossary_path, settings) if os.path.isfile(p)]
    stamps = []
    for path, origin in layers:
        info = os.stat(path)
        stamps.append((os.path.normcase(os.path.abspath(path)), origin, info.st_size, info.st_mtime_ns))
    key = (tuple(stamps), str(mode or "characters").lower(), tuple(custom_types or ()), int(min_chars or 0),
           int(autofix_level))
    with _load_lock:
        cached = _load_cache.get(key)
        if cached is not None:
            return cached
    entries = build_entries(_layer_rows(layers), mode=mode, custom_types=custom_types, min_chars=min_chars,
                            autofix_level=autofix_level)
    with _load_lock:
        _load_cache.clear()
        _load_cache[key] = entries
    return entries


def _ner_candidates(text, tagger):
    """Names the NER model finds in the output (person / place / organization / misc), keyed by
    ``_norm`` like ``_name_candidates``; NER also sees names at a sentence start and names that are
    ordinary words elsewhere ("Rose"), which the capitalization heuristic has to skip."""
    found = {}
    for entity in tagger.entities(text):
        core = _clean_candidate(entity.text)
        key = _norm(core)
        if core and len(key) >= 2:
            found.setdefault(key, core)
    return found


def check_chapter(source_text, output_markup, entries, output_file="", *, ner=None, title=False):
    """Compare one translated chapter with its source chapter. ``ner``: a ``ner_names.NerTagger``
    finds the candidate spellings (else the capitalization heuristic). ``title``: the text is a title in
    Title Case (``_title_candidates``)."""
    result = ChapterResult(output_file=output_file)
    if not entries or not source_text or not output_markup:
        return result
    output_text = html_to_text(output_markup)
    level = getattr(entries, "autofix_level", AUTOFIX_CONSERVATIVE)
    candidates = None
    matched_raws = []
    # a large (unified) glossary: skip entries whose first two characters never occur in the chapter
    bigrams = {source_text[i:i + 2] for i in range(len(source_text) - 1)}
    for entry in entries.entries:
        if len(entry.raw) >= 2 and entry.raw[:2] not in bigrams:
            continue
        count = source_text.count(entry.raw)
        if not count:
            continue
        # A raw name inside a longer matched one ("린" in "린 아르겐트") is only
        # expected when it also occurs on its own.
        covered = sum(source_text.count(longer) for longer in matched_raws if entry.raw in longer)
        matched_raws.append(entry.raw)
        if count <= covered:
            continue
        if _expected_present(entry.translated, output_text):
            # A close spelling next to the right one is more often a second
            # character (Adele / Adela) than drift, so only absences are searched.
            continue
        if candidates is None:
            # NER adds the names the heuristic cannot see (sentence starts, ordinary words); the
            # heuristic keeps the invented / lowercase-hyphen spellings the model does not tag (corpus)
            candidates = _name_candidates(output_text)
            if title:
                for key, core in _title_candidates(output_text).items():
                    candidates.setdefault(key, core)
            if ner is not None:
                for key, core in _ner_candidates(output_text, ner).items():
                    candidates.setdefault(key, core)
        expected_key = _norm(entry.translated)
        # Another glossary name is never drift; a respelling of this one
        # ("Yoo So-jeong" for "Yoo Sojeong") normalizes to the same key.
        variants = [core for key, core in candidates.items()
                    if (key == expected_key or key not in entries.known)
                    and _is_variant(entry.translated, expected_key, core, key)]
        if variants:
            kind = KIND_VARIANT if len(variants) == 1 else KIND_AMBIGUOUS
            occurrences = sum(len(_occurrences(v, output_markup)) for v in variants)
            result.findings.append(Finding(entry.raw, entry.translated, entry.entry_type, kind,
                                           tuple(sorted(variants)), occurrences, entry.origin, level))
        else:
            result.findings.append(Finding(entry.raw, entry.translated, entry.entry_type, KIND_MISSING,
                                           origin=entry.origin, autofix_level=level))
    return result


def _variant_pattern(variant):
    parts = [re.escape(p) for p in re.split(r"['’]", variant)]
    body = _APOSTROPHE_FORMS.join(parts)
    return re.compile(r"(?<![\w'’\-])" + body + r"(?![\w])")


def _occurrences(variant, markup):
    pattern = _variant_pattern(variant)
    hits = []
    for piece in _TAG_SPLIT.split(_DROP_BLOCKS.sub("", markup)):
        if piece.startswith("<"):
            continue
        hits.extend(pattern.findall(piece))
    return hits


def _snippets(text, patterns, width, limit):
    found = []
    for pattern in patterns:
        found.extend((m.start(), m.end()) for m in pattern.finditer(text))
    out = []
    for start, end in sorted(set(found))[:limit]:
        # whitespace runs become one space; the space next to the match stays
        before = re.sub(r"\s+", " ", text[max(0, start - width):start]).lstrip()
        after = re.sub(r"\s+", " ", text[end:end + width]).rstrip()
        out.append((("…" if start > width else "") + before, text[start:end],
                    after + ("…" if end + width < len(text) else "")))
    return out


def finding_snippets(output_markup, finding, *, width=90, limit=50):
    """``[(before, variant, after)]``: where the finding's variants occur in the translation (text
    only), in order, for the review's Preview."""
    text = html_to_text(output_markup)
    return _snippets(text, [_variant_pattern(v) for v in finding.variants], width, limit)


def source_snippets(source_text, finding, *, width=60, limit=20):
    """``[(before, raw name, after)]``: where the finding's raw name occurs in the source chapter."""
    return _snippets(source_text or "", [re.compile(re.escape(finding.raw))], width, limit)


def apply_fixes(output_markup, findings, *, reviewed=False):
    """Replace variants with the glossary name in text (never inside tags).

    Without ``reviewed`` only the automatic-fix cases (one variant of a
    character name) are replaced; a reviewed selection replaces every variant
    of every finding passed in.
    """
    replacements = [(_variant_pattern(v), f.expected)
                    for f in findings if (f.fixable or (reviewed and f.variants))
                    for v in f.variants]
    if not replacements:
        return output_markup, 0
    total = 0
    pieces = re.split(r"((?is:<(?:script|style)\b[^>]*>.*?</(?:script|style)\s*>)|<[^>]+>)",
                      output_markup)
    for i, piece in enumerate(pieces):
        if not piece or piece.startswith("<"):
            continue
        for pattern, expected in replacements:
            piece, n = pattern.subn(expected.replace("\\", "\\\\"), piece)
            total += n
        pieces[i] = piece
    return "".join(p for p in pieces if p is not None), total


def check_and_fix_file(output_path, source_text, entries, *, auto_fix, ner=None, record=False):
    """Check one output file; with ``auto_fix`` rewrite it with the fixable variants replaced.
    ``record``: keep the findings in the book's report (``record_findings``)."""
    with open(output_path, "r", encoding="utf-8", errors="ignore") as f:
        markup = f.read()
    result = check_chapter(source_text, markup, entries, os.path.basename(output_path), ner=ner)
    replaced = 0
    if auto_fix and result.fixable:
        new_markup, replaced = apply_fixes(markup, result.fixable)
        if replaced:
            _write_text(output_path, new_markup)
    if record:
        record_findings(os.path.dirname(output_path), result.output_file, result.reportable, markup,
                        fixed=bool(auto_fix and replaced))
    return result, replaced


def _write_text(path, text):
    tmp = path + ".namecheck.tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        f.write(text)
    os.replace(tmp, path)


def settings_from_env(env=None):
    """Entry types shared with Emergency Glossary Compliance; the name check's own minimum length
    (default 2: a 1-character raw name such as 강 also occurs inside ordinary words)."""
    env = os.environ if env is None else env
    mode = str(env.get("EMERGENCY_GLOSSARY_COMPLIANCE_MODE", "characters") or "characters").lower()
    try:
        custom_types = json.loads(env.get("EMERGENCY_GLOSSARY_COMPLIANCE_CUSTOM_TYPES", "[]") or "[]")
    except (TypeError, ValueError):
        custom_types = []
    try:
        min_chars = max(0, min(10, int(env.get("GLOSSARY_NAME_CHECK_MIN_CHARS", 2))))
    except (TypeError, ValueError):
        min_chars = 2
    return {"mode": mode, "custom_types": tuple(custom_types if mode == "custom" else ()),
            "min_chars": min_chars,
            "autofix_level": _autofix_level(env.get("GLOSSARY_NAME_CHECK_AUTOFIX_LEVEL"))}


def _autofix_level(value, default=AUTOFIX_CONSERVATIVE):
    try:
        return max(AUTOFIX_CONSERVATIVE, min(AUTOFIX_AGGRESSIVE, int(value)))
    except (TypeError, ValueError):
        return default


def settings_from_config(config, env=None):
    """Like ``settings_from_env`` but saved config keys win (the QA scanner's owner config)."""
    scope = settings_from_env(env)
    config = config if isinstance(config, dict) else {}
    if config.get("emergency_glossary_compliance_mode"):
        scope["mode"] = str(config["emergency_glossary_compliance_mode"]).lower()
    custom = config.get("emergency_glossary_compliance_custom_types")
    if isinstance(custom, (list, tuple)):
        scope["custom_types"] = tuple(custom) if scope["mode"] == "custom" else ()
    elif scope["mode"] != "custom":
        scope["custom_types"] = ()
    if "glossary_name_check_min_chars" in config:
        try:
            scope["min_chars"] = max(0, min(10, int(config["glossary_name_check_min_chars"])))
        except (TypeError, ValueError):
            pass
    if "glossary_name_check_autofix_level" in config:
        scope["autofix_level"] = _autofix_level(config["glossary_name_check_autofix_level"], scope["autofix_level"])
    return scope


def _truthy(value, default=True):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def use_ner_from_env(env=None):
    """"Use Named Entity Recognition model (NER)" (default on)."""
    env = os.environ if env is None else env
    return _truthy(env.get("GLOSSARY_NAME_CHECK_USE_NER"), True)


def use_ner_from_config(config, env=None):
    config = config if isinstance(config, dict) else {}
    if "glossary_name_check_use_ner" in config:
        return _truthy(config["glossary_name_check_use_ner"], True)
    return use_ner_from_env(env)


_ner_notes = set()
_ner_lock = threading.Lock()


def resolve_ner(use_ner, log=None, *, progress=None, cancel=None, retry=False):
    """The NER tagger when it is switched on and this build can run it (downloading the model on first
    use; ``progress`` gets ``manga_models.DownloadProgress`` reports, ``cancel()`` stops the download);
    None means the capitalization heuristic. Each fallback reason is logged once per process.

    A failed load (no network) is not tried again for every chapter: later calls use the heuristic
    until a ``retry`` call (the Check glossary names dialogs, where the user asked for the model)."""
    if not use_ner:
        return None
    try:
        import ner_names
    except ImportError:
        ner_names = None
    if ner_names is None or not ner_names.runtime_available():
        if log and "unsupported" not in _ner_notes:
            _ner_notes.add("unsupported")
            log("ℹ️ Glossary name check: NER is not supported in this build; using the capitalization heuristic")
        return None
    # parallel chapters wait here for the first load instead of each starting the download
    with _ner_lock:
        if "failed" in _ner_notes and not retry:
            return None
        try:
            tagger = ner_names.get_tagger(log, cancel, progress)
        except Exception as exc:
            if cancel is not None and cancel():
                return None  # the user stopped the download; a later run may download it
            if log and "failed" not in _ner_notes:
                log(f"⚠️ Glossary name check: the NER model could not be loaded ({exc}); using the capitalization heuristic")
            _ner_notes.add("failed")
            return None
        _ner_notes.discard("failed")
        return tagger


def enabled_from_env(env=None):
    env = os.environ if env is None else env
    return str(env.get("GLOSSARY_NAME_CHECK", "0")).strip().lower() in ("1", "true", "yes", "on")


def auto_fix_from_env(env=None):
    env = os.environ if env is None else env
    return str(env.get("GLOSSARY_NAME_CHECK_MODE", "auto")).strip().lower() != "review"


def source_chapters_for_folder(folder_path, progress_path=None):
    """Map each completed output file to its source chapter text.

    Uses ``translation_progress.json`` (output file, content hash, original
    basename) and the ``chapters_full.json`` extraction cache in the output
    folder. Returns ``{output_file: source_text}``; empty when either is missing.
    """
    progress_path = progress_path or os.path.join(folder_path, "translation_progress.json")
    full_path = os.path.join(folder_path, "chapters_full.json")
    if not (os.path.isfile(progress_path) and os.path.isfile(full_path)):
        return {}
    try:
        with open(full_path, "r", encoding="utf-8") as f:
            chapters = json.load(f)
        with open(progress_path, "r", encoding="utf-8") as f:
            progress = json.load(f)
    except (OSError, ValueError):
        return {}
    if not isinstance(chapters, list) or not isinstance(progress, dict):
        return {}
    by_hash = {}
    by_base = {}
    for chapter in chapters:
        if not isinstance(chapter, dict):
            continue
        if chapter.get("content_hash"):
            by_hash.setdefault(chapter["content_hash"], chapter)
        for key in ("original_basename", "original_filename", "filename"):
            value = chapter.get(key)
            if value:
                by_base.setdefault(os.path.splitext(os.path.basename(str(value)))[0], chapter)
    mapping = {}
    for info in (progress.get("chapters") or {}).values():
        if not isinstance(info, dict) or info.get("status") not in ("completed", "qa_failed"):
            continue
        output_file = info.get("output_file")
        if not output_file:
            continue
        chapter = by_hash.get(info.get("content_hash"))
        if chapter is None:
            base = os.path.splitext(os.path.basename(str(info.get("original_basename") or "")))[0]
            chapter = by_base.get(base)
        if chapter is not None and chapter.get("body"):
            mapping[os.path.basename(output_file)] = html_to_text(chapter["body"])
    return mapping


def max_workers_setting():
    """The QA scanner's "Maximum parallel workers" (AI Hunter performance settings; 0 = every core)."""
    try:
        from scan_html_folder import _resolve_scan_max_workers_config
        return _resolve_scan_max_workers_config()
    except Exception:
        return 1


def chapter_workers(ner, max_workers=None):
    """Chapters checked at once. Only NER calls run in parallel (the heuristic is pure Python), each
    using ``ner_names.ner_threads()`` cores, within the "Maximum parallel workers" budget: 16 cores and
    "All cores" check 4 chapters at once, 2 cores or "Single thread" one."""
    if ner is None:
        return 1
    cpus = os.cpu_count() or 1
    budget = max_workers_setting() if max_workers is None else max_workers
    budget = cpus if not budget or budget <= 0 else min(int(budget), cpus)
    try:
        import ner_names
        per_call = ner_names.ner_threads()
    except ImportError:
        per_call = 1
    return max(1, budget // per_call)


def scan_folder(folder_path, entries, *, progress_path=None, selected_files=None, stop=None, ner=None,
                progress=None, workers=None):
    """Check every mapped output file; returns ``[ChapterResult]`` with reportable findings, in file
    order. ``progress(done, total)`` is called as chapters finish; with NER ``workers`` (default
    ``chapter_workers``) chapters are checked at once."""
    if not entries:
        return []
    sources = source_chapters_for_folder(folder_path, progress_path)
    wanted = None if selected_files is None else {os.path.basename(f) for f in selected_files}
    names = [n for n in sorted(sources) if wanted is None or n in wanted]
    total = len(names)
    done = [0]
    done_lock = threading.Lock()

    def check(output_file):
        result = None
        if stop is None or not stop():
            path = os.path.join(folder_path, output_file)
            if os.path.isfile(path):
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    markup = f.read()
                result = check_chapter(sources[output_file], markup, entries, output_file, ner=ner)
        if progress is not None:
            with done_lock:
                done[0] += 1
                count = done[0]
            progress(count - 1, total)
        return result

    workers = chapter_workers(ner) if workers is None else max(1, int(workers))
    if workers > 1 and total > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=min(workers, total), thread_name_prefix="name-check") as pool:
            checked = list(pool.map(check, names))
    else:
        checked = []
        for name in names:
            if stop is not None and stop():
                break
            checked.append(check(name))
    return [r for r in checked if r is not None and r.reportable]


def find_glossary_for_folder(folder_path, source_path=None):
    """The glossary translation used for this output folder (Emergency Glossary Compliance's lookup)."""
    try:
        from TransateKRtoEN import find_glossary_file
        path = find_glossary_file(folder_path, source_path=source_path)
        if path:
            return path
    except Exception:
        pass
    fallback = os.path.join(folder_path, "glossary.csv")
    return fallback if os.path.isfile(fallback) else None


NO_GLOSSARY = "No glossary was found for this output folder."
NO_ENTRIES = ("No glossary entries match the Emergency Glossary Compliance entry types and the name "
              "check's minimum raw entry length.")
NO_SOURCES = ("This folder has no translation_progress.json / chapters_full.json, so translated "
              "chapters cannot be matched to their source chapters, and no translated_headers.txt, "
              "TOC.txt or metadata.json.")
ALL_MATCH = "Every glossary name in this folder matches the glossary."


@dataclass
class FolderReview:
    """What "Check glossary names in existing output" shows for one output folder."""
    folder: str
    glossary_path: str = ""
    chapters: list = field(default_factory=list)  # [ChapterResult] with reportable findings
    problem: str = ""  # why nothing could be checked (NO_GLOSSARY / NO_ENTRIES / NO_SOURCES)


def review_folder(folder_path, scope, *, stop=None, ner=None, progress=None, workers=None, settings=None):
    """Scan one output folder for the review dialog (desktop QA Scanner, Glossarion Mobile): its
    chapters, then the chapter titles, TOC and book metadata (``check_titles``)."""
    review = FolderReview(folder=folder_path)
    glossary_path = find_glossary_for_folder(folder_path)
    if not glossary_path:
        review.problem = NO_GLOSSARY
        return review
    review.glossary_path = glossary_path
    entries = load_entries(glossary_path, settings=settings, **scope)
    if not entries:
        review.problem = NO_ENTRIES
        return review
    has_titles = any(os.path.isfile(os.path.join(folder_path, name)) for name in TITLE_FILES)
    if not source_chapters_for_folder(folder_path) and not has_titles:
        review.problem = NO_SOURCES
        return review
    review.chapters = scan_folder(folder_path, entries, stop=stop, ner=ner, progress=progress, workers=workers)
    if not (stop is not None and stop()):
        review.chapters += check_titles(folder_path, entries, ner=ner)
    return review


def fix_files(folder_path, plan, *, backup_dir=None):
    """Apply reviewed fixes: ``plan`` maps output file -> [Finding]. Backs up each file first."""
    backup_dir = backup_dir or os.path.join(folder_path, "glossary_name_check_backup")
    changed = {}
    for output_file, findings in plan.items():
        path = os.path.join(folder_path, output_file)
        if not findings or not os.path.isfile(path):
            continue
        if output_file in TITLE_FILES:
            rows = fix_title_file(folder_path, output_file, _reviewed_findings(findings), reviewed=True,
                                  backup_dir=backup_dir)
            if rows:
                changed[output_file] = sum(n for _row, _new, _f, n in rows.values())
                mark_fixed(folder_path, {output_file: findings})
            continue
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            markup = f.read()
        new_markup, replaced = apply_fixes(markup, findings, reviewed=True)
        if not replaced:
            continue
        _backup(path, backup_dir)
        _write_text(path, new_markup)
        changed[output_file] = replaced
    mark_fixed(folder_path, {name: plan[name] for name in changed})
    return changed


def _backup(path, backup_dir):
    """Copy ``path`` into ``backup_dir`` once (the first, unfixed version is kept)."""
    import shutil
    if not backup_dir:
        return
    os.makedirs(backup_dir, exist_ok=True)
    backup = os.path.join(backup_dir, os.path.basename(path))
    if not os.path.exists(backup):
        shutil.copy2(path, backup)


# ---- chapter titles, TOC and book metadata ---------------------------------------------------------
# Chapter titles are translated in their own request (translated_headers.txt, written into each
# chapter's headers), the source table of contents in another (TOC.txt) and the book's title,
# description... in a third (metadata.json, the source text kept as ``original_<field>``). Each row
# is checked like a chapter: its source text against its translation.

HEADERS_FILE = "translated_headers.txt"
TOC_FILE = "TOC.txt"
METADATA_FILE = "metadata.json"
TITLE_FILES = (HEADERS_FILE, TOC_FILE, METADATA_FILE)

_ROW_NUMBER = re.compile(r"^\s*Chapter\s+(\d+)\s*:\s*\r?$")
_ROW_FIELD = re.compile(r"^(\s*)(Original|Translated|Output File|Target URI):(\s*)(.*?)(\r?)$")


@dataclass
class TitleRow:
    key: str
    original: str
    translated: str
    line: int = -1
    output_ref: str = ""


def title_rows(text):
    """``[TitleRow]`` of a translated_headers.txt / TOC.txt (``Chapter N:`` blocks)."""
    rows, current = [], None
    for i, line in enumerate(text.split("\n")):
        number = _ROW_NUMBER.match(line)
        if number:
            current = TitleRow(number.group(1), "", "")
            rows.append(current)
            continue
        found = _ROW_FIELD.match(line)
        if current is None or not found:
            continue
        label, value = found.group(2), found.group(4)
        if label == "Original":
            current.original = value
        elif label == "Translated":
            current.translated, current.line = value, i
        else:
            current.output_ref = value
    return [r for r in rows if r.original and r.translated and r.line >= 0]


def metadata_rows(metadata):
    """``[TitleRow]`` of the translated metadata fields (``title`` with ``original_title``...)."""
    rows = []
    for key, value in (metadata or {}).items():
        original = metadata.get(f"original_{key}")
        if str(key).startswith("original_") or not isinstance(value, str) or not isinstance(original, str):
            continue
        if value.strip() and original.strip() and value != original:
            rows.append(TitleRow(str(key), original, value))
    return rows


def _row_markup(row):
    return html_lib.escape(row.translated, quote=False)


def check_title_rows(rows, entries, label, *, ner=None):
    """One ``ChapterResult`` (``output_file`` = ``label``) for rows of one file, equal findings merged."""
    result = ChapterResult(output_file=label)
    merged = {}
    for row in rows:
        for found in check_chapter(row.original, _row_markup(row), entries, label, ner=ner, title=True).findings:
            key = (found.raw, found.expected, found.kind, found.variants)
            if key in merged:
                merged[key].occurrences += found.occurrences
            else:
                merged[key] = found
                result.findings.append(found)
    return result


def _read_title_rows(folder_path, name):
    path = os.path.join(folder_path, name)
    if not os.path.isfile(path):
        return None, []
    with open(path, "r", encoding="utf-8", errors="ignore", newline="") as f:  # keeps CRLF files CRLF
        text = f.read()
    if name != METADATA_FILE:
        return text, title_rows(text)
    try:
        data = json.loads(text)
    except ValueError:
        return None, []
    return data, (metadata_rows(data) if isinstance(data, dict) else [])


def check_titles(folder_path, entries, *, ner=None):
    """``[ChapterResult]`` with reportable findings for translated_headers.txt, TOC.txt and
    metadata.json (``output_file`` is the file name)."""
    results = []
    if not entries:
        return results
    for name in TITLE_FILES:
        _content, rows = _read_title_rows(folder_path, name)
        if rows:
            result = check_title_rows(rows, entries, name, ner=ner)
            if result.reportable:
                results.append(result)
    return results


def fix_title_rows(rows, findings_for, *, reviewed=False):
    """Fix row by row; ``findings_for(row)`` gives that row's findings. Returns
    ``{row.key: (row, new translation, findings, replaced)}`` for the rows that changed."""
    changed = {}
    for row in rows:
        findings = findings_for(row)
        if not findings:
            continue
        markup, replaced = apply_fixes(_row_markup(row), findings, reviewed=reviewed)
        if replaced:
            changed[row.key] = (row, html_lib.unescape(markup), findings, replaced)
    return changed


def _auto_findings(entries, ner):
    return lambda row: check_chapter(row.original, _row_markup(row), entries, ner=ner, title=True).fixable


def _reviewed_findings(findings):
    # a finding came from rows whose source has its raw name; other rows keep their translation
    return lambda row: [f for f in findings if f.raw in row.original]


def _chapter_html(folder_path, output_ref):
    """The output HTML file a translated_headers.txt row was written into."""
    stem = os.path.splitext(os.path.basename(str(output_ref or "").split("#")[0]))[0]
    if not stem:
        return None
    for name in (stem + ".html", "response_" + stem + ".html", stem + ".xhtml", "response_" + stem + ".xhtml"):
        path = os.path.join(folder_path, name)
        if os.path.isfile(path):
            return path
    return None


def fix_title_file(folder_path, name, findings_for, *, reviewed=False, backup_dir=None):
    """Fix one of ``TITLE_FILES`` in place (``fix_title_rows``); the translated_headers.txt titles
    are also fixed in the chapter HTML files they were written into. Returns the changed rows."""
    content, rows = _read_title_rows(folder_path, name)
    changed = fix_title_rows(rows, findings_for, reviewed=reviewed)
    if not changed:
        return changed
    path = os.path.join(folder_path, name)
    _backup(path, backup_dir)
    if name == METADATA_FILE:
        for key, (_row, new, _findings, _n) in changed.items():
            content[key] = new
        _write_text(path, json.dumps(content, ensure_ascii=False, indent=2))
        return changed
    lines = content.split("\n")
    for row, new, _findings, _n in changed.values():
        found = _ROW_FIELD.match(lines[row.line])
        lines[row.line] = f"{found.group(1)}Translated:{found.group(3)}{new}{found.group(5)}"
    _write_text(path, "\n".join(lines))
    if name == HEADERS_FILE:
        for row, _new, findings, _n in changed.values():
            html_path = _chapter_html(folder_path, row.output_ref)
            if not html_path:
                continue
            with open(html_path, "r", encoding="utf-8", errors="ignore") as f:
                markup = f.read()
            new_markup, replaced = apply_fixes(markup, findings, reviewed=reviewed)
            if replaced:
                _backup(html_path, backup_dir)
                _write_text(html_path, new_markup)
    return changed


def fix_metadata(metadata, entries, *, auto_fix, ner=None, record_dir=None):
    """Check (and with ``auto_fix`` fix) a metadata dict in memory. Returns ``(result, changed rows)``.
    ``record_dir``: keep the findings in that book's report."""
    rows = metadata_rows(metadata)
    result = check_title_rows(rows, entries, METADATA_FILE, ner=ner)
    if record_dir:
        record_findings(record_dir, METADATA_FILE, result.reportable,
                        "\n".join(html_lib.escape(r.translated, quote=False) for r in rows),
                        fixed=bool(auto_fix and result.fixable))
    changed = fix_title_rows(rows, _auto_findings(entries, ner)) if auto_fix and result.fixable else {}
    for key, (_row, new, _findings, _n) in changed.items():
        metadata[key] = new
    return result, changed


def check_and_fix_titles(folder_path, entries, *, auto_fix, ner=None, names=(HEADERS_FILE, TOC_FILE),
                         record=False):
    """The translation pipeline's title check: ``[(ChapterResult, fixed)]`` per file with findings.
    With ``auto_fix`` the automatic-fix cases are rewritten (``fix_title_file``); ``record`` keeps
    the findings in the book's report."""
    out = []
    for name in names:
        _content, rows = _read_title_rows(folder_path, name)
        if not rows:
            continue
        result = check_title_rows(rows, entries, name, ner=ner)
        if record:
            record_findings(folder_path, name, result.reportable,
                            "\n".join(html_lib.escape(r.translated, quote=False) for r in rows),
                            fixed=bool(auto_fix and result.fixable))
        if not result.reportable:
            continue
        fixed = {}
        if auto_fix and result.fixable:
            fixed = fix_title_file(folder_path, name, _auto_findings(entries, ner))
        out.append((result, fixed))
    return out


def log_title_results(results, log, *, auto_fix):
    """Log ``check_and_fix_titles`` / ``fix_metadata`` results like the chapter check does."""
    for result, fixed in results:
        for finding in result.reportable:
            done = auto_fix and finding.fixable and fixed
            log(f"📖 Glossary name check: {result.output_file}: {finding.describe()}{' (fixed)' if done else ''}")


# ---- the book's report: glossary_name_check/report.html ------------------------------------------
# Every check of a translated file (chapters as they are saved, chapter titles / TOC / metadata at
# EPUB build) replaces that file's record in glossary_name_check/findings.json; report.html is
# rewritten from it, so it always shows the book's current state: what was fixed and what is left to
# review (Review first, or what Auto-fix found too uncertain). Fixes applied later in "Check glossary
# names in existing output…" mark their rows as fixed.

REPORT_DIR = "glossary_name_check"
RECORDS_FILE = "findings.json"
REPORT_FILE = "report.html"
STATUS_FIXED = "fixed"
STATUS_FIXED_REVIEWED = "fixed (reviewed)"
STATUS_REVIEW = "to review"
_records_lock = threading.Lock()


def _records_path(output_dir):
    return os.path.join(output_dir, REPORT_DIR, RECORDS_FILE)


def _load_records(output_dir):
    try:
        with open(_records_path(output_dir), "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) and isinstance(data.get("files"), dict) else {"files": {}}
    except (OSError, ValueError):
        return {"files": {}}


def _save_records(output_dir, data):
    os.makedirs(os.path.join(output_dir, REPORT_DIR), exist_ok=True)
    _write_text(_records_path(output_dir), json.dumps(data, ensure_ascii=False, indent=1))
    write_report(output_dir, data)


def record_findings(output_dir, output_file, findings, markup, *, fixed):
    """Replace ``output_file``'s record with ``findings`` (with context from ``markup``, the file as
    checked, before any fix); ``fixed``: the automatic-fix cases were applied."""
    rows = []
    for finding in findings:
        rows.append({
            "raw": finding.raw, "expected": finding.expected, "variants": list(finding.variants),
            "occurrences": finding.occurrences, "entry_type": finding.entry_type, "origin": finding.origin,
            "kind": finding.kind,
            "status": STATUS_FIXED if fixed and finding.fixable else STATUS_REVIEW,
            "snippets": [list(snip) for snip in finding_snippets(markup, finding, width=70, limit=3)],
        })
    with _records_lock:
        data = _load_records(output_dir)
        if not rows and output_file not in data["files"]:
            return
        if rows:
            data["files"][output_file] = rows
        else:
            data["files"].pop(output_file, None)
        _save_records(output_dir, data)


def mark_fixed(output_dir, plan):
    """Fixes applied from the review (``plan``: file -> findings) count as fixed in the report."""
    with _records_lock:
        data = _load_records(output_dir)
        touched = False
        for output_file, findings in plan.items():
            keys = {(f.raw, tuple(f.variants)) for f in findings}
            for row in data["files"].get(output_file, ()):
                if (row.get("raw"), tuple(row.get("variants") or ())) in keys and row.get("status") == STATUS_REVIEW:
                    row["status"] = STATUS_FIXED_REVIEWED
                    touched = True
        if touched:
            _save_records(output_dir, data)


def report_counts(output_dir, data=None):
    """``{"files", "names", "fixed", "review"}`` over the book's record."""
    data = _load_records(output_dir) if data is None else data
    rows = [row for file_rows in data["files"].values() for row in file_rows]
    review = sum(1 for row in rows if row.get("status") == STATUS_REVIEW)
    return {"files": len(data["files"]), "names": len(rows), "fixed": len(rows) - review, "review": review}


def report_path(output_dir):
    path = os.path.join(output_dir, REPORT_DIR, REPORT_FILE)
    return path if os.path.isfile(path) else None


def write_report(output_dir, data=None):
    """Write glossary_name_check/report.html from the record; returns its path."""
    data = _load_records(output_dir) if data is None else data
    counts = report_counts(output_dir, data)
    esc = html_lib.escape

    def context(snippets):
        return "".join(f"<div class='ctx'>{esc(b)}<mark>{esc(f)}</mark>{esc(a)}</div>"
                       for b, f, a in snippets) or "<div class='ctx dim'>—</div>"

    body = []
    for output_file in sorted(data["files"], key=lambda n: (n in TITLE_FILES, n)):
        for row in data["files"][output_file]:
            status = row.get("status") or STATUS_REVIEW
            css = "review" if status == STATUS_REVIEW else "fixed"
            origin = row.get("origin") or "Book glossary"
            body.append(
                f"<tr class='{css}'><td>{esc(output_file)}</td>"
                f"<td><b>{esc(' / '.join(row.get('variants') or ()))}</b> → <b>{esc(row.get('expected', ''))}</b>"
                f"<div class='dim'>{esc(row.get('raw', ''))} · {esc(row.get('entry_type', ''))} · {esc(origin)}</div></td>"
                f"<td class='num'>{int(row.get('occurrences') or 0)}</td>"
                f"<td><span class='tag {css}'>{esc(status)}</span></td>"
                f"<td>{context(row.get('snippets') or ())}</td></tr>")
    title = os.path.basename(os.path.normpath(output_dir))
    page = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Glossary name check — {esc(title)}</title>
<style>
body {{ font-family: Segoe UI, Arial, sans-serif; background: #1e1e1e; color: #ddd; margin: 24px; }}
h1 {{ font-size: 20px; margin: 0 0 6px; }} .summary {{ margin: 0 0 18px; color: #aaa; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border-bottom: 1px solid #333; padding: 8px 10px; text-align: left; vertical-align: top; }}
th {{ background: #2a2a2a; position: sticky; top: 0; }}
.num {{ text-align: right; }} .dim {{ color: #888; font-size: 12px; margin-top: 2px; }}
.ctx {{ font-size: 13px; margin-bottom: 4px; }} mark {{ background: #6b4e16; color: #fff; padding: 0 2px; }}
.tag {{ padding: 2px 8px; border-radius: 10px; font-size: 12px; white-space: nowrap; }}
.tag.fixed {{ background: #1f4d2b; color: #9be3ae; }} .tag.review {{ background: #5a3d12; color: #ffd28a; }}
</style></head><body>
<h1>Glossary name check — {esc(title)}</h1>
<p class="summary">{counts['names']} name(s) in {counts['files']} file(s) differ from the glossary:
{counts['fixed']} fixed, {counts['review']} to review. Fix the rest with QA Scanner settings →
<i>Check glossary names in existing output…</i></p>
<table><tr><th>File</th><th>Found → glossary name</th><th>Times</th><th>Status</th><th>Context (as translated)</th></tr>
{''.join(body)}
</table></body></html>
"""
    os.makedirs(os.path.join(output_dir, REPORT_DIR), exist_ok=True)
    path = os.path.join(output_dir, REPORT_DIR, REPORT_FILE)
    _write_text(path, page)
    return path


def summary_line(output_dir):
    """The end-of-run log line, or None when the book has no record."""
    if not os.path.isfile(_records_path(output_dir)):
        return None
    counts = report_counts(output_dir)
    path = report_path(output_dir) or write_report(output_dir)
    if not counts["names"]:
        return f"📖 Glossary name check: every glossary name matches — report: {path}"
    return (f"📖 Glossary name check: {counts['names']} name(s) in {counts['files']} file(s) differ from the "
            f"glossary: {counts['fixed']} fixed, {counts['review']} to review — report: {path}")
