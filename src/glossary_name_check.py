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
KIND_MISSING = "missing"


@dataclass(frozen=True)
class NameEntry:
    raw: str
    translated: str
    entry_type: str


@dataclass
class Finding:
    raw: str
    expected: str
    entry_type: str
    kind: str
    variants: tuple = ()
    occurrences: int = 0

    @property
    def fixable(self):
        """The automatic-fix case: one close spelling of a character name. A raw name shorter than
        ``SHORT_RAW`` characters (surnames like 박 / 강, words like 하나) also matches unrelated source
        text, and its fuzzy matches were wrong on the corpus ("Sparks" for Park, "Hanja" for Hana),
        so only a spacing / hyphen respelling of it ("Soyoon" for So-yoon) is fixed automatically."""
        if self.kind != KIND_VARIANT or self.entry_type != "character":
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

    def __init__(self, entries, known_names):
        self.entries = tuple(entries)
        self.known = frozenset(known_names)

    def __bool__(self):
        return bool(self.entries)


def build_entries(glossary_entries, *, mode="characters", custom_types=(), min_chars=0):
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
        picked.setdefault(raw, NameEntry(raw, translated, entry_type))
    entries = sorted(picked.values(), key=lambda e: -len(e.raw))
    return GlossaryNameEntries(entries, known)


_load_lock = threading.Lock()
_load_cache = {}


def load_entries(glossary_path, *, mode="characters", custom_types=(), min_chars=0):
    """Parse a glossary file once per (file version, options)."""
    if not glossary_path or not os.path.isfile(glossary_path):
        return GlossaryNameEntries((), ())
    info = os.stat(glossary_path)
    key = (os.path.normcase(os.path.abspath(glossary_path)), info.st_size, info.st_mtime_ns,
           str(mode or "characters").lower(), tuple(custom_types or ()), int(min_chars or 0))
    with _load_lock:
        cached = _load_cache.get(key)
        if cached is not None:
            return cached
    import glossary_usage
    entries = build_entries(glossary_usage.parse_glossary_file(glossary_path), mode=mode,
                            custom_types=custom_types, min_chars=min_chars)
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


def check_chapter(source_text, output_markup, entries, output_file="", *, ner=None):
    """Compare one translated chapter with its source chapter. ``ner``: a ``ner_names.NerTagger``
    finds the candidate spellings (else the capitalization heuristic)."""
    result = ChapterResult(output_file=output_file)
    if not entries or not source_text or not output_markup:
        return result
    output_text = html_to_text(output_markup)
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
                                           tuple(sorted(variants)), occurrences))
        else:
            result.findings.append(Finding(entry.raw, entry.translated, entry.entry_type, KIND_MISSING))
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


def check_and_fix_file(output_path, source_text, entries, *, auto_fix, ner=None):
    """Check one output file; with ``auto_fix`` rewrite it with the fixable variants replaced."""
    with open(output_path, "r", encoding="utf-8", errors="ignore") as f:
        markup = f.read()
    result = check_chapter(source_text, markup, entries, os.path.basename(output_path), ner=ner)
    replaced = 0
    if auto_fix and result.fixable:
        new_markup, replaced = apply_fixes(markup, result.fixable)
        if replaced:
            _write_text(output_path, new_markup)
    return result, replaced


def _write_text(path, text):
    tmp = path + ".namecheck.tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        f.write(text)
    os.replace(tmp, path)


def settings_from_env(env=None):
    """Entry types shared with Emergency Glossary Compliance; the name check's own minimum length."""
    env = os.environ if env is None else env
    mode = str(env.get("EMERGENCY_GLOSSARY_COMPLIANCE_MODE", "characters") or "characters").lower()
    try:
        custom_types = json.loads(env.get("EMERGENCY_GLOSSARY_COMPLIANCE_CUSTOM_TYPES", "[]") or "[]")
    except (TypeError, ValueError):
        custom_types = []
    try:
        min_chars = max(0, min(10, int(env.get("GLOSSARY_NAME_CHECK_MIN_CHARS", 0))))
    except (TypeError, ValueError):
        min_chars = 0
    return {"mode": mode, "custom_types": tuple(custom_types if mode == "custom" else ()),
            "min_chars": min_chars}


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


def resolve_ner(use_ner, log=None):
    """The NER tagger when it is switched on and this build can run it (downloading the model on first
    use); None means the capitalization heuristic. Each fallback reason is logged once per process."""
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
    try:
        return ner_names.get_tagger(log)
    except Exception as exc:
        if log and "failed" not in _ner_notes:
            _ner_notes.add("failed")
            log(f"⚠️ Glossary name check: the NER model could not be loaded ({exc}); using the capitalization heuristic")
        return None


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


def scan_folder(folder_path, entries, *, progress_path=None, selected_files=None, stop=None, ner=None):
    """Check every mapped output file; returns ``[ChapterResult]`` with reportable findings."""
    results = []
    if not entries:
        return results
    sources = source_chapters_for_folder(folder_path, progress_path)
    wanted = None if selected_files is None else {os.path.basename(f) for f in selected_files}
    for output_file in sorted(sources):
        if stop is not None and stop():
            break
        if wanted is not None and output_file not in wanted:
            continue
        path = os.path.join(folder_path, output_file)
        if not os.path.isfile(path):
            continue
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            markup = f.read()
        result = check_chapter(sources[output_file], markup, entries, output_file, ner=ner)
        if result.reportable:
            results.append(result)
    return results


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
              "chapters cannot be matched to their source chapters.")
ALL_MATCH = "Every glossary name in this folder matches the glossary."


@dataclass
class FolderReview:
    """What "Fix glossary names in existing chapters" shows for one output folder."""
    folder: str
    glossary_path: str = ""
    chapters: list = field(default_factory=list)  # [ChapterResult] with reportable findings
    problem: str = ""  # why nothing could be checked (NO_GLOSSARY / NO_ENTRIES / NO_SOURCES)


def review_folder(folder_path, scope, *, stop=None, ner=None):
    """Scan one output folder for the review dialog (desktop QA Scanner, Glossarion Mobile)."""
    review = FolderReview(folder=folder_path)
    glossary_path = find_glossary_for_folder(folder_path)
    if not glossary_path:
        review.problem = NO_GLOSSARY
        return review
    review.glossary_path = glossary_path
    entries = load_entries(glossary_path, **scope)
    if not entries:
        review.problem = NO_ENTRIES
        return review
    if not source_chapters_for_folder(folder_path):
        review.problem = NO_SOURCES
        return review
    review.chapters = scan_folder(folder_path, entries, stop=stop, ner=ner)
    return review


def fix_files(folder_path, plan, *, backup_dir=None):
    """Apply reviewed fixes: ``plan`` maps output file -> [Finding]. Backs up each file first."""
    import shutil
    backup_dir = backup_dir or os.path.join(folder_path, "glossary_name_check_backup")
    changed = {}
    for output_file, findings in plan.items():
        path = os.path.join(folder_path, output_file)
        if not findings or not os.path.isfile(path):
            continue
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            markup = f.read()
        new_markup, replaced = apply_fixes(markup, findings, reviewed=True)
        if not replaced:
            continue
        os.makedirs(backup_dir, exist_ok=True)
        backup = os.path.join(backup_dir, output_file)
        if not os.path.exists(backup):
            shutil.copy2(path, backup)
        _write_text(path, new_markup)
        changed[output_file] = replaced
    return changed
