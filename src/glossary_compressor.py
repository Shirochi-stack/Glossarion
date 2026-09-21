# -*- coding: utf-8 -*-
"""
Glossary Compressor Module
Filters glossary entries based on source text to reduce token usage.

Supports:
  - Token-efficient CSV format (=== SECTION === headers)
  - Legacy CSV / Unit-Separator-delimited format
  - JSON dict and list formats
  - Fallback: any text format (.md, .txt, etc.) via raw-name scanning
"""

import os
import re
import json
import csv
import time
import threading
from contextvars import ContextVar
from io import StringIO
from gender_tracking import (
    BINARY_GENDERS,
    automatic_chapter_gender as _shared_automatic_chapter_gender,
    effective_gender as _shared_effective_gender,
    normalize_bias as _shared_normalize_bias,
    normalize_gender as _shared_normalize_gender,
    normalized_decision as _shared_normalized_decision,
    rare_genders as _shared_rare_genders,
    resolved_storage_gender as _shared_resolved_storage_gender,
    rarity_stats as _shared_rarity_stats,
    tracker_entry_for_raw as _shared_tracker_entry_for_raw,
    tracker_path_for_glossary as _shared_tracker_path_for_glossary,
)
from glossary_matching import (
    TIER_HONORIFIC,
    TIER_WEAK,
    MatchConfig,
    allowlist_path_for,
    is_gender_entry_type,
    legacy_text_contains_term,
    match_term,
    normalize_override_terms,
    prepare_source_text,
    strict_name_config,
    strict_name_in_text,
)

# Serialize glossary compression across translation worker threads.
# Compression is pure-Python (terms x chapter-text substring scans); when
# several chunk workers run it concurrently they monopolize the GIL and the
# GUI thread stalls for seconds (verified via the freeze watchdog in frozen
# builds). One compression at a time keeps the GUI responsive and costs
# almost nothing overall since API latency dominates per-chunk time.
_COMPRESS_GIL_LOCK = threading.Lock()


class _NullLock:
    """Stand-in for _COMPRESS_GIL_LOCK when serialization is switched off."""

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


_NULL_LOCK = _NullLock()


class GlossaryCompressionCancelled(Exception):
    """An immediate Stop arrived while a glossary was being compressed."""


def _hard_stop_requested():
    """An immediate (non-graceful) Stop is in effect.

    TRANSLATION_CANCELLED is only set for an immediate stop. A graceful stop
    lets in-flight requests finish, and those still need their glossary.
    """
    return os.environ.get("TRANSLATION_CANCELLED") == "1"


def _gil_yield(counter, every=64):
    """Briefly release the GIL every `every` iterations of a hot loop.

    Also the cancellation point. A cross-novel unified glossary is tens of
    thousands of entries, so one compression runs for seconds; without a way
    out, every queued worker still runs its full compression after Stop, one
    after another under _COMPRESS_GIL_LOCK, and the GUI thread is starved for
    the whole tail.
    """
    if counter % every == 0:
        if _hard_stop_requested():
            raise GlossaryCompressionCancelled()
        time.sleep(0)

try:
    from extract_glossary_from_epub import get_custom_entry_types as _get_custom_entry_types
except ImportError:
    _get_custom_entry_types = None

_gender_bias_log_seen = set()
_ACTIVE_GLOSSARY_SETTINGS = ContextVar(
    "glossary_compressor_settings", default=None
)
# The matcher context for the compression currently running on this thread.
# A ContextVar rather than a threaded parameter for the same reason the
# settings snapshot above is one: the alternative is adding an argument to
# eight functions that only pass it through. ContextVars are per-thread, so
# concurrent chunk workers each see their own chapter.
_ACTIVE_MATCH_CONTEXT = ContextVar(
    "glossary_compressor_match_context", default=None
)


def _setting(name, default=None):
    settings = _ACTIVE_GLOSSARY_SETTINGS.get()
    if isinstance(settings, dict) and name in settings:
        value = settings.get(name)
        return default if value is None else value
    return os.getenv(name, default)


def _should_use_raw_name_fallback():
    """Return True if raw-name scan fallback is appropriate.

    The raw-name scan is a brute-force approach that only makes sense for
    user-supplied (manual) glossaries whose format may be arbitrary.  For
    auto-generated glossaries (Full, Balanced, Minimal, Single Pass, etc.)
    the structured CSV/JSON parsing is authoritative and a 0-match result
    simply means no entries are relevant to this chapter.

    Allowed modes: 'off_no_automap' (Manual Glossary Only), 'off', 'off_fuzzy'.
    """
    mode = str(_setting('AUTO_GLOSSARY_MODE', 'off') or 'off').lower().strip()
    return mode in ('off', 'off_no_automap', 'off_fuzzy')


def _get_gender_types():
    """Return a set of entry type names that have has_gender enabled."""
    if _get_custom_entry_types:
        try:
            types = _get_custom_entry_types()
            return {t for t, cfg in types.items()
                    if cfg.get('enabled', True) and cfg.get('has_gender', False)}
        except Exception:
            pass
    return {'character'}  # safe fallback


def _strict_gender_name_matching_enabled():
    """Return True when gender-enabled entries must match the full raw name."""
    return str(_setting("COMPRESS_GLOSSARY_STRICT_GENDER_MATCHING", "0")).strip().lower() in (
        "1", "true", "yes", "on"
    )


def _consider_translated_column_enabled():
    """Return True when glossary compression may match translated_name too."""
    return str(_setting("COMPRESS_GLOSSARY_CONSIDER_TRANSLATED_COLUMN", "0")).strip().lower() in (
        "1", "true", "yes", "on"
    )

_ALLOWLIST_CACHE = {}
_ALLOWLIST_CACHE_LOCK = threading.Lock()


def _load_match_allowlist(glossary_path):
    """Reviewed overrides for this glossary, as (always_keep, always_drop).

    Written by `python tools/glossary_match_report.py --apply-verdicts` after
    you fill in verdicts.csv. This is the escape hatch for terms no heuristic
    will get right — a Korean noun inside a derived compound, say — so a
    decision you have already made by hand is never re-litigated by the
    matcher on the next chapter.

    Cached on (path, mtime) so editing the file takes effect without a
    restart, while a normal run reads it once.
    """
    path = allowlist_path_for(glossary_path)
    if not path or not os.path.isfile(path):
        return frozenset(), frozenset()
    try:
        stamp = os.path.getmtime(path)
    except OSError:
        return frozenset(), frozenset()
    key = (path, stamp)
    with _ALLOWLIST_CACHE_LOCK:
        cached = _ALLOWLIST_CACHE.get(key)
    if cached is not None:
        return cached
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        value = (
            normalize_override_terms(data.get("always_keep")),
            normalize_override_terms(data.get("always_drop")),
        )
    except (OSError, ValueError) as exc:
        print(f"⚠️ Glossary compression: could not read match overrides: {exc}")
        value = (frozenset(), frozenset())
    with _ALLOWLIST_CACHE_LOCK:
        _ALLOWLIST_CACHE[key] = value
    return value


def _serialize_compression_enabled():
    """Whether to hold _COMPRESS_GIL_LOCK. Defaults on."""
    return str(_setting("GLOSSARY_COMPRESS_SERIALIZE", "1")).strip().lower() in (
        "1", "true", "yes", "on"
    )


def _match_engine():
    """Which matcher decides: 'legacy' (default), 'shadow', or 'new'."""
    value = str(_setting("GLOSSARY_MATCH_ENGINE", "legacy") or "legacy").strip().lower()
    return value if value in ("legacy", "shadow", "new") else "legacy"


def _is_unified_glossary_path(path):
    stem = os.path.splitext(os.path.basename(str(path or "")))[0]
    return stem.lower() == "glossary_unified"


def _unified_whole_term_enabled():
    return str(_setting("UNIFIED_GLOSSARY_WHOLE_TERM_MATCHING", "1")).strip().lower() in (
        "1", "true", "yes", "on"
    )


def _unified_min_term_length():
    try:
        return max(1, int(str(_setting("UNIFIED_GLOSSARY_MIN_TERM_LENGTH", "2")).strip()))
    except (TypeError, ValueError):
        return 2


class _MatchContext:
    """Per-chapter matcher state: engine choice, config, index, recorder.

    Built once per compress_glossary() call. The prepared index is the reason
    this exists at all — without it every term would re-normalize the whole
    chapter, which is what makes the tiered matcher affordable.

    ``whole_term_only`` is set for the cross-novel unified glossary. Keeping a
    multi-word entry because one word of it appears is a recall safeguard for
    a single book's few hundred names (surname / given name). Across tens of
    thousands of entries from other novels it is the opposite: half of them
    are multi-word, and their parts are ordinary nouns (길드, 제국, 마법 …), so
    the token rule alone keeps over a thousand unrelated entries per chapter.
    There the whole term has to be present.
    """

    __slots__ = ("engine", "cfg", "prepared", "recorder", "source_text",
                 "glossary_path", "chapter_ref", "always_keep", "always_drop",
                 "whole_term_only", "min_term_length",
                 "strict_gender", "_strict_cfg", "_strict_prepared")

    def __init__(self, source_text, glossary_path=None, chapter_ref=None):
        self.engine = _match_engine()
        self.source_text = source_text or ""
        self.glossary_path = glossary_path
        self.chapter_ref = chapter_ref
        self.cfg = None
        self.prepared = None
        self.recorder = None
        self.always_keep = frozenset()
        self.always_drop = frozenset()
        self.whole_term_only = (
            _is_unified_glossary_path(glossary_path) and _unified_whole_term_enabled()
        )
        self.min_term_length = _unified_min_term_length() if self.whole_term_only else 1
        self.strict_gender = _strict_gender_name_matching_enabled()
        self._strict_cfg = None
        self._strict_prepared = None
        if self.engine != "legacy":
            self.cfg = MatchConfig.from_getter(_setting)
            if self.whole_term_only:
                # Exact / normalized / despaced / honorific-stripped forms of
                # the whole term still count; a part of it never does.
                self.cfg.min_tier = max(self.cfg.min_tier, TIER_HONORIFIC)
                self.cfg.allow_weak_token = False
                # 고려하면 is not Goryeo: verb endings and noun affixes are
                # homonym traps once the terms come from other novels.
                self.cfg.derived_forms = False
            self.prepared = prepare_source_text(self.source_text, self.cfg)
            # Overrides apply to the tiered verdict only: the legacy engine is
            # the untouched path and must stay bit-identical.
            self.always_keep, self.always_drop = _load_match_allowlist(glossary_path)
        if self.engine == "shadow":
            try:
                from glossary_match_shadow import get_recorder
                self.recorder = get_recorder(
                    _setting("GLOSSARY_MATCH_SHADOW_LOG_DIR", None)
                )
            except Exception as exc:  # pragma: no cover - logging must not break a run
                print(f"⚠️ Glossary shadow log unavailable: {exc}")
                self.engine = "legacy"

    def _strict_name_match(self, term):
        """Strict Gender Entry Precise Matching, whichever engine is on.

        Built lazily: the toggle is usually off, and under the legacy
        engine nothing else needs a prepared index.
        """
        if self._strict_cfg is None:
            base = self.cfg if self.cfg is not None else MatchConfig.from_getter(_setting)
            self._strict_cfg = strict_name_config(base)
            self._strict_prepared = (
                self.prepared if self.prepared is not None
                else prepare_source_text(self.source_text, self._strict_cfg)
            )
        return strict_name_in_text(
            self.source_text, term, self._strict_cfg, self._strict_prepared
        )

    def decide(self, term, is_character=False, *, translated_name="", entry_type=""):
        """Return whether this term counts as present in the chapter."""
        if self.whole_term_only:
            whole = str(term or "").strip()
            # A one-character entry from another novel (그, 신, 왕 …) is a
            # common word here, not that novel's character.
            if len(whole) < self.min_term_length:
                return False
            legacy = whole in self.source_text
        elif is_character and self.strict_gender:
            # The strict toggle is a precise matcher in its own right: the
            # whole name at a real word boundary. It replaces the loose
            # rule under every engine, so legacy / shadow / new agree on
            # gendered entries by construction.
            legacy = self._strict_name_match(term)
        else:
            legacy = legacy_text_contains_term(
                self.source_text, term, is_character=is_character,
                strict_gender=False,
            )
        if self.engine == "legacy":
            return legacy

        result = match_term(self.prepared, term, is_character, self.cfg)
        new_verdict = bool(result)
        override = ""
        if self.always_keep or self.always_drop:
            key = str(term or "").strip().casefold()
            if key in self.always_keep:
                new_verdict, override = True, "override_keep"
            elif key in self.always_drop:
                new_verdict, override = False, "override_drop"

        if self.engine == "new":
            return new_verdict

        # Shadow: legacy still decides, the disagreement is what we record.
        # Every decision is reported, including agreements — the recorder
        # needs them for the report's denominator and drops them itself.
        if self.recorder is not None:
            self.recorder.record(
                legacy=legacy, new=new_verdict, term=term,
                translated_name=translated_name, entry_type=entry_type,
                is_character=is_character, tier=result.tier,
                rule=override or result.rule,
                reject_reason=override or result.reject_reason,
                source_text=self.source_text,
                chapter_ref=self.chapter_ref, glossary_path=self.glossary_path,
            )
        return legacy


def _relax_on_zero_enabled():
    return str(_setting("GLOSSARY_COMPRESS_RELAX_ON_ZERO", "1")).strip().lower() in (
        "1", "true", "yes", "on"
    )


def _run_with_zero_match_relaxation(ctx, run, count_matches, label):
    """Re-run `run` at progressively looser tiers if it matches nothing.

    Tightening the matcher makes "0 entries survived" far more likely, and a
    zero-match CSV result means the chapter is sent with NO glossary at all.
    Rather than let a single strict gate cost a chapter its whole glossary,
    step the acceptance floor down and try again; a genuine no-match chapter
    still ends at zero, just by way of the loosest rule rather than the
    strictest.

    Only the tiered engine can hit the new cliff, so the legacy and shadow
    paths run exactly once and are bit-for-bit unchanged.
    """
    result = run()
    if ctx.engine != "new" or ctx.cfg is None or not _relax_on_zero_enabled():
        return result
    if getattr(ctx, "whole_term_only", False):
        # For the unified glossary "nothing from other novels is in this
        # chapter" is a normal result, not a cliff, and relaxing would switch
        # the part-of-a-name rule straight back on.
        return result
    if count_matches(result) > 0:
        return result

    original_floor = ctx.cfg.min_tier
    try:
        while ctx.cfg.min_tier > TIER_WEAK:
            ctx.cfg.min_tier -= 1
            relaxed = run()
            if count_matches(relaxed) > 0:
                print(
                    f"ℹ️ Glossary compression: 0 {label} matches at tier "
                    f"{ctx.cfg.min_tier + 1}, relaxed to tier {ctx.cfg.min_tier}"
                )
                return relaxed
    finally:
        ctx.cfg.min_tier = original_floor
    return result


def _active_match_context(source_text):
    """The context for this compression, or a legacy-only one for direct calls.

    _compress_fallback_text and the private format helpers are reachable from
    tests and from compress_glossary_file without going through the wrapper
    that installs the ContextVar, so fall back rather than requiring it.
    """
    ctx = _ACTIVE_MATCH_CONTEXT.get()
    if ctx is not None and ctx.source_text == (source_text or ""):
        return ctx
    return _MatchContext(source_text)


try:
    from GlossaryManager import GLOSSARY_SEP, _gsep, _is_glossary_header
except ImportError:
    GLOSSARY_SEP = '\x1F'
    def _gsep(text):
        return GLOSSARY_SEP if GLOSSARY_SEP in text else ','
    def _is_glossary_header(line):
        low = line.strip().lower()
        return low.startswith('type,raw_name') or low.startswith(f'type{GLOSSARY_SEP}raw_name')


# ─── Tokenization regex for fallback candidate extraction ────────────────────
# Splits on: comma, pipe, parentheses, brackets, braces, colon, semicolon,
# tab, Unit Separator (U+001F), forward slash,
# and spaced delimiters: dash, en-dash, em-dash, arrow, double-arrow, equals
_FALLBACK_SPLIT_RE = re.compile(
    r'[,|\(\)\[\]\{\}:\t;\x1F/]'    # single-char delimiters
    r'|(?:\s[-–—→⇒=]\s)'            # spaced delimiters (e.g. " - ", " → ")
)

# Patterns that mark a line as a self-contained glossary entry
_ENTRY_DELIMITERS = ['\x1F', '\t', ' = ', ' - ', ' – ', ' — ', ' → ', ' ⇒ ', ' : ']
_ENTRY_BULLET_RE = re.compile(r'^\s*(?:[*\-•]|\d+[.\)])\s')
_ENTRY_TABLE_RE = re.compile(r'^\s*\|')

# Section header patterns
_SECTION_HEADER_RE = re.compile(
    r'^\s*(?:'
    r'#{1,6}\s'           # Markdown headers: # ... ## ...
    r'|===\s.*===\s*$'    # === SECTION ===
    r'|---\s.*---\s*$'    # --- SECTION ---
    r')'
)


def _gender_tracker_path_for_glossary(glossary_path):
    return _shared_tracker_path_for_glossary(glossary_path) or None


def _load_gender_tracker(glossary_path):
    if str(_setting("GLOSSARY_SKIP_GENDER_TRACKING", "0")).strip().lower() in ("1", "true", "yes", "on"):
        return None
    tracker_path = _gender_tracker_path_for_glossary(glossary_path)
    if not tracker_path or not os.path.exists(tracker_path):
        return None
    try:
        with open(tracker_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and isinstance(data.get("entries"), dict):
            return data
    except Exception as e:
        print(f"⚠️ Glossary compression: could not load gender tracker: {e}")
    return None


def _normal_gender(value):
    return _shared_normalize_gender(value)


def _has_explicit_gender_value(value):
    return bool(str(value or "").strip())


def _chapter_ref_parts(chapter_ref):
    if isinstance(chapter_ref, dict):
        chapter_num = chapter_ref.get("chapter_num")
        chapter_file = chapter_ref.get("chapter_file")
    else:
        chapter_num = chapter_ref
        chapter_file = None
    if chapter_num is None:
        chapter_num = _setting("CURRENT_CHAPTER_NUM")
    if not chapter_file:
        chapter_file = _setting("CURRENT_CHAPTER_FILE")
    try:
        chapter_num_f = float(chapter_num)
    except Exception:
        chapter_num_f = None
    return chapter_num_f, os.path.basename(str(chapter_file or ""))


def _tracker_entry_for_raw(tracker, raw_name):
    return _shared_tracker_entry_for_raw(tracker, raw_name)


def _remember_available_gender(available_genders, raw_name, gender):
    raw_key = str(raw_name or "").strip().casefold()
    gender = _normal_gender(gender)
    if raw_key and gender not in {"", "unknown", "n/a", "na", "none", "-"}:
        available_genders.setdefault(raw_key, set()).add(gender)


def _available_gender_set(available_genders, raw_name):
    if not available_genders:
        return set()
    return set(available_genders.get(str(raw_name or "").strip().casefold(), set()))


def _gender_noise_threshold():
    try:
        value = float(_setting("GLOSSARY_GENDER_NOISE_THRESHOLD", "10"))
    except Exception:
        value = 10.0
    return max(0.0, min(100.0, value)) / 100.0


def _gender_bias():
    return _shared_normalize_bias(
        _setting("GLOSSARY_GENDER_TRACKING_BIAS", None)
    )


def _gender_rarity_stats(entry):
    return _shared_rarity_stats(entry)


def _rare_tracker_genders(entry):
    return _shared_rare_genders(entry, _gender_noise_threshold(), _gender_bias())


def _log_gender_bias_effect(entry, raw_name, actual_gender):
    bias = _gender_bias()
    actual_gender = _normal_gender(actual_gender)
    if bias != actual_gender:
        return
    threshold = _gender_noise_threshold()
    if threshold <= 0:
        return
    stats = _gender_rarity_stats(entry).get(actual_gender)
    if not stats or stats["ratio"] > threshold:
        return
    key = (str(raw_name or "").casefold(), actual_gender, int(threshold * 100))
    if key in _gender_bias_log_seen:
        return
    _gender_bias_log_seen.add(key)
    translated_name = str(entry.get("translated_name", "") or "").strip() if isinstance(entry, dict) else ""
    label = f"{raw_name} = {translated_name}" if translated_name and raw_name else (translated_name or str(raw_name or "unknown"))
    print(
        "📑 Gender tracker bias active: "
        f"keeping rare {actual_gender} variant for {label} "
        f"({stats['count']}/{stats['total']} = {stats['ratio'] * 100:.1f}%, "
        f"slider {threshold * 100:.0f}%, bias={bias})"
    )


def _gender_is_rare_noise(entry, actual_gender, raw_name=None):
    actual_gender = _normal_gender(actual_gender)
    if actual_gender in {"", "unknown", "n/a", "na", "none", "-"}:
        return False
    _log_gender_bias_effect(entry, raw_name, actual_gender)
    return actual_gender in _rare_tracker_genders(entry)


def _tracker_gender_for_entry(entry, chapter_ref=None):
    if isinstance(chapter_ref, dict) and chapter_ref.get("use_storage_gender"):
        return _shared_resolved_storage_gender(entry, "")
    return _shared_automatic_chapter_gender(
        entry,
        chapter_ref,
        _gender_noise_threshold(),
        _gender_bias(),
    )


def _gender_variant_allowed(tracker, raw_name, gender, chapter_ref=None, available_genders=None):
    actual = _normal_gender(gender)
    if not actual or actual in {"unknown", "n/a", "na", "none", "-"}:
        return True
    bias = _gender_bias()
    available = _available_gender_set(available_genders, raw_name)
    entry = _tracker_entry_for_raw(tracker, raw_name)
    decision = _shared_normalized_decision(entry)
    if entry and len(available) <= 1:
        # Consolidated glossaries have only a carrier row.  Never discard it
        # based on that stored gender; the emission step materializes the
        # chapter-aware Auto result or the manual decision.
        return True
    if decision in BINARY_GENDERS:
        return actual == decision
    if _gender_noise_threshold() >= 1.0 and bias == "none":
        return True
    if _gender_is_rare_noise(entry, actual, raw_name):
        if bias != "none" and bias not in available:
            return True
        return False
    wanted = _tracker_gender_for_entry(entry, chapter_ref)
    if not wanted:
        return True
    if wanted not in available:
        return True
    return actual == wanted


def _emitted_gender(tracker, raw_name, stored_gender, chapter_ref=None, available_genders=None):
    """Return the plain gender that should be emitted for a surviving row."""
    entry = _tracker_entry_for_raw(tracker, raw_name)
    if not entry:
        return _normal_gender(stored_gender)
    available = _available_gender_set(available_genders, raw_name)
    decision = _shared_normalized_decision(entry)
    # With old two-row Auto glossaries, filtering already selected the correct
    # row.  Preserve the 100% threshold's intentional keep-both behavior too.
    if len(available) > 1 and decision == "auto":
        return _normal_gender(stored_gender)
    return _shared_effective_gender(
        entry,
        stored_gender,
        chapter_ref,
        _gender_noise_threshold(),
        _gender_bias(),
    )


def _replace_token_gender(line, gender):
    """Replace or insert a token-efficient entry's bracketed gender."""
    gender = _normal_gender(gender)
    if gender not in BINARY_GENDERS:
        return line
    bracket = re.search(r"\s*\[[^\]]*\](?=\s*(?::|$))", line)
    if bracket:
        return f"{line[:bracket.start()]} [{gender}]{line[bracket.end():]}"

    # Insert immediately before the first top-level description colon.
    paren_depth = 0
    bracket_depth = 0
    for index, char in enumerate(line):
        if char == "(" and bracket_depth == 0:
            paren_depth += 1
        elif char == ")" and bracket_depth == 0 and paren_depth:
            paren_depth -= 1
        elif char == "[" and paren_depth == 0:
            bracket_depth += 1
        elif char == "]" and paren_depth == 0 and bracket_depth:
            bracket_depth -= 1
        elif char == ":" and paren_depth == 0 and bracket_depth == 0:
            return f"{line[:index].rstrip()} [{gender}]{line[index:]}"
    return f"{line.rstrip()} [{gender}]"


def compress_glossary(
    glossary_content,
    source_text,
    glossary_format='auto',
    glossary_path=None,
    chapter_ref=None,
    settings=None,
):
    """Run compression with an optional thread-safe request settings snapshot."""
    token = _ACTIVE_GLOSSARY_SETTINGS.set(settings)
    # Built after the settings token so the engine choice and the matcher
    # knobs come from this request's snapshot, not the ambient environment.
    match_token = _ACTIVE_MATCH_CONTEXT.set(
        _MatchContext(source_text, glossary_path=glossary_path, chapter_ref=chapter_ref)
    )
    try:
        return _compress_glossary_impl(
            glossary_content,
            source_text,
            glossary_format=glossary_format,
            glossary_path=glossary_path,
            chapter_ref=chapter_ref,
        )
    except GlossaryCompressionCancelled:
        # The request this was for is being aborted. "" is what compression
        # returns for "no matching entries", which every caller already
        # handles by appending nothing -- never the uncompressed glossary.
        return ""
    finally:
        _ACTIVE_MATCH_CONTEXT.reset(match_token)
        _ACTIVE_GLOSSARY_SETTINGS.reset(token)


def _compress_glossary_impl(glossary_content, source_text, glossary_format='auto', glossary_path=None, chapter_ref=None):
    """
    Compress glossary by excluding entries that don't appear in the source text.
    
    Args:
        glossary_content: Raw glossary content (CSV string, JSON dict/list, or plain text)
        source_text: The source text to check against
        glossary_format: 'csv', 'json', 'text', or 'auto' (detect from content)
    
    Returns:
        Compressed glossary in the same format as input
    """
    if not glossary_content or not source_text:
        return glossary_content
    
    # Auto-detect format
    if glossary_format == 'auto':
        if isinstance(glossary_content, str):
            stripped = glossary_content.strip()
            # Check if it looks like JSON
            if (stripped.startswith('{') or stripped.startswith('[')) and (stripped.endswith('}') or stripped.endswith(']')):
                glossary_format = 'json'
            else:
                # Check if it looks like CSV (has header row or Unit Separator)
                first_lines = stripped.split('\n', 5)
                has_csv_header = any(_is_glossary_header(l) for l in first_lines[:2])
                has_unit_sep = GLOSSARY_SEP in stripped[:500]
                has_section_headers = any(l.strip().startswith('===') for l in first_lines)
                has_glossary_columns = any(l.strip().lower().startswith('glossary columns:') for l in first_lines[:2])
                
                if has_csv_header or has_unit_sep or has_section_headers or has_glossary_columns:
                    glossary_format = 'csv'
                else:
                    # Not recognizable as CSV or JSON → use text fallback
                    glossary_format = 'text'
        elif isinstance(glossary_content, (dict, list)):
            glossary_format = 'json'
        else:
            return glossary_content
    
    # One compression at a time — see _COMPRESS_GIL_LOCK note above.
    # GLOSSARY_COMPRESS_SERIALIZE=0 removes the lock. Default stays on: the
    # freeze it prevents is verified, so it should only be lifted against a
    # benchmark (tools/glossary_match_bench.py measures the GUI-stall proxy
    # directly), not on the assumption that the prepared index made it cheap.
    if _hard_stop_requested():
        raise GlossaryCompressionCancelled()
    with _COMPRESS_GIL_LOCK if _serialize_compression_enabled() else _NULL_LOCK:
        # Re-check after the wait: workers queue here, and the ones still
        # waiting when Stop is pressed must not each run a full compression.
        if _hard_stop_requested():
            raise GlossaryCompressionCancelled()
        if glossary_format == 'csv':
            return _compress_csv_glossary(glossary_content, source_text, glossary_path=glossary_path, chapter_ref=chapter_ref)
        elif glossary_format == 'json':
            return _compress_json_glossary(glossary_content, source_text, glossary_path=glossary_path, chapter_ref=chapter_ref)
        elif glossary_format == 'text':
            if _should_use_raw_name_fallback():
                print("⚠️ Glossary compression: using fallback raw-name scan (unrecognized format)")
                return _compress_fallback_text(glossary_content, source_text)
            else:
                # Auto-generated glossary — skip raw-name scan, return as-is
                return glossary_content
        else:
            return glossary_content


def _compress_csv_glossary(csv_content, source_text, glossary_path=None, chapter_ref=None):
    """
    Compress CSV glossary by excluding entries not found in source text.
    Handles both legacy CSV format and token-efficient format.
    Falls back to text-based scanning if CSV parsing yields 0 entries.
    """
    if not isinstance(csv_content, str):
        return csv_content
    
    lines = csv_content.strip().split('\n')
    if not lines:
        return csv_content
    
    # Check if this is token-efficient format (has section headers like "=== CHARACTERS ===")
    is_token_efficient = any(line.strip().startswith('===') for line in lines)
    
    def _run():
        if is_token_efficient:
            return _compress_token_efficient_format(
                lines, source_text, glossary_path=glossary_path, chapter_ref=chapter_ref)
        return _compress_legacy_csv_format(
            lines, source_text, glossary_path=glossary_path, chapter_ref=chapter_ref)

    def _data_lines(value):
        if not isinstance(value, str):
            return []
        return [l for l in value.split('\n') if l.strip()
                and not _is_glossary_header(l)
                and not l.strip().startswith('===')
                and not l.strip().lower().startswith('glossary columns:')]

    result = _run_with_zero_match_relaxation(
        _active_match_context(source_text), _run,
        lambda value: len(_data_lines(value)), "CSV",
    )
    result_data_lines = _data_lines(result)

    original_data_count = sum(1 for l in lines if l.strip()
                              and not _is_glossary_header(l)
                              and not l.strip().startswith('===')
                              and not l.strip().lower().startswith('glossary'))
    
    if len(result_data_lines) == 0 and original_data_count > 0:
        # 0 matching entries → send NO glossary for this chapter. Do NOT fall back
        # to the raw-name scan: that returns (effectively) the whole glossary,
        # which is what caused multipass/refinement to ship the full ~72k-char
        # glossary. An empty result makes build_system_prompt skip the glossary
        # append entirely.
        print("ℹ️ Glossary compression: CSV produced 0 matching entries for this chapter — sending no glossary entries")
        return ""  # Return empty so the caller doesn't append a header-only/full glossary

    return result


def _token_entry_identity(line):
    body = line.strip()[2:].strip()

    def _split_head_desc(text):
        paren_depth = 0
        bracket_depth = 0
        for idx, ch in enumerate(text):
            if ch == '(' and bracket_depth == 0:
                paren_depth += 1
            elif ch == ')' and bracket_depth == 0 and paren_depth > 0:
                paren_depth -= 1
            elif ch == '[' and paren_depth == 0:
                bracket_depth += 1
            elif ch == ']' and paren_depth == 0 and bracket_depth > 0:
                bracket_depth -= 1
            elif ch == ':' and paren_depth == 0 and bracket_depth == 0:
                return text[:idx].rstrip(), text[idx + 1:].strip()
        return text, ""

    head, _desc = _split_head_desc(body)

    def _strip_custom_tails(text):
        while True:
            custom_tail = re.search(r"\s+\(([^()]*)\)\s*$", text)
            if not custom_tail or ":" not in custom_tail.group(1):
                return text.rstrip()
            text = text[:custom_tail.start()].rstrip()

    head = _strip_custom_tails(head)
    gender = ""
    gender_match = re.search(r"\s*\[([^\]]*)\]\s*$", head)
    if gender_match:
        gender = gender_match.group(1).strip()
        head = head[:gender_match.start()].rstrip()
    head = _strip_custom_tails(head)
    equal_match = re.match(r"^(?P<raw>.+?)\s*=\s*(?P<translated>.+?)\s*$", head)
    if equal_match:
        return equal_match.group("raw").strip(), equal_match.group("translated").strip(), gender
    name_match = re.match(r"^(?P<translated>.*)\s+\((?P<raw>.*?)\)\s*$", head)
    if not name_match:
        return "", "", ""
    return name_match.group("raw").strip(), name_match.group("translated").strip(), gender


def _entry_matches_source(source_text, raw_name, translated_name="", is_character=False,
                          entry_type=""):
    """Return True if the entry is relevant to the source text.

    The single decision point for every structured format. Which matcher
    actually answers is decided by the active _MatchContext.
    """
    raw_name = str(raw_name or "").strip()
    translated_name = str(translated_name or "").strip()
    ctx = _active_match_context(source_text)
    if raw_name and ctx.decide(
        raw_name, is_character,
        translated_name=translated_name, entry_type=entry_type,
    ):
        return True
    if (
        translated_name
        and translated_name != raw_name
        and _consider_translated_column_enabled()
        and ctx.decide(
            translated_name, is_character,
            translated_name=translated_name, entry_type=entry_type,
        )
    ):
        return True
    return False


def _compress_token_efficient_format(lines, source_text, glossary_path=None, chapter_ref=None):
    """Compress token-efficient glossary format with section headers."""
    filtered_lines = []
    current_section = None
    current_section_name = ''
    current_section_has_gender = False
    _gender_types = _get_gender_types()
    gender_tracker = _load_gender_tracker(glossary_path)
    available_genders = {}
    scan_section_has_gender = False
    for scan_line in lines:
        scan_stripped = scan_line.strip()
        if scan_stripped.startswith('==='):
            header_upper = scan_stripped.upper()
            scan_section_has_gender = any(t.upper() in header_upper for t in _gender_types)
            continue
        if scan_stripped.startswith('* '):
            raw_name, _translated_name, gender = _token_entry_identity(scan_stripped)
            if scan_section_has_gender or _has_explicit_gender_value(gender):
                _remember_available_gender(available_genders, raw_name, gender)
    
    for _yield_idx, line in enumerate(lines):
        _gil_yield(_yield_idx)
        stripped = line.strip()

        # Keep glossary header (e.g. "Glossary Columns: ...")
        if stripped.lower().startswith('glossary:') or stripped.lower().startswith('glossary columns:'):
            filtered_lines.append(line)
            continue
        
        # Track section headers
        if stripped.startswith('==='):
            current_section = line
            # current_section is cleared once the header has been emitted, so
            # keep the bare name separately for the shadow log's entry_type.
            current_section_name = stripped.strip('= ').strip().lower()
            # Check if any gender-enabled type name appears in the header
            header_upper = stripped.upper()
            current_section_has_gender = any(
                t.upper() in header_upper for t in _gender_types
            )
            continue
        
        # Process entry lines (start with "* ")
        if stripped.startswith('* '):
            raw_name, translated_name, gender = _token_entry_identity(stripped)
            is_gender_entry = current_section_has_gender or _has_explicit_gender_value(gender)
            if _entry_matches_source(source_text, raw_name, translated_name,
                                     is_character=is_gender_entry,
                                     entry_type=current_section_name):
                if not _gender_variant_allowed(gender_tracker, raw_name, gender, chapter_ref, available_genders):
                    continue
                # Add section header if this is the first entry in section
                if current_section:
                    filtered_lines.append(current_section)
                    current_section = None
                emitted_gender = _emitted_gender(
                    gender_tracker,
                    raw_name,
                    gender,
                    chapter_ref,
                    available_genders,
                )
                filtered_lines.append(_replace_token_gender(line, emitted_gender) if is_gender_entry else line)
        elif not stripped:
            # Keep blank lines
            filtered_lines.append(line)
    
    return '\n'.join(filtered_lines)


def _compress_legacy_csv_format(lines, source_text, glossary_path=None, chapter_ref=None):
    """Compress legacy CSV format with type,raw_name,translated_name columns."""
    if not lines:
        return ''
    
    # Check if first line is a header
    first_line = lines[0].strip().lower()
    has_header = _is_glossary_header(lines[0]) or first_line.startswith('type,') or 'raw_name' in first_line
    
    filtered_lines = []
    gender_tracker = _load_gender_tracker(glossary_path)
    
    # Keep header if present
    if has_header:
        filtered_lines.append(lines[0])
        header_parts = [p.strip().lower() for p in lines[0].split(_gsep(lines[0]))]
        data_lines = lines[1:]
    else:
        header_parts = []
        data_lines = lines
    
    # Auto-detect separator from content
    sample = '\n'.join(lines[:5])
    sep = _gsep(sample)
    available_genders = {}
    for scan_line in data_lines:
        try:
            parts = [p.strip() for p in scan_line.split(sep)]
            if len(parts) >= 3:
                entry_type = parts[0].strip().lower()
                if header_parts:
                    raw_idx = header_parts.index("raw_name") if "raw_name" in header_parts else 1
                    gender_idx = header_parts.index("gender") if "gender" in header_parts else 3
                else:
                    raw_idx = 1
                    gender_idx = 3
                raw_name = parts[raw_idx].strip() if len(parts) > raw_idx else ""
                gender = parts[gender_idx].strip() if len(parts) > gender_idx else ""
                if is_gender_entry_type(entry_type, gender, _get_gender_types()):
                    _remember_available_gender(available_genders, raw_name, gender)
        except Exception:
            pass
    
    # Process each CSV row
    for _yield_idx, line in enumerate(data_lines):
        _gil_yield(_yield_idx)
        if not line.strip():
            continue
        
        try:
            # Parse CSV line using detected separator
            parts = [p.strip() for p in line.split(sep)]
            if len(parts) >= 3:
                entry_type = parts[0].strip().lower()
                raw_name = parts[1].strip()
                translated_name = parts[2].strip()
                if header_parts:
                    raw_idx = header_parts.index("raw_name") if "raw_name" in header_parts else 1
                    translated_idx = header_parts.index("translated_name") if "translated_name" in header_parts else 2
                    gender_idx = header_parts.index("gender") if "gender" in header_parts else 3
                else:
                    raw_idx = 1
                    translated_idx = 2
                    gender_idx = 3
                raw_name = parts[raw_idx].strip() if len(parts) > raw_idx else raw_name
                translated_name = parts[translated_idx].strip() if len(parts) > translated_idx else translated_name
                gender = parts[gender_idx].strip() if len(parts) > gender_idx else ""
                
                is_char = is_gender_entry_type(entry_type, gender, _get_gender_types())
                if _entry_matches_source(source_text, raw_name, translated_name,
                                         is_character=is_char, entry_type=entry_type) \
                        and _gender_variant_allowed(gender_tracker, raw_name, gender, chapter_ref, available_genders):
                    emitted_gender = _emitted_gender(
                        gender_tracker,
                        raw_name,
                        gender,
                        chapter_ref,
                        available_genders,
                    )
                    if is_char and emitted_gender in BINARY_GENDERS and gender_idx < len(parts):
                        parts[gender_idx] = emitted_gender
                        filtered_lines.append(sep.join(parts))
                    else:
                        filtered_lines.append(line)
        except Exception:
            # If parsing fails, keep the line to be safe
            filtered_lines.append(line)
    
    return '\n'.join(filtered_lines)


def _compress_json_glossary(json_data, source_text, glossary_path=None, chapter_ref=None):
    """
    Compress JSON glossary by excluding entries not found in source text.
    Handles both dict format and list format.
    Falls back to text-based scanning if JSON parsing fails.
    """
    if isinstance(json_data, str):
        try:
            json_data = json.loads(json_data)
        except json.JSONDecodeError:
            if _should_use_raw_name_fallback():
                print("⚠️ Glossary compression: JSON parsing failed, falling back to raw-name scan")
                return _compress_fallback_text(json_data, source_text)
            else:
                print("⚠️ Glossary compression: JSON parsing failed (no fallback in auto mode)")
                return json_data
    gender_tracker = _load_gender_tracker(glossary_path)
    
    def _is_char_entry(val):
        """Check if a JSON entry value represents a gender-enabled type."""
        if isinstance(val, dict):
            return is_gender_entry_type(val.get('type'), val.get('gender'), _get_gender_types())
        return False

    def _json_translated_name(value, key=""):
        if isinstance(value, dict):
            return (
                value.get('translated_name')
                or value.get('translation')
                or value.get('translated')
                or value.get('name')
                or ""
            )
        if isinstance(value, str):
            return value
        return ""

    def _json_available_genders(container):
        available_genders = {}
        if isinstance(container, dict):
            items = container.get('entries', container).items()
            for key, value in items:
                if isinstance(value, dict) and _is_char_entry(value):
                    raw_name = value.get('raw_name') or value.get('original_name') or value.get('original') or key
                    _remember_available_gender(available_genders, raw_name, value.get("gender", ""))
        elif isinstance(container, list):
            for entry in container:
                if isinstance(entry, dict) and _is_char_entry(entry):
                    raw_name = entry.get('raw_name') or entry.get('original_name') or entry.get('original') or ''
                    _remember_available_gender(available_genders, raw_name, entry.get("gender", ""))
        return available_genders

    available_genders = _json_available_genders(json_data)

    def _resolved_json_value(value, raw_name, gender):
        if not isinstance(value, dict):
            return value
        emitted = _emitted_gender(
            gender_tracker,
            raw_name,
            gender,
            chapter_ref,
            available_genders,
        )
        if emitted not in BINARY_GENDERS or _normal_gender(value.get("gender", "")) == emitted:
            return value
        resolved = value.copy()
        resolved["gender"] = emitted
        return resolved
    
    if isinstance(json_data, dict):
        # Handle dict with 'entries' key
        if 'entries' in json_data:
            filtered_entries = {}
            for _yield_idx, (key, value) in enumerate(json_data['entries'].items()):
                _gil_yield(_yield_idx)
                gender = value.get("gender", "") if isinstance(value, dict) else ""
                is_char = _is_char_entry(value) or _has_explicit_gender_value(gender)
                raw_name = value.get('raw_name') if isinstance(value, dict) else key
                raw_name = raw_name or key
                translated_name = _json_translated_name(value, key)
                if _entry_matches_source(source_text, raw_name, translated_name,
                                         is_character=is_char,
                                         entry_type=(value.get('type') if isinstance(value, dict) else '') or '') \
                        and _gender_variant_allowed(gender_tracker, raw_name, gender, chapter_ref, available_genders):
                    filtered_entries[key] = _resolved_json_value(value, raw_name, gender)
            
            result = json_data.copy()
            result['entries'] = filtered_entries
            return result
        else:
            # Simple dict format
            filtered_dict = {}
            for _yield_idx, (key, value) in enumerate(json_data.items()):
                _gil_yield(_yield_idx)
                if key == 'metadata':
                    filtered_dict[key] = value
                else:
                    gender = value.get("gender", "") if isinstance(value, dict) else ""
                    is_char = _is_char_entry(value) or _has_explicit_gender_value(gender)
                    raw_name = value.get('raw_name') if isinstance(value, dict) else key
                    raw_name = raw_name or key
                    translated_name = _json_translated_name(value, key)
                    if _entry_matches_source(source_text, raw_name, translated_name,
                                         is_character=is_char,
                                         entry_type=(value.get('type') if isinstance(value, dict) else '') or '') \
                        and _gender_variant_allowed(gender_tracker, raw_name, gender, chapter_ref, available_genders):
                        filtered_dict[key] = _resolved_json_value(value, raw_name, gender)
            return filtered_dict
    
    elif isinstance(json_data, list):
        # List of entry objects
        filtered_list = []
        for _yield_idx, entry in enumerate(json_data):
            _gil_yield(_yield_idx)
            if isinstance(entry, dict):
                # Check various possible keys for the raw term
                raw_term = entry.get('raw_name') or entry.get('original_name') or entry.get('original') or ''
                translated_name = _json_translated_name(entry)
                gender = entry.get("gender", "")
                is_char = is_gender_entry_type(entry.get('type'), gender, _get_gender_types())
                if _entry_matches_source(source_text, raw_term, translated_name,
                                         is_character=is_char,
                                         entry_type=entry.get('type') or '') \
                        and _gender_variant_allowed(gender_tracker, raw_term, gender, chapter_ref, available_genders):
                    filtered_list.append(_resolved_json_value(entry, raw_term, gender))
        return filtered_list
    
    return json_data


# ─── Format-agnostic fallback ────────────────────────────────────────────────

def _is_section_header(line):
    """Check if a line is a section header (e.g. # Title, === SECTION ===)."""
    stripped = line.strip()
    if not stripped:
        return False
    return bool(_SECTION_HEADER_RE.match(stripped))


def _is_entry_line(line):
    """Check if a line looks like a self-contained glossary entry.
    
    Returns True if the line has: a known delimiter between terms,
    a bullet/list marker, or a table row marker.
    """
    stripped = line.strip()
    if not stripped:
        return False
    
    # Starts with bullet marker: * , - , • , 1. , 2)
    if _ENTRY_BULLET_RE.match(stripped):
        return True
    
    # Starts with table pipe
    if _ENTRY_TABLE_RE.match(stripped):
        return True
    
    # Contains a known delimiter between terms
    for delim in _ENTRY_DELIMITERS:
        if delim in stripped:
            return True
    
    # Comma-separated: only if 3+ fields (to distinguish CSV-like entries
    # from prose that happens to contain a comma like "The protagonist, male")
    if ',' in stripped and len([p for p in stripped.split(',') if p.strip()]) >= 3:
        return True
    
    # Contains parenthesized text (common pattern: "word (other_word)")
    if re.search(r'\S\s*\([^)]+\)', stripped):
        return True
    
    return False


def _fallback_max_candidates():
    try:
        return max(1, int(str(_setting("GLOSSARY_FALLBACK_MAX_CANDIDATES", "6")).strip()))
    except (TypeError, ValueError):
        return 6


def _entry_head_segment(text):
    """The part of an entry unit that can plausibly hold the term.

    The fallback used to split the WHOLE entry — description included — and
    test every fragment as if it were a term, so a common word in a
    description kept the entry. No amount of matcher tightening fixes that;
    the fix is to stop offering descriptions as candidates.

    The cut is at the *description* delimiter, not the first delimiter of any
    kind: "루나 = Luna: the protagonist" keeps both 루나 and Luna and drops
    only the description. Cutting at " = " instead would lose the term
    entirely in glossaries written the other way round ("Luna = 루나").

    Depth-aware so a colon inside brackets or parentheses does not split.
    Continuation lines are description by definition and are dropped.
    """
    first_line = text.split("\n", 1)[0]
    paren = bracket = 0
    for idx, ch in enumerate(first_line):
        if ch == '(' and bracket == 0:
            paren += 1
        elif ch == ')' and bracket == 0 and paren > 0:
            paren -= 1
        elif ch == '[' and paren == 0:
            bracket += 1
        elif ch == ']' and paren == 0 and bracket > 0:
            bracket -= 1
        elif ch in ':\t\x1F' and paren == 0 and bracket == 0:
            return first_line[:idx]
    return first_line


def _extract_candidates(text, head_only=False):
    """Extract candidate terms from text by splitting on common delimiters.

    Returns a list of candidate strings (stripped, non-empty, >= 2 chars,
    non-numeric). These are potential raw names to check against source text.

    With head_only, only the entry's head segment is considered — see
    _entry_head_segment. Callers fall back to the full unit when the head
    yields nothing, since a hand-written glossary may put the term after the
    delimiter rather than before it.
    """
    if head_only:
        text = _entry_head_segment(text)
    tokens = _FALLBACK_SPLIT_RE.split(text)
    candidates = []
    _skip_words = {'type', 'raw_name', 'translated_name', 'gender', 'description',
                   'raw', 'translation', 'notes', 'name', 'comment', 'context'}
    for t in tokens:
        t = t.strip().strip('"\'´`*•')  # strip surrounding quotes/markers
        # Strip leading bullet markers: "- term" → "term"
        t = re.sub(r'^[-\-\u2013\u2014]\s*', '', t).strip()
        # Strip leading numbered list markers: "1. term" → "term"
        t = re.sub(r'^\d+[.)\]]\s*', '', t).strip()
        if len(t) >= 2 and not t.isdigit() and t.lower() not in _skip_words:
            candidates.append(t)
    return candidates


def _compress_fallback_text(content, source_text):
    """Format-agnostic fallback: scan for raw names in any text format.
    
    Algorithm:
      1. Classify each line as HEADER, ENTRY, CONTINUATION, or BLANK.
      2. Group lines into entry units (entry line + its continuation lines).
      3. Extract candidate terms from each entry unit.
      4. Keep entire entry units whose candidates appear in source text.
      5. Keep section headers only if at least one child entry survives.
      6. Always operates on full lines — never cuts mid-line.
    """
    if not isinstance(content, str):
        return content
    
    lines = content.split('\n')
    if not lines:
        return content
    
    # ── Phase 1: Classify each line ──────────────────────────────────────
    # Types: 'header', 'entry', 'continuation', 'blank', 'meta'
    classifications = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            classifications.append('blank')
        elif stripped.lower().startswith('glossary columns:') or stripped.lower().startswith('glossary:'):
            classifications.append('meta')  # always keep
        elif _is_section_header(stripped):
            classifications.append('header')
        elif _is_entry_line(line):
            classifications.append('entry')
        else:
            classifications.append('continuation')
    
    # ── Phase 2: Group lines into entry units ────────────────────────────
    # An entry unit = an ENTRY line + any following CONTINUATION lines
    # (until the next ENTRY, HEADER, BLANK, or META line).
    
    # Each item: {'type': 'entry'|'header'|'blank'|'meta', 
    #             'line_indices': [int, ...]}
    groups = []
    i = 0
    while i < len(lines):
        cls = classifications[i]
        
        if cls == 'meta':
            groups.append({'type': 'meta', 'line_indices': [i]})
            i += 1
        elif cls == 'blank':
            groups.append({'type': 'blank', 'line_indices': [i]})
            i += 1
        elif cls == 'header':
            groups.append({'type': 'header', 'line_indices': [i]})
            i += 1
        elif cls == 'entry':
            # Collect this entry line + any following continuation lines
            indices = [i]
            i += 1
            while i < len(lines) and classifications[i] == 'continuation':
                indices.append(i)
                i += 1
            groups.append({'type': 'entry', 'line_indices': indices})
        elif cls == 'continuation':
            # Orphan continuation (no preceding entry) — treat as a standalone entry
            indices = [i]
            i += 1
            while i < len(lines) and classifications[i] == 'continuation':
                indices.append(i)
                i += 1
            groups.append({'type': 'entry', 'line_indices': indices})
        else:
            i += 1
    
    # ── Phase 3: Match entry groups against source text ──────────────────
    # For each entry group, extract candidates and check against source.
    # Track which section header (if any) precedes each entry group for
    # character-type detection.
    _gender_types = _get_gender_types()
    current_section_has_gender = False
    for _yield_idx, group in enumerate(groups):
        _gil_yield(_yield_idx)
        if group['type'] == 'header':
            header_text = lines[group['line_indices'][0]].strip().upper()
            current_section_has_gender = any(
                t.upper() in header_text for t in _gender_types
            )
        elif group['type'] == 'entry':
            entry_text = '\n'.join(lines[idx] for idx in group['line_indices'])
            # Prefer the entry head; fall back to the whole unit only when the
            # head yields nothing, so an unusual manual format still works.
            candidates = _extract_candidates(entry_text, head_only=True)
            if not candidates:
                candidates = _extract_candidates(entry_text)
            candidates = candidates[:_fallback_max_candidates()]
            _ctx = _active_match_context(source_text)
            group['keep'] = any(
                _ctx.decide(c, current_section_has_gender, entry_type="fallback")
                for c in candidates
            )
        elif group['type'] in ('meta', 'blank'):
            group['keep'] = True  # always keep meta lines and blanks (blanks filtered later)
        elif group['type'] == 'header':
            group['keep'] = False  # determined by child entries below
    
    # ── Phase 4: Floating header logic ───────────────────────────────────
    # A header is kept only if at least one entry after it (before the next
    # header) is kept.
    for gi, group in enumerate(groups):
        if group['type'] != 'header':
            continue
        # Look forward for kept entries under this header
        has_kept_child = False
        for gj in range(gi + 1, len(groups)):
            if groups[gj]['type'] == 'header':
                break  # next header reached, stop looking
            if groups[gj]['type'] == 'entry' and groups[gj].get('keep'):
                has_kept_child = True
                break
        group['keep'] = has_kept_child
    
    # ── Phase 5: Reassemble ──────────────────────────────────────────────
    # Collect kept lines, then strip trailing blank lines from dropped sections.
    kept_line_set = set()
    for group in groups:
        if group.get('keep'):
            for idx in group['line_indices']:
                kept_line_set.add(idx)
    
    # Build result, preserving original line order
    result_lines = []
    for i, line in enumerate(lines):
        if i in kept_line_set:
            result_lines.append(line)
        # For blank lines: include only if adjacent to kept content
        elif classifications[i] == 'blank':
            # Check if there's kept content both before and after
            has_before = any(j in kept_line_set for j in range(max(0, i - 3), i))
            has_after = any(j in kept_line_set for j in range(i + 1, min(len(lines), i + 4)))
            if has_before and has_after:
                result_lines.append(line)
    
    # Strip consecutive trailing blank lines
    while result_lines and not result_lines[-1].strip():
        result_lines.pop()
    
    return '\n'.join(result_lines)


def _text_contains_term(text, term, is_character=False):
    """
    Check if term appears in text using substring matching.
    Works well with any language — CJK, Latin, Arabic, etc.

    For multi-word terms (e.g. "미샤 랄토스"), also checks if ANY
    individual word appears in the source text, so that a partial
    name match (family name or given name alone) still keeps the
    glossary entry.

    The algorithm lives in glossary_matching so that glossary_usage can
    share it without importing this module's heavy dependency chain.
    This wrapper only supplies the strict-gender setting.

    Args:
        is_character: When True (character-type entries), accept
            partial tokens of any length (≥1 char). When False,
            require ≥2 chars to reduce false positives on
            non-character entries like terms and places.
    """
    if is_character and _strict_gender_name_matching_enabled():
        return strict_name_in_text(
            text, term, strict_name_config(MatchConfig.from_getter(_setting))
        )
    return legacy_text_contains_term(
        text, term, is_character=is_character, strict_gender=False
    )


def compress_glossary_file(glossary_path, source_text):
    """
    Load, compress, and return glossary from file path.
    
    Args:
        glossary_path: Path to glossary file (.csv, .json, .md, .txt, etc.)
        source_text: The source text to check against
    
    Returns:
        Compressed glossary content in appropriate format
    """
    if not glossary_path or not os.path.exists(glossary_path):
        return None
    
    try:
        with open(glossary_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Determine format from file extension
        ext = os.path.splitext(glossary_path)[1].lower()
        if ext == '.csv':
            return compress_glossary(content, source_text, glossary_format='csv', glossary_path=glossary_path)
        elif ext == '.json':
            json_data = json.loads(content)
            compressed_data = compress_glossary(json_data, source_text, glossary_format='json', glossary_path=glossary_path)
            # Return as JSON string
            return json.dumps(compressed_data, ensure_ascii=False, indent=2)
        else:
            # .md, .txt, or any other extension — use text fallback
            if _should_use_raw_name_fallback():
                print(f"⚠️ Glossary compression: using fallback raw-name scan (format: {ext or 'unknown'})")
                return compress_glossary(content, source_text, glossary_format='text')
            else:
                # Auto mode — return content as-is, no raw-name scan
                return content
    except Exception as e:
        print(f"⚠️ Failed to compress glossary: {e}")
        return None
