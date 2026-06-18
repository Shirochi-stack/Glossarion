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
from io import StringIO

# Serialize glossary compression across translation worker threads.
# Compression is pure-Python (terms x chapter-text substring scans); when
# several chunk workers run it concurrently they monopolize the GIL and the
# GUI thread stalls for seconds (verified via the freeze watchdog in frozen
# builds). One compression at a time keeps the GUI responsive and costs
# almost nothing overall since API latency dominates per-chunk time.
_COMPRESS_GIL_LOCK = threading.Lock()


def _gil_yield(counter, every=64):
    """Briefly release the GIL every `every` iterations of a hot loop."""
    if counter % every == 0:
        time.sleep(0)

try:
    from extract_glossary_from_epub import get_custom_entry_types as _get_custom_entry_types
except ImportError:
    _get_custom_entry_types = None

_gender_bias_log_seen = set()


def _should_use_raw_name_fallback():
    """Return True if raw-name scan fallback is appropriate.

    The raw-name scan is a brute-force approach that only makes sense for
    user-supplied (manual) glossaries whose format may be arbitrary.  For
    auto-generated glossaries (Full, Balanced, Minimal, Single Pass, etc.)
    the structured CSV/JSON parsing is authoritative and a 0-match result
    simply means no entries are relevant to this chapter.

    Allowed modes: 'off_no_automap' (Manual Glossary Only), 'off', 'off_fuzzy'.
    """
    mode = os.getenv('AUTO_GLOSSARY_MODE', 'off').lower().strip()
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
    return str(os.getenv("COMPRESS_GLOSSARY_STRICT_GENDER_MATCHING", "0")).strip().lower() in (
        "1", "true", "yes", "on"
    )


def _consider_translated_column_enabled():
    """Return True when glossary compression may match translated_name too."""
    return str(os.getenv("COMPRESS_GLOSSARY_CONSIDER_TRANSLATED_COLUMN", "0")).strip().lower() in (
        "1", "true", "yes", "on"
    )

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
    if not glossary_path:
        return None
    stem, _ext = os.path.splitext(glossary_path)
    if stem.endswith("_glossary"):
        stem = stem[:-len("_glossary")]
    elif os.path.basename(stem).lower() == "glossary":
        stem = os.path.join(os.path.dirname(stem), "gender")
    return f"{stem}_gender_tracker.json"


def _load_gender_tracker(glossary_path):
    if str(os.getenv("GLOSSARY_SKIP_GENDER_TRACKING", "0")).strip().lower() in ("1", "true", "yes", "on"):
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
    gender = str(value or "").strip().lower()
    aliases = {
        "m": "male",
        "man": "male",
        "boy": "male",
        "masc": "male",
        "masculine": "male",
        "f": "female",
        "woman": "female",
        "girl": "female",
        "fem": "female",
        "feminine": "female",
    }
    return aliases.get(gender, gender)


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
        chapter_num = os.getenv("CURRENT_CHAPTER_NUM")
    if not chapter_file:
        chapter_file = os.getenv("CURRENT_CHAPTER_FILE")
    try:
        chapter_num_f = float(chapter_num)
    except Exception:
        chapter_num_f = None
    return chapter_num_f, os.path.basename(str(chapter_file or ""))


def _tracker_entry_for_raw(tracker, raw_name):
    if not tracker or not raw_name:
        return None
    entry = tracker.get("entries", {}).get(str(raw_name).strip().casefold())
    return entry if isinstance(entry, dict) else None


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
        value = float(os.getenv("GLOSSARY_GENDER_NOISE_THRESHOLD", "10"))
    except Exception:
        value = 10.0
    return max(0.0, min(100.0, value)) / 100.0


def _gender_bias():
    bias = _normal_gender(os.getenv("GLOSSARY_GENDER_TRACKING_BIAS", "none"))
    return bias if bias in {"male", "female"} else "none"


def _gender_rarity_stats(entry):
    occurrences = [o for o in entry.get("occurrences", []) if isinstance(o, dict)] if isinstance(entry, dict) else []
    genders = [_normal_gender(o.get("gender")) for o in occurrences]
    genders = [g for g in genders if g not in {"", "unknown", "n/a", "na", "none", "-"}]
    total = len(genders)
    stats = {}
    if not total:
        return stats
    for gender in set(genders):
        count = sum(1 for g in genders if g == gender)
        stats[gender] = {
            "count": count,
            "total": total,
            "ratio": count / total,
        }
    return stats


def _rare_tracker_genders(entry):
    if not entry:
        return set()
    threshold = _gender_noise_threshold()
    if threshold <= 0:
        return set()
    bias = _gender_bias()
    stats = _gender_rarity_stats(entry)
    rare = set()
    for gender, values in stats.items():
        if gender == bias:
            continue
        if values["ratio"] <= threshold:
            rare.add(gender)
    return rare


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
    if not isinstance(entry, dict):
        return None
    occurrences = [o for o in entry.get("occurrences", []) if isinstance(o, dict)]
    rare_genders = _rare_tracker_genders(entry)
    if rare_genders:
        filtered_occurrences = [o for o in occurrences if _normal_gender(o.get("gender")) not in rare_genders]
        if filtered_occurrences:
            occurrences = filtered_occurrences
    if not occurrences:
        return None
    chapter_num, chapter_file = _chapter_ref_parts(chapter_ref)
    if chapter_file:
        for occ in reversed(occurrences):
            if os.path.basename(str(occ.get("chapter_file", ""))) == chapter_file:
                return _normal_gender(occ.get("gender"))
    if chapter_num is not None:
        best = None
        best_num = None
        for occ in occurrences:
            try:
                occ_num = float(occ.get("chapter_num"))
            except Exception:
                continue
            if occ_num <= chapter_num and (best_num is None or occ_num >= best_num):
                best = occ
                best_num = occ_num
        if best:
            return _normal_gender(best.get("gender"))
    return _normal_gender(occurrences[-1].get("gender"))


def _gender_variant_allowed(tracker, raw_name, gender, chapter_ref=None, available_genders=None):
    actual = _normal_gender(gender)
    if not actual or actual in {"unknown", "n/a", "na", "none", "-"}:
        return True
    bias = _gender_bias()
    available = _available_gender_set(available_genders, raw_name)
    if _gender_noise_threshold() >= 1.0 and bias == "none":
        return True
    entry = _tracker_entry_for_raw(tracker, raw_name)
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


def compress_glossary(glossary_content, source_text, glossary_format='auto', glossary_path=None, chapter_ref=None):
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
    with _COMPRESS_GIL_LOCK:
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
    
    if is_token_efficient:
        result = _compress_token_efficient_format(lines, source_text, glossary_path=glossary_path, chapter_ref=chapter_ref)
    else:
        result = _compress_legacy_csv_format(lines, source_text, glossary_path=glossary_path, chapter_ref=chapter_ref)
    
    # If CSV parsing produced 0 data entries, fall back to text scan
    if isinstance(result, str):
        result_data_lines = [l for l in result.split('\n') if l.strip()
                             and not _is_glossary_header(l)
                             and not l.strip().startswith('===')
                             and not l.strip().lower().startswith('glossary columns:')]
    else:
        result_data_lines = []
    
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


def _entry_matches_source(source_text, raw_name, translated_name="", is_character=False):
    """Return True if the entry is relevant to the source text."""
    raw_name = str(raw_name or "").strip()
    translated_name = str(translated_name or "").strip()
    if raw_name and _text_contains_term(source_text, raw_name, is_character=is_character):
        return True
    if (
        translated_name
        and translated_name != raw_name
        and _consider_translated_column_enabled()
        and _text_contains_term(source_text, translated_name, is_character=is_character)
    ):
        return True
    return False


def _compress_token_efficient_format(lines, source_text, glossary_path=None, chapter_ref=None):
    """Compress token-efficient glossary format with section headers."""
    filtered_lines = []
    current_section = None
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
            if _entry_matches_source(source_text, raw_name, translated_name, is_character=is_gender_entry):
                if not _gender_variant_allowed(gender_tracker, raw_name, gender, chapter_ref, available_genders):
                    continue
                # Add section header if this is the first entry in section
                if current_section:
                    filtered_lines.append(current_section)
                    current_section = None
                filtered_lines.append(line)
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
                if entry_type in _get_gender_types() or _has_explicit_gender_value(gender):
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
                
                is_char = entry_type in _get_gender_types() or _has_explicit_gender_value(gender)
                if _entry_matches_source(source_text, raw_name, translated_name, is_character=is_char) and _gender_variant_allowed(gender_tracker, raw_name, gender, chapter_ref, available_genders):
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
            return val.get('type', '').lower() in _get_gender_types()
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
                if _entry_matches_source(source_text, raw_name, translated_name, is_character=is_char) and _gender_variant_allowed(gender_tracker, raw_name, gender, chapter_ref, available_genders):
                    filtered_entries[key] = value
            
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
                    if _entry_matches_source(source_text, raw_name, translated_name, is_character=is_char) and _gender_variant_allowed(gender_tracker, raw_name, gender, chapter_ref, available_genders):
                        filtered_dict[key] = value
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
                is_char = entry.get('type', '').lower() in _get_gender_types() or _has_explicit_gender_value(gender)
                if _entry_matches_source(source_text, raw_term, translated_name, is_character=is_char) and _gender_variant_allowed(gender_tracker, raw_term, gender, chapter_ref, available_genders):
                    filtered_list.append(entry)
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


def _extract_candidates(text):
    """Extract candidate terms from text by splitting on common delimiters.
    
    Returns a list of candidate strings (stripped, non-empty, >= 2 chars,
    non-numeric). These are potential raw names to check against source text.
    """
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
            candidates = _extract_candidates(entry_text)
            group['keep'] = any(_text_contains_term(source_text, c, is_character=current_section_has_gender) for c in candidates)
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
    
    Args:
        is_character: When True (character-type entries), accept
            partial tokens of any length (≥1 char). When False,
            require ≥2 chars to reduce false positives on
            non-character entries like terms and places.
    """
    if not term or not text:
        return False
    
    # Full term match first (fast path)
    if term in text:
        return True

    if is_character and _strict_gender_name_matching_enabled():
        return False
    
    # Multi-word: check individual tokens
    # Character entries: accept any token length (≥1 char)
    # Non-character entries: require ≥2 chars to limit false positives
    min_token_len = 1 if is_character else 2
    if ' ' in term:
        for token in term.split():
            if len(token) >= min_token_len and token in text:
                return True

    return False


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
