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
from io import StringIO

try:
    from extract_glossary_from_epub import get_custom_entry_types as _get_custom_entry_types
except ImportError:
    _get_custom_entry_types = None


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


def compress_glossary(glossary_content, source_text, glossary_format='auto'):
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
    
    if glossary_format == 'csv':
        return _compress_csv_glossary(glossary_content, source_text)
    elif glossary_format == 'json':
        return _compress_json_glossary(glossary_content, source_text)
    elif glossary_format == 'text':
        print("⚠️ Glossary compression: using fallback raw-name scan (unrecognized format)")
        return _compress_fallback_text(glossary_content, source_text)
    else:
        return glossary_content


def _compress_csv_glossary(csv_content, source_text):
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
        result = _compress_token_efficient_format(lines, source_text)
    else:
        result = _compress_legacy_csv_format(lines, source_text)
    
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
        print("⚠️ Glossary compression: CSV produced 0 matching entries, falling back to raw-name scan")
        return _compress_fallback_text(csv_content, source_text)
    
    return result


def _compress_token_efficient_format(lines, source_text):
    """Compress token-efficient glossary format with section headers."""
    filtered_lines = []
    current_section = None
    current_section_has_gender = False
    _gender_types = _get_gender_types()
    
    for line in lines:
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
            # Extract the raw name from the entry
            # Format: * TranslatedName (RawName) [Gender]
            match = re.search(r'\(([^)]+)\)', stripped)
            if match:
                raw_name = match.group(1).strip()
                # Check if raw name appears in source text
                if _text_contains_term(source_text, raw_name, is_character=current_section_has_gender):
                    # Add section header if this is the first entry in section
                    if current_section:
                        filtered_lines.append(current_section)
                        current_section = None
                    filtered_lines.append(line)
        elif not stripped:
            # Keep blank lines
            filtered_lines.append(line)
    
    return '\n'.join(filtered_lines)


def _compress_legacy_csv_format(lines, source_text):
    """Compress legacy CSV format with type,raw_name,translated_name columns."""
    if not lines:
        return ''
    
    # Check if first line is a header
    first_line = lines[0].strip().lower()
    has_header = _is_glossary_header(lines[0]) or first_line.startswith('type,') or 'raw_name' in first_line
    
    filtered_lines = []
    
    # Keep header if present
    if has_header:
        filtered_lines.append(lines[0])
        data_lines = lines[1:]
    else:
        data_lines = lines
    
    # Auto-detect separator from content
    sample = '\n'.join(lines[:5])
    sep = _gsep(sample)
    
    # Process each CSV row
    for line in data_lines:
        if not line.strip():
            continue
        
        try:
            # Parse CSV line using detected separator
            parts = [p.strip() for p in line.split(sep)]
            if len(parts) >= 3:
                entry_type = parts[0].strip().lower()
                raw_name = parts[1].strip()
                translated_name = parts[2].strip()
                
                is_char = entry_type in _get_gender_types()
                # Check if raw name appears in source text
                if _text_contains_term(source_text, raw_name, is_character=is_char):
                    filtered_lines.append(line)
        except Exception:
            # If parsing fails, keep the line to be safe
            filtered_lines.append(line)
    
    return '\n'.join(filtered_lines)


def _compress_json_glossary(json_data, source_text):
    """
    Compress JSON glossary by excluding entries not found in source text.
    Handles both dict format and list format.
    Falls back to text-based scanning if JSON parsing fails.
    """
    if isinstance(json_data, str):
        try:
            json_data = json.loads(json_data)
        except json.JSONDecodeError:
            print("⚠️ Glossary compression: JSON parsing failed, falling back to raw-name scan")
            return _compress_fallback_text(json_data, source_text)
    
    def _is_char_entry(val):
        """Check if a JSON entry value represents a gender-enabled type."""
        if isinstance(val, dict):
            return val.get('type', '').lower() in _get_gender_types()
        return False
    
    if isinstance(json_data, dict):
        # Handle dict with 'entries' key
        if 'entries' in json_data:
            filtered_entries = {}
            for key, value in json_data['entries'].items():
                is_char = _is_char_entry(value)
                if _text_contains_term(source_text, key, is_character=is_char):
                    filtered_entries[key] = value
            
            result = json_data.copy()
            result['entries'] = filtered_entries
            return result
        else:
            # Simple dict format
            filtered_dict = {}
            for key, value in json_data.items():
                if key == 'metadata':
                    filtered_dict[key] = value
                else:
                    is_char = _is_char_entry(value)
                    if _text_contains_term(source_text, key, is_character=is_char):
                        filtered_dict[key] = value
            return filtered_dict
    
    elif isinstance(json_data, list):
        # List of entry objects
        filtered_list = []
        for entry in json_data:
            if isinstance(entry, dict):
                # Check various possible keys for the raw term
                raw_term = entry.get('raw_name') or entry.get('original_name') or entry.get('original') or ''
                is_char = entry.get('type', '').lower() in _get_gender_types()
                if raw_term and _text_contains_term(source_text, raw_term, is_character=is_char):
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
    for group in groups:
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
            return compress_glossary(content, source_text, glossary_format='csv')
        elif ext == '.json':
            json_data = json.loads(content)
            compressed_data = compress_glossary(json_data, source_text, glossary_format='json')
            # Return as JSON string
            return json.dumps(compressed_data, ensure_ascii=False, indent=2)
        else:
            # .md, .txt, or any other extension — use text fallback
            print(f"⚠️ Glossary compression: using fallback raw-name scan (format: {ext or 'unknown'})")
            return compress_glossary(content, source_text, glossary_format='text')
    except Exception as e:
        print(f"⚠️ Failed to compress glossary: {e}")
        return None
