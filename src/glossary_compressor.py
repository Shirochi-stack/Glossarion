# -*- coding: utf-8 -*-
"""
Glossary Compressor Module
Filters glossary entries based on source text to reduce token usage
"""

import os
import re
import json
import csv
from io import StringIO


def compress_glossary(glossary_content, source_text, glossary_format='auto'):
    """
    Compress glossary by excluding entries that don't appear in the source text.
    
    Args:
        glossary_content: Raw glossary content (CSV string or JSON dict/list)
        source_text: The source text to check against
        glossary_format: 'csv', 'json', or 'auto' (detect from content)
    
    Returns:
        Compressed glossary in the same format as input
    """
    if not glossary_content or not source_text:
        return glossary_content
    
    # Auto-detect format
    if glossary_format == 'auto':
        if isinstance(glossary_content, str):
            # Check if it looks like JSON
            stripped = glossary_content.strip()
            if (stripped.startswith('{') or stripped.startswith('[')) and (stripped.endswith('}') or stripped.endswith(']')):
                glossary_format = 'json'
            else:
                glossary_format = 'csv'
        elif isinstance(glossary_content, (dict, list)):
            glossary_format = 'json'
        else:
            return glossary_content
    
    if glossary_format == 'csv':
        return _compress_csv_glossary(glossary_content, source_text)
    elif glossary_format == 'json':
        return _compress_json_glossary(glossary_content, source_text)
    else:
        return glossary_content


def _compress_csv_glossary(csv_content, source_text):
    """
    Compress CSV glossary by excluding entries not found in source text.
    Handles both legacy CSV format and token-efficient format.
    """
    if not isinstance(csv_content, str):
        return csv_content
    
    lines = csv_content.strip().split('\n')
    if not lines:
        return csv_content
    
    # Check if this is token-efficient format (has section headers like "=== CHARACTERS ===")
    is_token_efficient = any(line.strip().startswith('===') for line in lines)
    
    if is_token_efficient:
        return _compress_token_efficient_format(lines, source_text)
    else:
        return _compress_legacy_csv_format(lines, source_text)


def _compress_token_efficient_format(lines, source_text):
    """Compress token-efficient glossary format with section headers."""
    filtered_lines = []
    current_section = None
    
    for line in lines:
        stripped = line.strip()
        
        # Keep glossary header
        if stripped.lower().startswith('glossary:'):
            filtered_lines.append(line)
            continue
        
        # Track section headers
        if stripped.startswith('==='):
            current_section = line
            continue
        
        # Process entry lines (start with "* ")
        if stripped.startswith('* '):
            # Extract the raw name from the entry
            # Format: * TranslatedName (RawName) [Gender]
            match = re.search(r'\(([^)]+)\)', stripped)
            if match:
                raw_name = match.group(1).strip()
                # Check if raw name appears in source text
                if _text_contains_term(source_text, raw_name):
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
    has_header = first_line.startswith('type,') or 'raw_name' in first_line
    
    filtered_lines = []
    
    # Keep header if present
    if has_header:
        filtered_lines.append(lines[0])
        data_lines = lines[1:]
    else:
        data_lines = lines
    
    # Process each CSV row
    for line in data_lines:
        if not line.strip():
            continue
        
        try:
            # Parse CSV line
            parts = list(csv.reader(StringIO(line)))[0]
            if len(parts) >= 3:
                entry_type = parts[0].strip()
                raw_name = parts[1].strip()
                translated_name = parts[2].strip()
                
                # Check if raw name appears in source text
                if _text_contains_term(source_text, raw_name):
                    filtered_lines.append(line)
        except Exception:
            # If parsing fails, keep the line to be safe
            filtered_lines.append(line)
    
    return '\n'.join(filtered_lines)


def _compress_json_glossary(json_data, source_text):
    """
    Compress JSON glossary by excluding entries not found in source text.
    Handles both dict format and list format.
    """
    if isinstance(json_data, str):
        try:
            json_data = json.loads(json_data)
        except json.JSONDecodeError:
            return json_data
    
    if isinstance(json_data, dict):
        # Handle dict with 'entries' key
        if 'entries' in json_data:
            filtered_entries = {}
            for key, value in json_data['entries'].items():
                if _text_contains_term(source_text, key):
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
                elif _text_contains_term(source_text, key):
                    filtered_dict[key] = value
            return filtered_dict
    
    elif isinstance(json_data, list):
        # List of entry objects
        filtered_list = []
        for entry in json_data:
            if isinstance(entry, dict):
                # Check various possible keys for the raw term
                raw_term = entry.get('raw_name') or entry.get('original_name') or entry.get('original') or ''
                if raw_term and _text_contains_term(source_text, raw_term):
                    filtered_list.append(entry)
        return filtered_list
    
    return json_data


def _text_contains_term(text, term):
    """
    Check if term appears in text using regex for word boundary matching.
    Case-insensitive search.
    """
    if not term or not text:
        return False
    
    # Escape special regex characters in the term
    escaped_term = re.escape(term)
    
    # Use word boundaries for more accurate matching
    # \b doesn't work well with non-ASCII characters, so we use a more flexible pattern
    pattern = r'(?<![^\s\W])' + escaped_term + r'(?![^\s\W])'
    
    try:
        return bool(re.search(pattern, text, re.IGNORECASE | re.UNICODE))
    except re.error:
        # If regex fails, fall back to simple substring search
        return term.lower() in text.lower()


def compress_glossary_file(glossary_path, source_text):
    """
    Load, compress, and return glossary from file path.
    
    Args:
        glossary_path: Path to glossary file (.csv or .json)
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
        if glossary_path.lower().endswith('.csv'):
            return compress_glossary(content, source_text, glossary_format='csv')
        elif glossary_path.lower().endswith('.json'):
            json_data = json.loads(content)
            compressed_data = compress_glossary(json_data, source_text, glossary_format='json')
            # Return as JSON string
            return json.dumps(compressed_data, ensure_ascii=False, indent=2)
        else:
            return content
    except Exception as e:
        print(f"⚠️ Failed to compress glossary: {e}")
        return None
