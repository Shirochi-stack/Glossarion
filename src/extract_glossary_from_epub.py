import os
import json
import argparse
import zipfile
import time
import sys
import tiktoken
import threading
import queue
import ebooklib
import re
from ebooklib import epub
from chapter_splitter import ChapterSplitter

# Fix for PyInstaller - handle stdout reconfigure more carefully
if sys.platform.startswith("win"):
    try:
        # Try to reconfigure if the method exists
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        # If reconfigure doesn't work, try to set up UTF-8 another way
        import io
        import locale
        if sys.stdout and hasattr(sys.stdout, 'buffer'):
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

MODEL = os.getenv("MODEL", "gemini-1.5-flash")

# Parse token limit from environment variable (same logic as translation)
def parse_glossary_token_limit():
    """Parse token limit from environment variable"""
    env_value = os.getenv("GLOSSARY_TOKEN_LIMIT", "1000000").strip()
    
    if not env_value or env_value == "":
        return None, "unlimited"
    
    if env_value.lower() == "unlimited":
        return None, "unlimited"
    
    if env_value.isdigit() and int(env_value) > 0:
        limit = int(env_value)
        return limit, str(limit)
    
    # Default fallback
    return 1000000, "1000000 (default)"

MAX_GLOSSARY_TOKENS, GLOSSARY_LIMIT_STR = parse_glossary_token_limit()



# Global stop flag for GUI integration
_stop_requested = False

def set_stop_flag(value):
    """Set the global stop flag"""
    global _stop_requested
    _stop_requested = value

def is_stop_requested():
    """Check if stop was requested"""
    global _stop_requested
    return _stop_requested

# ─── resilient tokenizer setup ───
try:
    enc = tiktoken.encoding_for_model(MODEL)
except Exception:
    try:
        enc = tiktoken.get_encoding("cl100k_base")
    except Exception:
        enc = None

def count_tokens(text: str) -> int:
    if enc:
        return len(enc.encode(text))
    # crude fallback: assume ~1 token per 4 chars
    return max(1, len(text) // 4)

from ebooklib import epub
from bs4 import BeautifulSoup
from unified_api_client import UnifiedClient
from typing import List, Dict
import re
PROGRESS_FILE = "glossary_progress.json"

def set_output_redirect(log_callback=None):
    """Redirect print statements to a callback function for GUI integration"""
    if log_callback:
        import sys
        import io
        
        class CallbackWriter:
            def __init__(self, callback):
                self.callback = callback
                self.buffer = ""
                
            def write(self, text):
                if text.strip():
                    self.callback(text.strip())
                    
            def flush(self):
                pass
                
        sys.stdout = CallbackWriter(log_callback)

def load_config(path: str) -> Dict:
    with open(path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)

    # override context_limit_chapters if GUI passed GLOSSARY_CONTEXT_LIMIT
    env_limit = os.getenv("GLOSSARY_CONTEXT_LIMIT")
    if env_limit is not None:
        try:
            cfg['context_limit_chapters'] = int(env_limit)
        except ValueError:
            pass  # keep existing config value on parse error

    # override temperature if GUI passed GLOSSARY_TEMPERATURE
    env_temp = os.getenv("GLOSSARY_TEMPERATURE")
    if env_temp is not None:
        try:
            cfg['temperature'] = float(env_temp)
        except ValueError:
            pass  # keep existing config value on parse error

    return cfg

def save_progress(completed: List[int], glossary: List[Dict], context_history: List[Dict]):
    with open(PROGRESS_FILE, 'w', encoding='utf-8') as f:
        json.dump({"completed": completed, "glossary": glossary, "context_history": context_history}, f, ensure_ascii=False, indent=2)

def save_glossary_json(glossary: List[Dict], output_path: str):
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(glossary, f, ensure_ascii=False, indent=2)

def save_glossary_md(glossary: List[Dict], output_path: str):
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# Character Glossary\n\n")
        for char in glossary:
            f.write(f"## {char.get('name')} ({char.get('original_name')})\n")
            for key, val in char.items():
                if key not in ['name', 'original_name']:
                    f.write(f"- **{key}**: {val}\n")
            f.write("\n")

def extract_chapters_from_epub(epub_path: str) -> List[str]:
    chapters = []
    items = []
    
    # Add this helper function
    def is_html_document(item):
        """Check if an EPUB item is an HTML document"""
        if hasattr(item, 'media_type'):
            return item.media_type in [
                'application/xhtml+xml',
                'text/html',
                'application/html+xml',
                'text/xml'
            ]
        # Fallback for items that don't have media_type
        if hasattr(item, 'get_name'):
            name = item.get_name()
            return name.lower().endswith(('.html', '.xhtml', '.htm'))
        return False
    
    try:
        book = epub.read_epub(epub_path)
        # Replace the problematic line with media type checking
        items = [item for item in book.get_items() if is_html_document(item)]
    except Exception as e:
        print(f"[Warning] Manifest load failed, falling back to raw EPUB scan: {e}")
        try:
            with zipfile.ZipFile(epub_path, 'r') as zf:
                names = [n for n in zf.namelist() if n.lower().endswith(('.html', '.xhtml'))]
                for name in names:
                    try:
                        data = zf.read(name)
                        items.append(type('X', (), {
                            'get_content': lambda self, data=data: data,
                            'get_name': lambda self, name=name: name,
                            'media_type': 'text/html'  # Add media_type for consistency
                        })())
                    except Exception:
                        print(f"[Warning] Could not read zip file entry: {name}")
        except Exception as ze:
            print(f"[Fatal] Cannot open EPUB as zip: {ze}")
            return chapters
            
    for item in items:
        try:
            raw = item.get_content()
            soup = BeautifulSoup(raw, 'html.parser')
            text = soup.get_text("\n", strip=True)
            if text:
                chapters.append(text)
        except Exception as e:
            name = item.get_name() if hasattr(item, 'get_name') else repr(item)
            print(f"[Warning] Skipped corrupted chapter {name}: {e}")
            
    return chapters

def trim_context_history(history: List[Dict], limit: int, rolling_window: bool = False) -> List[Dict]:
    """
    Handle context history with either reset or rolling window mode
    
    Args:
        history: List of conversation history
        limit: Maximum number of exchanges to keep
        rolling_window: Whether to use rolling window mode
    """
    # Count current exchanges
    current_exchanges = len(history)
    
    # Handle based on mode
    if limit > 0 and current_exchanges >= limit:
        if rolling_window:
            # Rolling window: keep the most recent exchanges
            print(f"🔄 Rolling glossary context window: keeping last {limit} chapters")
            # Keep only the most recent exchanges
            history = history[-(limit-1):] if limit > 1 else []
        else:
            # Reset mode (original behavior)
            print(f"🔄 Reset glossary context after {limit} chapters")
            return []  # Return empty to reset context
    
    # Convert to message format
    trimmed = []
    for entry in history:
        trimmed.append({"role": "user", "content": entry["user"]})
        trimmed.append({"role": "assistant", "content": entry["assistant"]})
    return trimmed


def load_progress() -> Dict:
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"completed": [], "glossary": [], "context_history": []}

def dedupe_keep_order(old, new):
    seen = set()
    out = []
    for x in (old if isinstance(old, list) else [old]) + \
             (new if isinstance(new, list) else [new]):
        if isinstance(x, str):
            lx = x.lower()
            if lx not in seen:
                seen.add(lx)
                out.append(x)
    return out


# Add validation for extracted data with custom fields:
def validate_extracted_entry(entry):
    """Validate that extracted entry has required fields"""
    # Check if original_name field is enabled
    extract_original_name = os.getenv('GLOSSARY_EXTRACT_ORIGINAL_NAME', '1') == '1'
    
    # Validation rule 1: Must have an identifier
    # - If original_name is enabled, it must be present (original behavior)
    # - If original_name is disabled, must have 'name' field instead
    if extract_original_name:
        if 'original_name' not in entry or not entry['original_name']:
            return False
    else:
        # When original_name is disabled, 'name' becomes the required identifier
        if 'name' not in entry or not entry['name']:
            return False
    
    # Validation rule 2: Must have at least one content field
    # Build list of enabled content fields (exactly as original)
    enabled_fields = []
    
    if os.getenv('GLOSSARY_EXTRACT_NAME', '1') == '1':
        enabled_fields.append('name')
    if os.getenv('GLOSSARY_EXTRACT_GENDER', '1') == '1':
        enabled_fields.append('gender')
    if os.getenv('GLOSSARY_EXTRACT_TITLE', '1') == '1':
        enabled_fields.append('title')
    if os.getenv('GLOSSARY_EXTRACT_GROUP_AFFILIATION', '1') == '1':
        enabled_fields.append('group_affiliation')
    if os.getenv('GLOSSARY_EXTRACT_TRAITS', '1') == '1':
        enabled_fields.append('traits')
    if os.getenv('GLOSSARY_EXTRACT_HOW_THEY_REFER_TO_OTHERS', '1') == '1':
        enabled_fields.append('how_they_refer_to_others')
    if os.getenv('GLOSSARY_EXTRACT_LOCATIONS', '1') == '1':
        enabled_fields.append('locations')
    
    # Add custom fields (exactly as original)
    custom_fields_json = os.getenv('GLOSSARY_CUSTOM_FIELDS', '[]')
    try:
        custom_fields = json.loads(custom_fields_json)
        enabled_fields.extend(custom_fields)
    except:
        pass
    
    # Check for at least one content field (exactly as original)
    has_content = False
    for field in enabled_fields:
        if field in entry and entry[field]:
            has_content = True
            break
    
    return has_content

# Updated build_prompt function to handle custom prompts and fields:

def build_prompt(chapter_text: str) -> str:
    """
    Build the extraction prompt based on enabled fields and custom settings.
    Supports both custom prompts with placeholders and default prompts.
    """
    # Get custom prompt from environment or use default
    custom_prompt = os.getenv('GLOSSARY_SYSTEM_PROMPT', '').strip()
    
    # Check which fields are enabled via environment variables
    field_settings = {
        'original_name': os.getenv('GLOSSARY_EXTRACT_ORIGINAL_NAME', '1') == '1',
        'name': os.getenv('GLOSSARY_EXTRACT_NAME', '1') == '1',
        'gender': os.getenv('GLOSSARY_EXTRACT_GENDER', '1') == '1',
        'title': os.getenv('GLOSSARY_EXTRACT_TITLE', '1') == '1',
        'group_affiliation': os.getenv('GLOSSARY_EXTRACT_GROUP_AFFILIATION', '1') == '1',
        'traits': os.getenv('GLOSSARY_EXTRACT_TRAITS', '1') == '1',
        'how_they_refer_to_others': os.getenv('GLOSSARY_EXTRACT_HOW_THEY_REFER_TO_OTHERS', '1') == '1',
        'locations': os.getenv('GLOSSARY_EXTRACT_LOCATIONS', '1') == '1'
    }
    
    # Field descriptions for the prompt
    field_descriptions = {
        'original_name': "- original_name: name in the original script",
        'name': "- name: English/romanized name",
        'gender': "- gender",
        'title': "- title (with romanized suffix)",
        'group_affiliation': "- group_affiliation",
        'traits': "- traits",
        'how_they_refer_to_others': "- how_they_refer_to_others (mapping with romanized suffix)",
        'locations': "- locations: list of place names mentioned (include the original language in brackets)"
    }
    
    # Build field list based on enabled fields
    fields = []
    enabled_fields = []
    
    for field_name, is_enabled in field_settings.items():
        if is_enabled:
            fields.append(field_descriptions[field_name])
            enabled_fields.append(field_name)
    
    # Add custom fields
    custom_fields_json = os.getenv('GLOSSARY_CUSTOM_FIELDS', '[]')
    try:
        custom_fields = json.loads(custom_fields_json)
        for field in custom_fields:
            fields.append(f"- {field}")
            enabled_fields.append(field)
    except Exception as e:
        print(f"[Warning] Failed to parse custom fields: {e}")
    
    # Ensure we have at least one field to extract
    if not fields:
        # Fallback logic: try to enable at least one identifier field
        fallback_fields = ['name', 'original_name', 'title']
        for fallback in fallback_fields:
            if fallback in field_descriptions:
                fields.append(field_descriptions[fallback])
                enabled_fields.append(fallback)
                print(f"[Warning] No fields selected, defaulting to {fallback}")
                break
        
        # If still no fields, force original_name as absolute fallback
        if not fields:
            fields.append(field_descriptions['original_name'])
            enabled_fields.append('original_name')
            print("[Warning] No fields selected, forcing original_name as fallback")
    
    # Log which fields are enabled for debugging
    print(f"[DEBUG] Enabled extraction fields: {', '.join(enabled_fields)}")
    
    # Build the prompt
    if custom_prompt:
        # Use custom prompt with placeholders
        fields_str = '\n'.join(fields)
        prompt = custom_prompt
        
        # Replace placeholders (case-insensitive)
        prompt = prompt.replace('{fields}', fields_str)
        prompt = prompt.replace('{chapter_text}', chapter_text)
        
        # Also support alternative placeholder formats
        prompt = prompt.replace('{{fields}}', fields_str)
        prompt = prompt.replace('{{chapter_text}}', chapter_text)
        prompt = prompt.replace('{text}', chapter_text)
        prompt = prompt.replace('{{text}}', chapter_text)
        
        # Validate that placeholders were replaced
        if '{' in prompt and '}' in prompt:
            print("[Warning] Custom prompt may contain unreplaced placeholders")
        
        return prompt
    else:
        # Use default prompt structure
        fields_str = chr(10).join(fields)  # Using chr(10) for newline as in original
        
        prompt = f"""Output exactly a JSON array of objects and nothing else.
You are a glossary extractor for Korean, Japanese, or Chinese novels.
- Extract character information (e.g., name, traits), locations (countries, regions, cities), and translate them into English (romanization or equivalent).
- Romanize all untranslated honorifics and suffixes (e.g., 님 to '-nim', さん to '-san').
- all output must be in english, unless specified otherwise
For each character, provide JSON fields:
{fields_str}
Sort by appearance order; respond with a JSON array only.

Text:
{chapter_text}"""
        
        return prompt

# Updated merge_glossary_entries to handle custom fields:
def merge_glossary_entries(glossary):
    """
    Merge duplicate glossary entries based on configurable key field.
    Supports fuzzy matching for finding similar entries.
    """
    from difflib import SequenceMatcher
    
    merged = {}
    
    # Get list of custom fields
    custom_fields_json = os.getenv('GLOSSARY_CUSTOM_FIELDS', '[]')
    try:
        custom_fields_list = json.loads(custom_fields_json)
    except:
        custom_fields_list = []
    
    # Standard fields that use list merging
    list_fields = ['locations', 'traits', 'group_affiliation'] + custom_fields_list
    
    # Check configuration - try to load from config file first
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.json')
    duplicate_key_mode = 'auto'
    custom_field = ''
    fuzzy_threshold = 0.85
    
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                duplicate_key_mode = config.get('glossary_duplicate_key_mode', 
                                              os.getenv('GLOSSARY_DUPLICATE_KEY_MODE', 'auto'))
                custom_field = config.get('glossary_duplicate_custom_field',
                                        os.getenv('GLOSSARY_DUPLICATE_CUSTOM_FIELD', ''))
                fuzzy_threshold = float(config.get('glossary_fuzzy_threshold', 
                                                 os.getenv('GLOSSARY_FUZZY_THRESHOLD', '85'))) / 100.0
        else:
            duplicate_key_mode = os.getenv('GLOSSARY_DUPLICATE_KEY_MODE', 'auto')
            custom_field = os.getenv('GLOSSARY_DUPLICATE_CUSTOM_FIELD', '')
            fuzzy_threshold = float(os.getenv('GLOSSARY_FUZZY_THRESHOLD', '85')) / 100.0
    except:
        duplicate_key_mode = os.getenv('GLOSSARY_DUPLICATE_KEY_MODE', 'auto')
        custom_field = os.getenv('GLOSSARY_DUPLICATE_CUSTOM_FIELD', '')
        fuzzy_threshold = 0.85
    
    extract_original_name = os.getenv('GLOSSARY_EXTRACT_ORIGINAL_NAME', '1') == '1'
    
    # Track statistics for logging
    merge_stats = {
        'total_entries': len(glossary),
        'unique_entries': 0,
        'merged_entries': 0,
        'key_field_used': None,
        'fuzzy_matches': 0
    }
    
    # Helper function to get the key from an entry
    def get_entry_key(entry):
        key = None
        
        if duplicate_key_mode == 'original_name':
            key = entry.get('original_name')
            merge_stats['key_field_used'] = 'original_name'
        elif duplicate_key_mode == 'name':
            key = entry.get('name')
            merge_stats['key_field_used'] = 'name'
        elif duplicate_key_mode == 'custom':
            if custom_field:
                key = entry.get(custom_field)
                merge_stats['key_field_used'] = f'custom ({custom_field})'
            else:
                key = entry.get('original_name') or entry.get('name')
                merge_stats['key_field_used'] = 'fallback (no custom field specified)'
        elif duplicate_key_mode == 'auto' or duplicate_key_mode == 'fuzzy':
            if extract_original_name and 'original_name' in entry:
                key = entry['original_name']
                merge_stats['key_field_used'] = 'original_name (auto)'
            elif not extract_original_name and 'name' in entry:
                key = entry['name']
                merge_stats['key_field_used'] = 'name (auto)'
            else:
                key = entry.get('original_name') or entry.get('name')
                merge_stats['key_field_used'] = 'fallback'
        
        return str(key).strip() if key else None
    
    # Helper function to find fuzzy match
    def find_fuzzy_match(key, existing_keys):
        """Find the best fuzzy match for a key among existing keys"""
        if not key:
            return None
            
        best_match = None
        best_score = 0
        
        for existing_key in existing_keys:
            # Calculate similarity
            score = SequenceMatcher(None, key.lower(), existing_key.lower()).ratio()
            
            if score >= fuzzy_threshold and score > best_score:
                best_match = existing_key
                best_score = score
        
        if best_match and best_score < 1.0:  # Only count as fuzzy if not exact
            merge_stats['fuzzy_matches'] += 1
            print(f"🔍 Fuzzy match: '{key}' → '{best_match}' (similarity: {best_score:.2%})")
        
        return best_match
    
    # Helper function to calculate entry completeness score
    def calculate_entry_score(entry):
        """Calculate how complete/informative an entry is"""
        score = 0
        
        # Check each field
        if entry.get('original_name'):
            score += 2  # Original name is important
        if entry.get('name'):
            score += 2
        if entry.get('gender'):
            score += 1
        if entry.get('title'):
            score += 1
        
        # List fields - count non-empty lists
        for field in ['traits', 'locations', 'group_affiliation']:
            if entry.get(field) and len(entry[field]) > 0:
                score += len(entry[field])
        
        # How they refer to others
        if entry.get('how_they_refer_to_others'):
            score += len(entry.get('how_they_refer_to_others', {}))
        
        # Custom fields
        for field in custom_fields_list:
            if entry.get(field):
                if isinstance(entry[field], list):
                    score += len(entry[field])
                else:
                    score += 1
        
        return score
    
    # Process entries based on mode
    if duplicate_key_mode == 'fuzzy':
        print(f"🔍 Using fuzzy matching with threshold: {fuzzy_threshold:.0%}")
        
        # For fuzzy matching, we need to process entries one by one
        for entry in glossary:
            key = get_entry_key(entry)
            
            if not key:
                print(f"⚠️  Skipping entry without key field: {entry}")
                continue
            
            # Find best fuzzy match among existing keys
            existing_keys = list(merged.keys())
            match_key = find_fuzzy_match(key, existing_keys)
            
            if match_key:
                # Merge with existing entry
                merge_stats['merged_entries'] += 1
                existing_entry = merged[match_key]
                
                # Compare scores to decide which entry to keep as base
                existing_score = calculate_entry_score(existing_entry)
                new_score = calculate_entry_score(entry)
                
                if new_score > existing_score:
                    # New entry is more complete, use it as base but preserve the matched key
                    merged_entry = entry.copy()
                    # Merge in any missing fields from existing
                    for field in existing_entry:
                        if field not in merged_entry or not merged_entry[field]:
                            merged_entry[field] = existing_entry[field]
                    merged[match_key] = merged_entry
                else:
                    # Existing entry is more complete, merge new data into it
                    # Merge list fields
                    for field in list_fields:
                        old = existing_entry.get(field) or []
                        new = entry.get(field) or []
                        existing_entry[field] = dedupe_keep_order(old, new)
                    
                    # Merge how_they_refer_to_others
                    old_map = existing_entry.get('how_they_refer_to_others', {}) or {}
                    new_map = entry.get('how_they_refer_to_others', {}) or {}
                    for k, v in new_map.items():
                        if v is not None and k not in old_map:
                            old_map[k] = v
                    existing_entry['how_they_refer_to_others'] = old_map
                    
                    # Merge single-value fields (keep first non-None value)
                    single_value_fields = ['name', 'gender', 'title']
                    if extract_original_name:
                        single_value_fields.insert(0, 'original_name')
                    
                    for field in single_value_fields:
                        if field not in existing_entry or existing_entry.get(field) is None:
                            if field in entry and entry[field] is not None:
                                existing_entry[field] = entry[field]
            else:
                # No fuzzy match found, add as new entry
                merged[key] = entry.copy()
                merge_stats['unique_entries'] += 1
                
                # Initial normalize all list fields
                for field in list_fields:
                    if field in merged[key]:
                        merged[key][field] = dedupe_keep_order(
                            entry.get(field) or [], []
                        )
    
    else:
        # Non-fuzzy modes - use exact matching (original logic)
        for entry in glossary:
            key = get_entry_key(entry)
            
            if not key:
                print(f"⚠️  Skipping entry without key field: {entry}")
                continue
            
            if key not in merged:
                merged[key] = entry.copy()
                merge_stats['unique_entries'] += 1
                
                # Initial normalize all list fields
                for field in list_fields:
                    if field in merged[key]:
                        merged[key][field] = dedupe_keep_order(
                            entry.get(field) or [], []
                        )
            else:
                merge_stats['merged_entries'] += 1
                
                # Merge list fields
                for field in list_fields:
                    old = merged[key].get(field) or []
                    new = entry.get(field) or []
                    merged[key][field] = dedupe_keep_order(old, new)
                
                # Merge how_they_refer_to_others
                old_map = merged[key].get('how_they_refer_to_others', {}) or {}
                new_map = entry.get('how_they_refer_to_others', {}) or {}
                for k, v in new_map.items():
                    if v is not None and k not in old_map:
                        old_map[k] = v
                merged[key]['how_they_refer_to_others'] = old_map
                
                # Merge single-value fields (keep first non-None value)
                single_value_fields = ['name', 'gender', 'title']
                if extract_original_name:
                    single_value_fields.insert(0, 'original_name')
                
                for field in single_value_fields:
                    if field not in merged[key] or merged[key].get(field) is None:
                        if field in entry and entry[field] is not None:
                            merged[key][field] = entry[field]
    
    # Strip out any None fields
    for entry in merged.values():
        # Remove None values from all fields
        for field in list(entry.keys()):
            if entry.get(field) is None:
                entry.pop(field, None)
        
        # Sanitize how_they_refer_to_others
        htr = entry.get('how_they_refer_to_others')
        if isinstance(htr, dict):
            entry['how_they_refer_to_others'] = {
                k: v for k, v in htr.items() if v is not None
            }
            # Remove empty dict
            if not entry['how_they_refer_to_others']:
                entry.pop('how_they_refer_to_others', None)
        else:
            entry.pop('how_they_refer_to_others', None)
    
    # Log merge statistics
    if merge_stats['merged_entries'] > 0 or merge_stats['fuzzy_matches'] > 0:
        print(f"🔀 Merged {merge_stats['merged_entries']} duplicate entries")
        if duplicate_key_mode == 'fuzzy':
            print(f"   Fuzzy matches: {merge_stats['fuzzy_matches']}")
            print(f"   Threshold: {fuzzy_threshold:.0%}")
        print(f"   Key field: {merge_stats['key_field_used']}")
        print(f"   Total entries: {merge_stats['total_entries']} → {merge_stats['unique_entries']}")
    
    return list(merged.values())

# Update main function to log custom fields:

def main(log_callback=None, stop_callback=None):
    """Modified main function that can accept a logging callback and stop callback"""
    if log_callback:
        set_output_redirect(log_callback)
    
    # Set up stop checking
    def check_stop():
        if stop_callback and stop_callback():
            print("❌ Glossary extraction stopped by user request.")
            return True
        return is_stop_requested()
        
    start = time.time()
    
    # Handle both command line and GUI calls
    if '--epub' in sys.argv:
        # Command line mode
        parser = argparse.ArgumentParser(description='Extract glossary from EPUB/TXT')
        parser.add_argument('--epub', required=True, help='Path to EPUB/TXT file')
        parser.add_argument('--output', required=True, help='Output glossary path')
        parser.add_argument('--config', help='Config file path')
        # keep any other add_argument lines you have
        
        args = parser.parse_args()
        epub_path = args.epub
    else:
        # GUI mode - get from environment
        epub_path = os.getenv("EPUB_PATH", "")
        if not epub_path and len(sys.argv) > 1:
            epub_path = sys.argv[1]


    is_text_file = epub_path.lower().endswith('.txt')
    
    if is_text_file:
        # Import text processor
        from extract_glossary_from_txt import extract_chapters_from_txt
        chapters = extract_chapters_from_txt(epub_path)
        file_base = os.path.splitext(os.path.basename(epub_path))[0]
    else:
        # Existing EPUB code
        chapters = extract_chapters_from_epub(epub_path)
        epub_base = os.path.splitext(os.path.basename(epub_path))[0]
        file_base = epub_base

    # If user didn't override --output, derive it from the EPUB filename:
    if args.output == 'glossary.json':
        args.output = f"{file_base}_glossary.json" 

    # ensure we have a Glossary subfolder next to the JSON/MD outputs
    glossary_dir = os.path.join(os.path.dirname(args.output), "Glossary")
    os.makedirs(glossary_dir, exist_ok=True)

    # override the module‐level PROGRESS_FILE to include epub name
    global PROGRESS_FILE
    PROGRESS_FILE = os.path.join(
        glossary_dir,
        f"{file_base}_glossary_progress.json"  # CHANGED from epub_base
    )

    config = load_config(args.config)
    
    # Get API key from environment variables (set by GUI) or config file
    api_key = (os.getenv("API_KEY") or 
               os.getenv("OPENAI_API_KEY") or 
               os.getenv("OPENAI_OR_Gemini_API_KEY") or
               os.getenv("GEMINI_API_KEY") or
               config.get('api_key'))

    # Get model from environment or config
    model = os.getenv("MODEL") or config.get('model', 'gemini-1.5-flash')

    # Define output directory (use current directory as default)
    out = os.path.dirname(args.output) if hasattr(args, 'output') else os.getcwd()

    # Use the variables we just retrieved
    client = UnifiedClient(api_key=api_key, model=model, output_dir=out)
    
    #API call delay
    api_delay = float(os.getenv("SEND_INTERVAL_SECONDS", "2"))
    print(f"⏱️  API call delay: {api_delay} seconds")
    
    # Get compression factor from environment
    compression_factor = float(os.getenv("COMPRESSION_FACTOR", "1.0"))
    print(f"📐 Compression Factor: {compression_factor}")

    # Initialize chapter splitter with compression factor
    chapter_splitter = ChapterSplitter(model_name=model, compression_factor=compression_factor)
    
    # Get temperature from environment or config
    temp = float(os.getenv("GLOSSARY_TEMPERATURE") or config.get('temperature', 0.1))
    
    env_max_output = os.getenv("MAX_OUTPUT_TOKENS")
    if env_max_output and env_max_output.isdigit():
        mtoks = int(env_max_output)
        print(f"[DEBUG] Output Token Limit: {mtoks} (from GUI)")
    else:
        mtoks = config.get('max_tokens', 4196)
        print(f"[DEBUG] Output Token Limit: {mtoks} (from config)")
    
    sys_prompt = config.get('system_prompt', 'You are a helpful assistant.')
    
    # Get context limit from environment or config
    ctx_limit = int(os.getenv("GLOSSARY_CONTEXT_LIMIT") or config.get('context_limit_chapters', 3))

    # Parse chapter range from environment
    chapter_range = os.getenv("CHAPTER_RANGE", "").strip()
    range_start = None
    range_end = None
    if chapter_range and re.match(r"^\d+\s*-\s*\d+$", chapter_range):
        range_start, range_end = map(int, chapter_range.split("-", 1))
        print(f"📊 Chapter Range Filter: {range_start} to {range_end}")
    elif chapter_range:
        print(f"⚠️ Invalid chapter range format: {chapter_range} (use format: 5-10)")

    # Log enabled fields
    print("📑 Extraction Fields Configuration:")
    original_name_enabled = os.environ.get('GLOSSARY_EXTRACT_ORIGINAL_NAME', '1') == '1'
    status = "✅" if original_name_enabled else "❌"
    print(f"   • Original Name: {status}")
    print(f"   • Name: {'✅' if os.getenv('GLOSSARY_EXTRACT_NAME', '1') == '1' else '❌'}")
    print(f"   • Gender: {'✅' if os.getenv('GLOSSARY_EXTRACT_GENDER', '1') == '1' else '❌'}")
    print(f"   • Title: {'✅' if os.getenv('GLOSSARY_EXTRACT_TITLE', '1') == '1' else '❌'}")
    print(f"   • Group: {'✅' if os.getenv('GLOSSARY_EXTRACT_GROUP_AFFILIATION', '1') == '1' else '❌'}")
    print(f"   • Traits: {'✅' if os.getenv('GLOSSARY_EXTRACT_TRAITS', '1') == '1' else '❌'}")
    print(f"   • References: {'✅' if os.getenv('GLOSSARY_EXTRACT_HOW_THEY_REFER_TO_OTHERS', '1') == '1' else '❌'}")
    print(f"   • Locations: {'✅' if os.getenv('GLOSSARY_EXTRACT_LOCATIONS', '1') == '1' else '❌'}")
    
    # Log custom fields
    custom_fields_json = os.getenv('GLOSSARY_CUSTOM_FIELDS', '[]')
    try:
        custom_fields = json.loads(custom_fields_json)
        if custom_fields:
            print(f"   • Custom Fields: {', '.join(custom_fields)}")
    except:
        pass
    
    # Check if custom prompt is being used
    if os.getenv('GLOSSARY_SYSTEM_PROMPT'):
        print("📑 Using custom extraction prompt")
    else:
        print("📑 Using default extraction prompt")

    if is_text_file:
        from extract_glossary_from_txt import extract_chapters_from_txt
        chapters = extract_chapters_from_txt(args.epub)
    else:
        chapters = extract_chapters_from_epub(args.epub)
    
    # The rest of the function remains exactly the same
    if not chapters:
        print("No chapters found. Exiting.")
        return
        
    if not chapters:
        print("No chapters found. Exiting.")
        return

    # Check for stop before starting processing
    if check_stop():
        return

    prog = load_progress()
    completed = prog['completed']
    glossary = prog['glossary']
    history = prog['context_history']
    total_chapters = len(chapters) 
    
    # Count chapters that will be processed with range filter
    chapters_to_process = 0
    for idx, chap in enumerate(chapters):
        # Skip if chapter is outside the range
        if range_start is not None and range_end is not None:
            # Get chapter number - assuming chapters have a 'num' field or use idx+1
            chapter_num = idx + 1  # 1-based chapter numbering
            if not (range_start <= chapter_num <= range_end):
                continue
        chapters_to_process += 1
    
    if chapters_to_process < total_chapters:
        print(f"📊 Processing {chapters_to_process} out of {total_chapters} chapters due to range filter")
    
    for idx, chap in enumerate(chapters):
        # Check for stop at the beginning of each chapter
        if check_stop():
            print(f"❌ Glossary extraction stopped at chapter {idx+1}")
            return
        
        # Apply chapter range filter
        if range_start is not None and range_end is not None:
            chapter_num = idx + 1  # 1-based chapter numbering
            if not (range_start <= chapter_num <= range_end):
                # Check if this is from a text file
                is_text_chapter = hasattr(chap, 'filename') and chap.get('filename', '').endswith('.txt')
                terminology = "Section" if is_text_chapter else "Chapter"
                print(f"[SKIP] {terminology} {chapter_num} - outside range filter")
                continue
            
        if idx in completed:
            # Check if processing text file chapters
            is_text_chapter = hasattr(chap, 'filename') and chap.get('filename', '').endswith('.txt')
            terminology = "section" if is_text_chapter else "chapter"
            print(f"Skipping {terminology} {idx+1} (already processed)")
            continue
                
        print(f"🔄 Processing Chapter {idx+1}/{total_chapters}")
        # Get both settings
        contextual_enabled = os.getenv('CONTEXTUAL', '1') == '1'
        rolling_window = os.getenv('GLOSSARY_HISTORY_ROLLING', '0') == '1'

        # Check if history will reset on this chapter
        if contextual_enabled and len(history) >= ctx_limit and ctx_limit > 0 and not rolling_window:
            print(f"  📌 Glossary context will reset after this chapter (current: {len(history)}/{ctx_limit} chapters)")        

        try:
            if not contextual_enabled:
                # No context at all
                msgs = [{"role":"system","content":sys_prompt}] \
                     + [{"role":"user","content":build_prompt(chap)}]
            else:
                # Use context with trim_context_history handling the mode
                msgs = [{"role":"system","content":sys_prompt}] \
                     + trim_context_history(history, ctx_limit, rolling_window) \
                     + [{"role":"user","content":build_prompt(chap)}]
            
            total_tokens = sum(count_tokens(m["content"]) for m in msgs)
            
            # READ THE TOKEN LIMIT
            env_value = os.getenv("MAX_INPUT_TOKENS", "1000000").strip()
            if not env_value or env_value == "":
                token_limit = None
                limit_str = "unlimited"
            elif env_value.isdigit() and int(env_value) > 0:
                token_limit = int(env_value)
                limit_str = str(token_limit)
            else:
                token_limit = 1000000
                limit_str = "1000000 (default)"
            
            print(f"[DEBUG] Glossary prompt tokens = {total_tokens} / {limit_str}")
            
            # Check if we're over the token limit and need to split
            if token_limit is not None and total_tokens > token_limit:
                print(f"⚠️ Chapter {idx+1} exceeds token limit: {total_tokens} > {token_limit}")
                print(f"📄 Using ChapterSplitter to split into smaller chunks...")
                
                # Calculate available tokens for content
                system_tokens = chapter_splitter.count_tokens(sys_prompt)
                context_tokens = sum(chapter_splitter.count_tokens(m["content"]) for m in trim_context_history(history, ctx_limit, rolling_window))
                safety_margin = 1000
                available_tokens = token_limit - system_tokens - context_tokens - safety_margin
                
                # Since glossary extraction works with plain text, wrap it in a simple HTML structure
                chapter_html = f"<html><body><p>{chap.replace(chr(10)+chr(10), '</p><p>')}</p></body></html>"
                
                # Use ChapterSplitter to split the chapter
                chunks = chapter_splitter.split_chapter(chapter_html, available_tokens)
                print(f"📄 Chapter split into {len(chunks)} chunks")
                
                # Process each chunk
                for chunk_html, chunk_idx, total_chunks in chunks:
                    print(f"🔄 Processing chunk {chunk_idx}/{total_chunks} of Chapter {idx+1}")
                    
                    # Extract text from the chunk HTML
                    from bs4 import BeautifulSoup
                    soup = BeautifulSoup(chunk_html, 'html.parser')
                    chunk_text = soup.get_text(strip=True)
            
            # Check for stop before API call
            if check_stop():
                print(f"❌ Glossary extraction stopped before API call for chapter {idx+1}")
                return
            
            # API call with stop checking
            result_queue = queue.Queue()
            api_error = None
            
            def api_call():
                try:
                    result = client.send(msgs, temperature=temp, max_tokens=mtoks, context='glossary')
                    result_queue.put(result)
                except Exception as e:
                    result_queue.put(e)
            
            api_thread = threading.Thread(target=api_call)
            api_thread.daemon = True
            api_thread.start()
            
            # Check for stop while waiting for API
            timeout = 300  # 5 minute total timeout
            check_interval = 0.5
            elapsed = 0
            
            while elapsed < timeout:
                try:
                    # Try to get result with short timeout
                    result = result_queue.get(timeout=check_interval)
                    if isinstance(result, Exception):
                        raise result
                    raw = result
                    break
                except queue.Empty:
                    # No result yet, check if stop requested
                    if check_stop():
                        print(f"❌ Glossary extraction stopped during API call for chapter {idx+1}")
                        return
                    elapsed += check_interval
            else:
                # Timeout reached
                print(f"⚠️ API call timed out after {timeout} seconds")
                continue
                
            # Process response (rest of the code remains the same)
            resp = raw[0] if isinstance(raw, tuple) else raw

            # Save the raw response
            os.makedirs("Payloads", exist_ok=True)
            with open(f"Payloads/failed_response_chap{idx+1}.txt", "w", encoding="utf-8", errors="replace") as f:
                f.write(resp)

            # Extract JSON
            m = re.search(r"\[.*\]", resp, re.DOTALL)
            if not m:
                print(f"[Warning] Couldn't find JSON array in chapter {idx+1}, saving raw…")
                continue

            json_str = m.group(0) if m else resp

            # Parse JSON and validate entries
            try:
                data = json.loads(json_str)
                
                # Filter out invalid entries
                valid_data = []
                for entry in data:
                    if validate_extracted_entry(entry):
                        valid_data.append(entry)
                    else:
                        print(f"[Debug] Skipped invalid entry: {entry.get('original_name', 'unknown')}")
                
                data = valid_data
                total_ent = len(data)
                
                # Log entries
                for eidx, entry in enumerate(data, start=1):
                    if check_stop():
                        print(f"❌ Glossary extraction stopped during entry processing for chapter {idx+1}")
                        return
                        
                    elapsed = time.time() - start
                    if idx == 0 and eidx == 1:
                        eta = 0
                    else:
                        avg = elapsed / ((idx * 100) + eidx)
                        eta = avg * (total_chapters * 100 - ((idx * 100) + eidx))
                    name = entry.get("original_name","?")
                    print(f'[Chapter {idx+1}/{total_chapters}] [{eidx}/{total_ent}] ({elapsed:.1f}s elapsed, ETA {eta:.1f}s) → Entry "{name}"')
            except json.JSONDecodeError as e:
                print(f"[Warning] JSON decode error chap {idx+1}: {e}")
                continue    
                
            # Merge and save
            glossary.extend(data)
            glossary[:] = merge_glossary_entries(glossary)
            completed.append(idx)

            # Only add to history if contextual is enabled
            if contextual_enabled:
                history.append({"user": build_prompt(chap), "assistant": resp})
                
                # Reset history when limit reached without rolling window
                if not rolling_window and len(history) >= ctx_limit and ctx_limit > 0:
                    print(f"🔄 Resetting glossary context (reached {ctx_limit} chapter limit)")
                    history = []
                    prog['context_history'] = []

            save_progress(completed, glossary, history)
            save_glossary_json(glossary, os.path.join(glossary_dir, os.path.basename(args.output)))
            save_glossary_md(glossary, os.path.join(glossary_dir, os.path.basename(args.output).replace('.json', '.md')))
            
            # Add delay before next API call (but not after the last chapter)
            if idx < len(chapters) - 1:
                # Check if we're within the range or if there are more chapters to process
                next_chapter_in_range = True
                if range_start is not None and range_end is not None:
                    next_chapter_num = idx + 2  # idx+1 is current, idx+2 is next
                    next_chapter_in_range = (range_start <= next_chapter_num <= range_end)
                else:
                    # No range filter, check if next chapter is already completed
                    next_chapter_in_range = (idx + 1) not in completed
                
                if next_chapter_in_range:
                    print(f"⏱️  Waiting {api_delay}s before next chapter...")
                    
                    # Use interruptible sleep
                    sleep_elapsed = 0
                    sleep_interval = 0.5
                    while sleep_elapsed < api_delay:
                        if check_stop():
                            print(f"❌ Glossary extraction stopped during delay")
                            return
                        time.sleep(min(sleep_interval, api_delay - sleep_elapsed))
                        sleep_elapsed += sleep_interval
                        
            # Check for stop after processing chapter
            if check_stop():
                print(f"❌ Glossary extraction stopped after processing chapter {idx+1}")
                return

        except Exception as e:
            print(f"Error at chapter {idx+1}: {e}")
            # Check for stop even after error
            if check_stop():
                print(f"❌ Glossary extraction stopped after error in chapter {idx+1}")
                return

    print(f"Done. Glossary saved to {args.output}")

if __name__=='__main__':
    main()
