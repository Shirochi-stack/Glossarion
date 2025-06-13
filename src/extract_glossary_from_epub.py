import os
import json
import argparse
import zipfile
import time
import sys
import tiktoken
import threading
import queue

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
    try:
        book = epub.read_epub(epub_path)
        items = [item for item in book.get_items() if item.get_type() == epub.ITEM_DOCUMENT]
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
                            'get_type': lambda self: epub.ITEM_DOCUMENT
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

def trim_context_history(history: List[Dict], limit: int) -> List[Dict]:
    """Reset history when limit is reached instead of trimming"""
    # Count current exchanges
    current_exchanges = len(history)
    
    # Reset when limit is reached
    if limit > 0 and current_exchanges >= limit:
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
                out.append(lx)
    return out


# Add validation for extracted data with custom fields:
def validate_extracted_entry(entry):
    """Validate that extracted entry has required fields"""
    # original_name is always required
    if 'original_name' not in entry or not entry['original_name']:
        return False
    
    # Get enabled fields
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
    
    # Add custom fields
    custom_fields_json = os.getenv('GLOSSARY_CUSTOM_FIELDS', '[]')
    try:
        custom_fields = json.loads(custom_fields_json)
        enabled_fields.extend(custom_fields)
    except:
        pass
    
    # Entry should have at least one other field besides original_name
    has_content = False
    for field in enabled_fields:
        if field in entry and entry[field]:
            has_content = True
            break
    
    return has_content

# Updates for extract_glossary_from_epub.py

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
    merged = {}
    
    # Get list of custom fields
    custom_fields_json = os.getenv('GLOSSARY_CUSTOM_FIELDS', '[]')
    try:
        custom_fields_list = json.loads(custom_fields_json)
    except:
        custom_fields_list = []
    
    # Standard fields that use list merging
    list_fields = ['locations', 'traits', 'group_affiliation'] + custom_fields_list
    
    for entry in glossary:
        # Check if original_name field is enabled
        extract_original_name = os.getenv('GLOSSARY_EXTRACT_ORIGINAL_NAME', '1') == '1'
        
        # Determine the key to use for merging
        key = None
        if extract_original_name and 'original_name' in entry:
            key = entry['original_name']
        elif not extract_original_name and 'name' in entry:
            key = entry['name']  # Use name as key if original_name is disabled
        elif 'original_name' in entry:  # Fallback to original_name if it exists
            key = entry['original_name']
        elif 'name' in entry:  # Fallback to name
            key = entry['name']
        else:
            # Skip entries without any identifier
            continue
            
        if key not in merged:
            merged[key] = entry.copy()
            # Initial normalize all list fields
            for field in list_fields:
                if field in merged[key]:
                    merged[key][field] = dedupe_keep_order(
                        entry.get(field) or [], []
                    )
        else:
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
    
    return list(merged.values())

# Add validation for extracted data with custom fields:

def validate_extracted_entry(entry):
    """Validate that extracted entry has required fields"""
    # original_name is always required
    if 'original_name' not in entry or not entry['original_name']:
        return False
    
    # Get enabled fields
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
    
    # Add custom fields
    custom_fields_json = os.getenv('GLOSSARY_CUSTOM_FIELDS', '[]')
    try:
        custom_fields = json.loads(custom_fields_json)
        enabled_fields.extend(custom_fields)
    except:
        pass
    
    # Entry should have at least one other field besides original_name
    has_content = False
    for field in enabled_fields:
        if field in entry and entry[field]:
            has_content = True
            break
    
    return has_content

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
    parser = argparse.ArgumentParser(description="Extract character glossary from EPUB via ChatGPT")
    parser.add_argument('--epub',   required=True)
    parser.add_argument('--output', default='glossary.json',
                        help="Name of the JSON file to write.  (Defaults to <epub_basename>_glossary.json)")
    parser.add_argument('--config', default='config.json')
    args = parser.parse_args()

    epub_base = os.path.splitext(os.path.basename(args.epub))[0]

    # If user didn't override --output, derive it from the EPUB filename:
    if args.output == 'glossary.json':
        args.output = f"{epub_base}_glossary.json"

    # ensure we have a Glossary subfolder next to the JSON/MD outputs
    glossary_dir = os.path.join(os.path.dirname(args.output), "Glossary")
    os.makedirs(glossary_dir, exist_ok=True)

    # override the module‐level PROGRESS_FILE to include epub name
    global PROGRESS_FILE
    PROGRESS_FILE = os.path.join(
        glossary_dir,
        f"{epub_base}_glossary_progress.json"
    )

    config = load_config(args.config)
    # Get model from environment variable (set by GUI) or config file
    model = os.getenv("MODEL") or config.get('model', 'gemini-1.5-flash')
    # Get API key from environment variables (set by GUI) or config file
    api_key = (os.getenv("API_KEY") or 
               os.getenv("OPENAI_API_KEY") or 
               os.getenv("OPENAI_OR_Gemini_API_KEY") or
               os.getenv("GEMINI_API_KEY") or
               config.get('api_key'))

    client = UnifiedClient(model=model, api_key=api_key)
    
    # Get temperature from environment or config
    temp = float(os.getenv("GLOSSARY_TEMPERATURE") or config.get('temperature', 0.3))
    
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

    # Log enabled fields
    print("📑 Extraction Fields Configuration:")
    print(f"   • Original Name: ✅ (always enabled)")
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

    chapters = extract_chapters_from_epub(args.epub)
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
    
    for idx, chap in enumerate(chapters):
        # Check for stop at the beginning of each chapter
        if check_stop():
            print(f"❌ Glossary extraction stopped at chapter {idx+1}")
            return
            
        if idx in completed:
            print(f"Skipping chapter {idx+1} (already processed)")
            continue
                
        print(f"🔄 Processing Chapter {idx+1}/{total_chapters}")
        # Check if history will reset on this chapter
        if len(history) >= ctx_limit and ctx_limit > 0:
            print(f"  📌 Glossary context will reset after this chapter (current: {len(history)}/{ctx_limit} chapters)")        
        try:
            msgs = [{"role":"system","content":sys_prompt}] \
                 + trim_context_history(history, ctx_limit) \
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
            
            # Check if we're over the token limit
            if token_limit is not None and total_tokens > token_limit:
                print(f"⚠️ Chapter {idx+1} exceeds token limit: {total_tokens} > {token_limit}")
                print(f"⚠️ Skipping chapter {idx+1} due to token limit")
                completed.append(idx)
                save_progress(completed, glossary, history)
                continue
            
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
            history.append({"user": build_prompt(chap), "assistant": resp})
            save_progress(completed, glossary, history)
            save_glossary_json(glossary, os.path.join(glossary_dir, os.path.basename(args.output)))
            save_glossary_md(glossary, os.path.join(glossary_dir, os.path.basename(args.output).replace('.json', '.md')))

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
