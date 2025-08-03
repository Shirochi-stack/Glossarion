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
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple
from unified_api_client import UnifiedClient, UnifiedClientError

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

MODEL = os.getenv("MODEL", "gemini-2.0-flash")

def interruptible_sleep(duration, check_stop_fn, interval=0.1):
    """Sleep that can be interrupted by stop request"""
    elapsed = 0
    while elapsed < duration:
        if check_stop_fn():
            return False  # Interrupted
        sleep_time = min(interval, duration - elapsed)
        time.sleep(sleep_time)
        elapsed += sleep_time
    return True  # Completed normally

def cancel_all_futures(futures):
    """Cancel all pending futures immediately"""
    cancelled_count = 0
    for future in futures:
        if not future.done() and future.cancel():
            cancelled_count += 1
    return cancelled_count

def send_with_interrupt(messages, client, temperature, max_tokens, stop_check_fn, chunk_timeout=None):
    """Send API request with interrupt capability and optional timeout retry"""
    result_queue = queue.Queue()
    
    def api_call():
        try:
            start_time = time.time()
            result = client.send(messages, temperature=temperature, max_tokens=max_tokens, context='glossary')
            elapsed = time.time() - start_time
            result_queue.put((result, elapsed))
        except Exception as e:
            result_queue.put(e)
    
    api_thread = threading.Thread(target=api_call)
    api_thread.daemon = True
    api_thread.start()
    
    timeout = chunk_timeout if chunk_timeout is not None else 86400
    check_interval = 0.1  # Reduced from 0.5 to 0.1 for faster response
    elapsed = 0
    
    while elapsed < timeout:
        try:
            # Check for results with shorter timeout
            result = result_queue.get(timeout=check_interval)
            if isinstance(result, Exception):
                raise result
            if isinstance(result, tuple):
                api_result, api_time = result
                if chunk_timeout and api_time > chunk_timeout:
                    if hasattr(client, '_in_cleanup'):
                        client._in_cleanup = True
                    if hasattr(client, 'cancel_current_operation'):
                        client.cancel_current_operation()
                    raise UnifiedClientError(f"API call took {api_time:.1f}s (timeout: {chunk_timeout}s)")
                return api_result
            return result
        except queue.Empty:
            if stop_check_fn():
                # More aggressive cancellation
                print("🛑 Stop requested - cancelling API call immediately...")
                
                # Set cleanup flag
                if hasattr(client, '_in_cleanup'):
                    client._in_cleanup = True
                
                # Try to cancel the operation
                if hasattr(client, 'cancel_current_operation'):
                    client.cancel_current_operation()
                
                # Don't wait for the thread to finish - just raise immediately
                raise UnifiedClientError("Glossary extraction stopped by user")
            
            elapsed += check_interval
    
    # Timeout occurred
    if hasattr(client, '_in_cleanup'):
        client._in_cleanup = True
    if hasattr(client, 'cancel_current_operation'):
        client.cancel_current_operation()
    raise UnifiedClientError(f"API call timed out after {timeout} seconds")

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

def remove_honorifics(name):
    """Remove common honorifics from names"""
    if not name:
        return name
    
    # Check if honorifics filtering is disabled
    if os.getenv('GLOSSARY_DISABLE_HONORIFICS_FILTER', '0') == '1':
        return name.strip()
    
    # Modern Korean honorifics
    korean_honorifics = [
        '님', '씨', '씨는', '군', '양', '선생님', '선생', '사장님', '사장', 
        '과장님', '과장', '대리님', '대리', '주임님', '주임', '이사님', '이사',
        '부장님', '부장', '차장님', '차장', '팀장님', '팀장', '실장님', '실장',
        '교수님', '교수', '박사님', '박사', '원장님', '원장', '회장님', '회장',
        '소장님', '소장', '전무님', '전무', '상무님', '상무', '이사장님', '이사장'
    ]
    
    # Archaic/Historical Korean honorifics
    korean_archaic = [
        '공', '옹', '어른', '나리', '나으리', '대감', '영감', '마님', '마마',
        '대군', '군', '옹주', '공주', '왕자', '세자', '영애', '영식', '도령',
        '낭자', '낭군', '서방', '영감님', '대감님', '마님', '아씨', '도련님',
        '아가씨', '나으리', '진사', '첨지', '영의정', '좌의정', '우의정',
        '판서', '참판', '정승', '대원군'
    ]
    
    # Modern Japanese honorifics
    japanese_honorifics = [
        'さん', 'さま', '様', 'くん', '君', 'ちゃん', 'せんせい', '先生',
        'どの', '殿', 'たん', 'ぴょん', 'ぽん', 'ちん', 'りん', 'せんぱい',
        '先輩', 'こうはい', '後輩', 'し', '氏', 'ふじん', '夫人', 'かちょう',
        '課長', 'ぶちょう', '部長', 'しゃちょう', '社長'
    ]
    
    # Archaic/Historical Japanese honorifics
    japanese_archaic = [
        'どの', '殿', 'たいゆう', '大夫', 'きみ', '公', 'あそん', '朝臣',
        'おみ', '臣', 'むらじ', '連', 'みこと', '命', '尊', 'ひめ', '姫',
        'みや', '宮', 'おう', '王', 'こう', '侯', 'はく', '伯', 'し', '子',
        'だん', '男', 'じょ', '女', 'ひこ', '彦', 'ひめみこ', '姫御子',
        'すめらみこと', '天皇', 'きさき', '后', 'みかど', '帝'
    ]
    
    # Modern Chinese honorifics
    chinese_honorifics = [
        '先生', '女士', '小姐', '老师', '师傅', '大人', '公', '君', '总',
        '老总', '老板', '经理', '主任', '处长', '科长', '股长', '教授',
        '博士', '院长', '校长', '同志', '师兄', '师姐', '师弟', '师妹',
        '学长', '学姐', '前辈', '阁下'
    ]
    
    # Archaic/Historical Chinese honorifics
    chinese_archaic = [
        '公', '侯', '伯', '子', '男', '王', '君', '卿', '大夫', '士',
        '陛下', '殿下', '阁下', '爷', '老爷', '大人', '夫人', '娘娘',
        '公子', '公主', '郡主', '世子', '太子', '皇上', '皇后', '贵妃',
        '娘子', '相公', '官人', '郎君', '小姐', '姑娘', '公公', '嬷嬷',
        '大侠', '少侠', '前辈', '晚辈', '在下', '足下', '兄台', '仁兄',
        '贤弟', '老夫', '老朽', '本座', '本尊', '真人', '上人', '尊者'
    ]
    
    # Combine all honorifics
    all_honorifics = (
        korean_honorifics + korean_archaic +
        japanese_honorifics + japanese_archaic +
        chinese_honorifics + chinese_archaic
    )
    
    # Remove honorifics from the end of the name
    name_cleaned = name.strip()
    
    # Sort by length (longest first) to avoid partial matches
    sorted_honorifics = sorted(all_honorifics, key=len, reverse=True)
    
    for honorific in sorted_honorifics:
        if name_cleaned.endswith(honorific):
            name_cleaned = name_cleaned[:-len(honorific)].strip()
            # Only remove one honorific per pass
            break
    
    return name_cleaned

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
    """Save glossary in the new simple format"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(glossary, f, ensure_ascii=False, indent=2)

def save_glossary_csv(glossary: List[Dict], output_path: str):
    """Save glossary in CSV format exactly like the example"""
    import csv
    
    csv_path = output_path.replace('.json', '.csv')
    
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        # Write without header, just like the example
        for entry in glossary:
            if entry['type'] == 'character':
                f.write(f"character,{entry['raw_name']},{entry['translated_name']},{entry.get('gender', '')}\n")
            else:
                f.write(f"term,{entry['raw_name']},{entry['translated_name']},\n")

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
        # Add stop check before reading
        if is_stop_requested():
            return []
            
        book = epub.read_epub(epub_path)
        # Replace the problematic line with media type checking
        items = [item for item in book.get_items() if is_html_document(item)]
    except Exception as e:
        print(f"[Warning] Manifest load failed, falling back to raw EPUB scan: {e}")
        try:
            with zipfile.ZipFile(epub_path, 'r') as zf:
                names = [n for n in zf.namelist() if n.lower().endswith(('.html', '.xhtml'))]
                for name in names:
                    # Add stop check in loop
                    if is_stop_requested():
                        return chapters
                        
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
        # Add stop check before processing each chapter
        if is_stop_requested():
            return chapters
            
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

def parse_api_response(response_text: str) -> List[Dict]:
    """Parse API response to extract glossary entries - handles both JSON and CSV-like formats"""
    entries = []
    
    # First try JSON parsing
    try:
        # Clean up response text
        cleaned_text = response_text.strip()
        
        # Remove markdown code blocks if present
        if '```json' in cleaned_text or '```' in cleaned_text:
            # Extract content between code blocks
            import re
            code_block_match = re.search(r'```(?:json)?\s*(.*?)\s*```', cleaned_text, re.DOTALL)
            if code_block_match:
                cleaned_text = code_block_match.group(1)
        
        # Try to find JSON array or object
        import re
        json_match = re.search(r'[\[\{].*[\]\}]', cleaned_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            data = json.loads(json_str)
            
            if isinstance(data, list):
                # Process each entry to fix structure
                for item in data:
                    if isinstance(item, dict):
                        # Fix entries where the model put the type value in the wrong field
                        if 'character' in item and 'type' not in item:
                            # Model put raw_name in 'character' field
                            fixed_entry = {
                                'type': 'character',
                                'raw_name': item.get('character', ''),
                                'translated_name': item.get('translated_name', ''),
                                'gender': item.get('gender', 'Unknown')
                            }
                            # Copy any other fields
                            for k, v in item.items():
                                if k not in ['character', 'translated_name', 'gender']:
                                    fixed_entry[k] = v
                            entries.append(fixed_entry)
                        elif 'term' in item and 'type' not in item:
                            # Model put raw_name in 'term' field
                            fixed_entry = {
                                'type': 'term',
                                'raw_name': item.get('term', item.get('raw_name', '')),
                                'translated_name': item.get('translated_name', '')
                            }
                            # Copy any other fields
                            for k, v in item.items():
                                if k not in ['term', 'translated_name']:
                                    fixed_entry[k] = v
                            entries.append(fixed_entry)
                        elif 'type' in item:
                            # Entry has correct structure
                            entries.append(item)
                        else:
                            # Try to infer type from other fields
                            if 'gender' in item:
                                item['type'] = 'character'
                            else:
                                item['type'] = 'term'
                            entries.append(item)
                
                # If we processed entries, return them
                if entries:
                    return entries
                
                # Otherwise return original data
                return data
                
            elif isinstance(data, dict):
                # It might be a single entry or have a wrapper
                if 'type' in data and 'raw_name' in data:
                    # Single entry with correct structure
                    return [data]
                elif 'character' in data or 'term' in data:
                    # Single entry with wrong structure
                    if 'character' in data:
                        fixed_entry = {
                            'type': 'character',
                            'raw_name': data.get('character', ''),
                            'translated_name': data.get('translated_name', ''),
                            'gender': data.get('gender', 'Unknown')
                        }
                    else:
                        fixed_entry = {
                            'type': 'term',
                            'raw_name': data.get('term', data.get('raw_name', '')),
                            'translated_name': data.get('translated_name', '')
                        }
                    # Copy other fields
                    for k, v in data.items():
                        if k not in ['character', 'term', 'translated_name', 'gender', 'type', 'raw_name']:
                            fixed_entry[k] = v
                    return [fixed_entry]
                else:
                    # Check for common wrapper keys
                    for key in ['entries', 'glossary', 'characters', 'terms', 'data']:
                        if key in data and isinstance(data[key], list):
                            # Recursively parse the wrapped data
                            return parse_api_response(json.dumps(data[key]))
                    # If no wrapper found but it has valid fields, treat as single entry
                    if any(k in data for k in ['raw_name', 'translated_name']):
                        return [data]
    except (json.JSONDecodeError, AttributeError) as e:
        print(f"[Debug] JSON parsing failed: {e}")
        pass
    
    # If not JSON, try parsing as CSV-like format
    lines = response_text.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            # Skip empty lines and comments
            continue
        
        # Skip header lines
        if 'type' in line.lower() and 'raw_name' in line.lower():
            continue
        
        # Try to parse CSV-like format
        # Handle quoted values properly
        parts = []
        current_part = []
        in_quotes = False
        
        for char in line + ',':  # Add comma to process last field
            if char == '"':
                in_quotes = not in_quotes
            elif char == ',' and not in_quotes:
                parts.append(''.join(current_part).strip())
                current_part = []
            else:
                current_part.append(char)
        
        # Remove last empty part if line ended with comma
        if parts and parts[-1] == '':
            parts = parts[:-1]
        
        if len(parts) >= 3:
            entry_type = parts[0].lower()
            
            if entry_type in ['character', 'term']:
                entry = {
                    'type': entry_type,
                    'raw_name': parts[1],
                    'translated_name': parts[2]
                }
                
                # Add gender for characters
                if entry_type == 'character' and len(parts) > 3 and parts[3]:
                    entry['gender'] = parts[3]
                elif entry_type == 'character':
                    entry['gender'] = 'Unknown'
                
                # Add any custom fields
                custom_fields_json = os.getenv('GLOSSARY_CUSTOM_FIELDS', '[]')
                try:
                    custom_fields = json.loads(custom_fields_json)
                    for i, field in enumerate(custom_fields):
                        if len(parts) > 4 + i:
                            entry[field] = parts[4 + i]
                except:
                    pass
                
                entries.append(entry)
    
    return entries

def validate_extracted_entry(entry):
    """Validate that extracted entry has required fields for simple format"""
    # Check based on entry type
    if 'type' not in entry:
        return False
    
    entry_type = entry['type'].lower()
    
    # Must have raw_name and translated_name for all entries
    if 'raw_name' not in entry or not entry['raw_name']:
        return False
    if 'translated_name' not in entry or not entry['translated_name']:
        return False
    
    # Characters should have gender, but it's not strictly required
    # Custom fields are allowed, so we don't reject entries with extra fields
    
    return True

def build_prompt(chapter_text: str) -> str:
    """Build the extraction prompt - NO HARDCODED PROMPTS"""
    # Get custom prompt from environment
    custom_prompt = os.getenv('GLOSSARY_SYSTEM_PROMPT', '').strip()
    
    if not custom_prompt:
        # If no custom prompt provided, return minimal instruction
        return f"Extract glossary from this text:\n\n{chapter_text}"
    
    # Build fields specification for {fields} placeholder
    fields_spec = []
    
    # Standard format
    fields_spec.append("For characters: character,raw_name,translated_name,gender")
    fields_spec.append("For terms/locations: term,raw_name,translated_name,")
    
    # Add custom fields if any
    custom_fields_json = os.getenv('GLOSSARY_CUSTOM_FIELDS', '[]')
    try:
        custom_fields = json.loads(custom_fields_json)
        if custom_fields:
            # Add note about custom fields
            custom_fields_str = ','.join(custom_fields)
            fields_spec.append(f"Custom fields (add as extra columns): {custom_fields_str}")
    except:
        custom_fields = []
    
    fields_str = '\n'.join(fields_spec)
    
    # Replace placeholders
    prompt = custom_prompt
    prompt = prompt.replace('{fields}', fields_str)
    prompt = prompt.replace('{chapter_text}', chapter_text)
    prompt = prompt.replace('{{fields}}', fields_str)
    prompt = prompt.replace('{{chapter_text}}', chapter_text)
    prompt = prompt.replace('{text}', chapter_text)
    prompt = prompt.replace('{{text}}', chapter_text)
    
    return prompt

def skip_duplicate_entries(glossary):
    """
    Skip entries with duplicate raw names (after removing honorifics).
    Returns deduplicated list maintaining first occurrence of each unique raw name.
    """
    seen_raw_names = set()
    deduplicated = []
    skipped_count = 0
    
    for entry in glossary:
        # Get raw_name and clean it
        raw_name = entry.get('raw_name', '')
        if not raw_name:
            continue
            
        # Remove honorifics for comparison (unless disabled)
        cleaned_name = remove_honorifics(raw_name)
        
        # Check if we've seen this cleaned name before
        if cleaned_name.lower() in seen_raw_names:
            skipped_count += 1
            print(f"[Skip] Duplicate entry: {raw_name} (cleaned: {cleaned_name})")
            continue
        
        # Add to seen set and keep the entry
        seen_raw_names.add(cleaned_name.lower())
        deduplicated.append(entry)
    
    if skipped_count > 0:
        print(f"⏭️ Skipped {skipped_count} duplicate entries")
        print(f"✅ Kept {len(deduplicated)} unique entries")
    
    return deduplicated

# Replace the merge_glossary_entries function with skip_duplicate_entries
merge_glossary_entries = skip_duplicate_entries  # For backward compatibility

# Batch processing functions
def process_chapter_batch(chapters_batch: List[Tuple[int, str]], 
                         client: UnifiedClient,
                         config: Dict,
                         contextual_enabled: bool,
                         history: List[Dict],
                         ctx_limit: int,
                         rolling_window: bool,
                         check_stop,
                         chunk_timeout: int = None) -> List[Dict]:
    """
    Process a batch of chapters in parallel with improved interrupt support
    """
    sys_prompt = config.get('system_prompt', 'You are a helpful assistant.')
    temp = float(os.getenv("GLOSSARY_TEMPERATURE") or config.get('temperature', 0.1))
    
    env_max_output = os.getenv("MAX_OUTPUT_TOKENS")
    if env_max_output and env_max_output.isdigit():
        mtoks = int(env_max_output)
    else:
        mtoks = config.get('max_tokens', 4196)
    
    results = []
    
    with ThreadPoolExecutor(max_workers=len(chapters_batch)) as executor:
        futures = {}
        
        for idx, chap in chapters_batch:
            if check_stop():
                break
                
            # Build messages for this chapter
            if not contextual_enabled:
                msgs = [{"role":"system","content":sys_prompt}] \
                     + [{"role":"user","content":build_prompt(chap)}]
            else:
                msgs = [{"role":"system","content":sys_prompt}] \
                     + trim_context_history(history, ctx_limit, rolling_window) \
                     + [{"role":"user","content":build_prompt(chap)}]
            
            # Submit to thread pool
            future = executor.submit(
                process_single_chapter_api_call,
                idx, chap, msgs, client, temp, mtoks, check_stop, chunk_timeout
            )
            futures[future] = (idx, chap)
        
        # Process results with better cancellation
        try:
            for future in as_completed(futures, timeout=1):  # Add timeout to as_completed
                if check_stop():
                    print("🛑 Stop detected - cancelling all pending operations...")
                    # Cancel all pending futures immediately
                    cancelled = cancel_all_futures(list(futures.keys()))
                    if cancelled > 0:
                        print(f"✅ Cancelled {cancelled} pending API calls")
                    # Shutdown executor immediately
                    executor.shutdown(wait=False)
                    break
                    
                idx, chap = futures[future]
                try:
                    result = future.result(timeout=0.5)  # Short timeout on result retrieval
                    result['chap'] = chap
                    results.append(result)
                except Exception as e:
                    if "stopped by user" in str(e).lower():
                        print(f"✅ Chapter {idx+1} stopped by user")
                    else:
                        print(f"Error processing chapter {idx+1}: {e}")
                    results.append({
                        'idx': idx,
                        'data': [],
                        'resp': "",
                        'chap': chap,
                        'error': str(e)
                    })
        except TimeoutError:
            # Check for stop more frequently during processing
            while any(not f.done() for f in futures):
                if check_stop():
                    print("🛑 Stop detected during batch processing...")
                    cancelled = cancel_all_futures(list(futures.keys()))
                    if cancelled > 0:
                        print(f"✅ Cancelled {cancelled} pending API calls")
                    executor.shutdown(wait=False)
                    break
                time.sleep(0.1)  # Short sleep to check frequently
    
    # Sort results by chapter index
    results.sort(key=lambda x: x['idx'])
    return results

def process_single_chapter_api_call(idx: int, chap: str, msgs: List[Dict], 
                                  client: UnifiedClient, temp: float, mtoks: int,
                                  stop_check_fn, chunk_timeout: int = None) -> Dict:
    """Process a single chapter API call with thread-safe payload handling"""
    start_time = time.time()
    print(f"[BATCH] Starting API call for Chapter {idx+1} at {time.strftime('%H:%M:%S')}")
    
    # Thread-safe payload directory
    thread_name = threading.current_thread().name
    thread_id = threading.current_thread().ident
    thread_dir = os.path.join("Payloads", "glossary", f"{thread_name}_{thread_id}")
    os.makedirs(thread_dir, exist_ok=True)
    
    try:
        # Save request payload before API call
        payload_file = os.path.join(thread_dir, f"chapter_{idx+1}_request.json")
        with open(payload_file, 'w', encoding='utf-8') as f:
            json.dump({
                'chapter': idx + 1,
                'messages': msgs,
                'temperature': temp,
                'max_tokens': mtoks,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }, f, indent=2, ensure_ascii=False)
        
        # Use send_with_interrupt for API call
        raw = send_with_interrupt(
            messages=msgs,
            client=client, 
            temperature=temp,
            max_tokens=mtoks,
            stop_check_fn=stop_check_fn,
            chunk_timeout=chunk_timeout
        )
        
        resp = raw[0] if isinstance(raw, tuple) else raw
        
        # Save the raw response in thread-safe location
        response_file = os.path.join(thread_dir, f"chapter_{idx+1}_response.txt")
        with open(response_file, "w", encoding="utf-8", errors="replace") as f:
            f.write(resp)
        
        # Parse response using the new parser
        data = parse_api_response(resp)
        
        # Filter out invalid entries
        valid_data = []
        for entry in data:
            if validate_extracted_entry(entry):
                # Clean the raw_name
                if 'raw_name' in entry:
                    entry['raw_name'] = entry['raw_name'].strip()
                valid_data.append(entry)
            else:
                print(f"[Debug] Skipped invalid entry in chapter {idx+1}: {entry}")
        
        elapsed = time.time() - start_time
        print(f"[BATCH] Completed Chapter {idx+1} in {elapsed:.1f}s at {time.strftime('%H:%M:%S')} - Extracted {len(valid_data)} entries")
        
        return {
            'idx': idx,
            'data': valid_data,
            'resp': resp,
            'error': None
        }
            
    except UnifiedClientError as e:
        print(f"[Error] API call interrupted/failed for chapter {idx+1}: {e}")
        return {
            'idx': idx,
            'data': [],
            'resp': "",
            'error': str(e)
        }
    except Exception as e:
        print(f"[Error] Unexpected error for chapter {idx+1}: {e}")
        return {
            'idx': idx,
            'data': [],
            'resp': "",
            'error': str(e)
        }

# Update main function to support batch processing:
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
    
    # Check for batch mode
    batch_enabled = os.getenv("BATCH_TRANSLATION", "0") == "1"
    batch_size = int(os.getenv("BATCH_SIZE", "5"))
    
    print(f"[DEBUG] BATCH_TRANSLATION = {os.getenv('BATCH_TRANSLATION')} (enabled: {batch_enabled})")
    print(f"[DEBUG] BATCH_SIZE = {batch_size}")
    
    if batch_enabled:
        print(f"🚀 Batch mode enabled with size: {batch_size}")
    
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

    # Log settings
    print("📑 Glossary Format: Simple (type, raw_name, translated_name, gender)")
    
    # Check honorifics filter toggle
    honorifics_disabled = os.getenv('GLOSSARY_DISABLE_HONORIFICS_FILTER', '0') == '1'
    if honorifics_disabled:
        print("📑 Honorifics Filtering: ❌ DISABLED")
    else:
        print("📑 Honorifics Filtering: ✅ ENABLED")
    
    # Log custom fields
    custom_fields_json = os.getenv('GLOSSARY_CUSTOM_FIELDS', '[]')
    try:
        custom_fields = json.loads(custom_fields_json)
        if custom_fields:
            print(f"📑 Custom Fields: {', '.join(custom_fields)}")
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
    
    # Get both settings
    contextual_enabled = os.getenv('CONTEXTUAL', '1') == '1'
    rolling_window = os.getenv('GLOSSARY_HISTORY_ROLLING', '0') == '1'
    
    # Count chapters that will be processed with range filter
    chapters_to_process = []
    for idx, chap in enumerate(chapters):
        # Skip if chapter is outside the range
        if range_start is not None and range_end is not None:
            chapter_num = idx + 1  # 1-based chapter numbering
            if not (range_start <= chapter_num <= range_end):
                continue
        if idx not in completed:
            chapters_to_process.append((idx, chap))
    
    if len(chapters_to_process) < total_chapters:
        print(f"📊 Processing {len(chapters_to_process)} out of {total_chapters} chapters")
    
    # Get chunk timeout from environment  
    chunk_timeout = int(os.getenv("CHUNK_TIMEOUT", "900"))  # 15 minutes default
    
    # Process chapters based on mode
    if batch_enabled and len(chapters_to_process) > 0:
        # BATCH MODE: Process in batches
        total_batches = (len(chapters_to_process) + batch_size - 1) // batch_size
        
        for batch_num in range(total_batches):
            # Check for stop at the beginning of each batch
            if check_stop():
                print(f"❌ Glossary extraction stopped at batch {batch_num+1}")
                return
            
            # Get current batch
            batch_start = batch_num * batch_size
            batch_end = min(batch_start + batch_size, len(chapters_to_process))
            current_batch = chapters_to_process[batch_start:batch_end]
            
            print(f"\n🔄 Processing Batch {batch_num+1}/{total_batches} (Chapters: {[idx+1 for idx, _ in current_batch]})")
            print(f"[BATCH] Submitting {len(current_batch)} chapters for parallel processing...")
            batch_start_time = time.time()
            
            # Process batch in parallel
            batch_results = process_chapter_batch(
                current_batch, client, config, contextual_enabled,
                history, ctx_limit, rolling_window, check_stop, chunk_timeout
            )
            
            batch_elapsed = time.time() - batch_start_time
            print(f"[BATCH] All {len(current_batch)} chapters completed in {batch_elapsed:.1f}s total")
            avg_time = batch_elapsed / len(current_batch)
            print(f"[BATCH] Average time per chapter: {avg_time:.1f}s (vs sequential: ~{api_delay + avg_time:.1f}s)")
            
            # Process results from the batch
            batch_glossary_entries = []
            
            for result in batch_results:
                if check_stop():
                    print(f"❌ Glossary extraction stopped during batch processing")
                    return
                
                idx = result['idx']
                data = result['data']
                resp = result['resp']
                chap = result['chap']
                error = result.get('error')
                
                if error:
                    print(f"[Chapter {idx+1}] Error: {error}")
                    continue
                
                # Log entries
                total_ent = len(data)
                for eidx, entry in enumerate(data, start=1):
                    elapsed = time.time() - start
                    if idx == 0 and eidx == 1:
                        eta = 0
                    else:
                        avg = elapsed / ((idx * 100) + eidx)
                        eta = avg * (total_chapters * 100 - ((idx * 100) + eidx))
                    
                    # Get entry info based on new format
                    entry_type = entry.get("type", "?")
                    raw_name = entry.get("raw_name", "?")
                    trans_name = entry.get("translated_name", "?")
                    
                    print(f'[Chapter {idx+1}/{total_chapters}] [{eidx}/{total_ent}] ({elapsed:.1f}s elapsed, ETA {eta:.1f}s) → {entry_type}: {raw_name} ({trans_name})')
                
                # Collect entries for batch merging
                batch_glossary_entries.extend(data)
                completed.append(idx)
                
                # Only add to history if contextual is enabled
                if contextual_enabled:
                    history.append({"user": build_prompt(chap), "assistant": resp})
            
            # Apply skip logic to batch entries
            if batch_glossary_entries:
                print(f"🔀 Processing {len(batch_glossary_entries)} entries from batch {batch_num+1}")
                glossary.extend(batch_glossary_entries)
                glossary[:] = skip_duplicate_entries(glossary)
            
            # Save progress after each batch
            save_progress(completed, glossary, history)
            save_glossary_json(glossary, os.path.join(glossary_dir, os.path.basename(args.output)))
            save_glossary_csv(glossary, os.path.join(glossary_dir, os.path.basename(args.output)))
            
            # Handle context history
            if contextual_enabled:
                # Reset history when limit reached without rolling window
                if not rolling_window and len(history) >= ctx_limit and ctx_limit > 0:
                    print(f"🔄 Resetting glossary context (reached {ctx_limit} chapter limit)")
                    history = []
                    prog['context_history'] = []
            
            # Add delay between batches (but not after the last batch)
            if batch_num < total_batches - 1:
                print(f"⏱️  Waiting {api_delay}s before next batch...")
                if not interruptible_sleep(api_delay, check_stop, 0.1):
                    print(f"❌ Glossary extraction stopped during delay")
                    return
    
    else:
        # SEQUENTIAL MODE: Original behavior
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
                    chapter_glossary_data = []  # Collect data from all chunks
                    
                    for chunk_html, chunk_idx, total_chunks in chunks:
                        if check_stop():
                            print(f"❌ Glossary extraction stopped during chunk {chunk_idx} of chapter {idx+1}")
                            return
                            
                        print(f"🔄 Processing chunk {chunk_idx}/{total_chunks} of Chapter {idx+1}")
                        
                        # Extract text from the chunk HTML
                        from bs4 import BeautifulSoup
                        soup = BeautifulSoup(chunk_html, 'html.parser')
                        chunk_text = soup.get_text(strip=True)
                        
                        # Build messages for this chunk (same logic as main chapter)
                        if not contextual_enabled:
                            chunk_msgs = [{"role":"system","content":sys_prompt}] \
                                        + [{"role":"user","content":build_prompt(chunk_text)}]
                        else:
                            chunk_msgs = [{"role":"system","content":sys_prompt}] \
                                        + trim_context_history(history, ctx_limit, rolling_window) \
                                        + [{"role":"user","content":build_prompt(chunk_text)}]
                        
                        # API call for chunk
                        try:
                            chunk_raw = send_with_interrupt(
                                messages=chunk_msgs,
                                client=client,
                                temperature=temp,
                                max_tokens=mtoks,
                                stop_check_fn=check_stop,
                                chunk_timeout=chunk_timeout
                            )
                        except UnifiedClientError as e:
                            if "stopped by user" in str(e).lower():
                                print(f"❌ Glossary extraction stopped during chunk {chunk_idx} API call")
                                return
                            elif "timeout" in str(e).lower():
                                print(f"⚠️ Chunk {chunk_idx} API call timed out: {e}")
                                continue  # Skip this chunk
                            else:
                                print(f"❌ Chunk {chunk_idx} API error: {e}")
                                continue  # Skip this chunk
                        except Exception as e:
                            print(f"❌ Unexpected error in chunk {chunk_idx}: {e}")
                            continue  # Skip this chunk
                        
                        # Process chunk response
                        chunk_resp = chunk_raw[0] if isinstance(chunk_raw, tuple) else chunk_raw
                        
                        # Save chunk response with thread-safe location
                        thread_name = threading.current_thread().name
                        thread_id = threading.current_thread().ident
                        thread_dir = os.path.join("Payloads", "glossary", f"{thread_name}_{thread_id}")
                        os.makedirs(thread_dir, exist_ok=True)
                        
                        with open(os.path.join(thread_dir, f"chunk_response_chap{idx+1}_chunk{chunk_idx}.txt"), "w", encoding="utf-8", errors="replace") as f:
                            f.write(chunk_resp)
                        
                        # Extract JSON from chunk
                        chunk_resp_data = parse_api_response(chunk_resp)
                        
                        if not chunk_resp_data:
                            print(f"[Warning] No data found in chunk {chunk_idx}, skipping...")
                            continue
                        
                        # Parse chunk JSON
                        try:
                            chunk_data = json.loads(chunk_json_str)
                            
                            # Filter out invalid entries
                            valid_chunk_data = []
                            for entry in chunk_data:
                                if validate_extracted_entry(entry):
                                    valid_chunk_data.append(entry)
                                else:
                                    print(f"[Debug] Skipped invalid entry in chunk {chunk_idx}: {entry}")
                            
                            chapter_glossary_data.extend(valid_chunk_data)
                            print(f"✅ Chunk {chunk_idx}/{total_chunks}: extracted {len(valid_chunk_data)} entries")
                            
                            # Add chunk to history if contextual
                            if contextual_enabled:
                                history.append({"user": build_prompt(chunk_text), "assistant": chunk_resp})
                                
                        except json.JSONDecodeError as e:
                            print(f"[Warning] JSON decode error in chunk {chunk_idx}: {e}")
                            continue
                        
                        # Add delay between chunks (but not after last chunk)
                        if chunk_idx < total_chunks:
                            print(f"⏱️  Waiting {api_delay}s before next chunk...")
                            if not interruptible_sleep(api_delay, check_stop, 0.1):
                                print(f"❌ Glossary extraction stopped during chunk delay")
                                return
                    
                    # Use the collected data from all chunks
                    data = chapter_glossary_data
                    print(f"✅ Chapter {idx+1} processed in {len(chunks)} chunks, total entries: {len(data)}")
                    
                else:
                    # Original single-chapter processing (your existing logic)
                    # Check for stop before API call
                    if check_stop():
                        print(f"❌ Glossary extraction stopped before API call for chapter {idx+1}")
                        return
                
                    try:
                        # Use send_with_interrupt for API call
                        raw = send_with_interrupt(
                            messages=msgs,
                            client=client,
                            temperature=temp,
                            max_tokens=mtoks,
                            stop_check_fn=check_stop,
                            chunk_timeout=chunk_timeout
                        )
                    except UnifiedClientError as e:
                        if "stopped by user" in str(e).lower():
                            print(f"❌ Glossary extraction stopped during API call for chapter {idx+1}")
                            return
                        elif "timeout" in str(e).lower():
                            print(f"⚠️ API call timed out for chapter {idx+1}: {e}")
                            continue
                        else:
                            print(f"❌ API error for chapter {idx+1}: {e}")
                            continue
                    except Exception as e:
                        print(f"❌ Unexpected error for chapter {idx+1}: {e}")
                        continue
                    
                    # Process response
                    resp = raw[0] if isinstance(raw, tuple) else raw

                    # Save the raw response with thread-safe location
                    thread_name = threading.current_thread().name
                    thread_id = threading.current_thread().ident
                    thread_dir = os.path.join("Payloads", "glossary", f"{thread_name}_{thread_id}")
                    os.makedirs(thread_dir, exist_ok=True)
                    
                    with open(os.path.join(thread_dir, f"response_chap{idx+1}.txt"), "w", encoding="utf-8", errors="replace") as f:
                        f.write(resp)

                    # Parse response using the new parser
                    data = parse_api_response(resp)
                    
                    # Filter out invalid entries
                    valid_data = []
                    for entry in data:
                        if validate_extracted_entry(entry):
                            valid_data.append(entry)
                        else:
                            print(f"[Debug] Skipped invalid entry: {entry}")
                    
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
                        
                        # Get entry info based on new format
                        entry_type = entry.get("type", "?")
                        raw_name = entry.get("raw_name", "?")
                        trans_name = entry.get("translated_name", "?")
                        
                        print(f'[Chapter {idx+1}/{total_chapters}] [{eidx}/{total_ent}] ({elapsed:.1f}s elapsed, ETA {eta:.1f}s) → {entry_type}: {raw_name} ({trans_name})')    
                    
                # Apply skip logic and save
                glossary.extend(data)
                glossary[:] = skip_duplicate_entries(glossary)
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
                save_glossary_csv(glossary, os.path.join(glossary_dir, os.path.basename(args.output)))
                
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
                        if not interruptible_sleep(api_delay, check_stop, 0.1):
                            print(f"❌ Glossary extraction stopped during delay")
                            return
                            
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
    
    # Also save as CSV format for compatibility
    try:
        csv_output = args.output.replace('.json', '.csv')
        csv_path = os.path.join(glossary_dir, os.path.basename(csv_output))
        save_glossary_csv(glossary, os.path.join(glossary_dir, os.path.basename(args.output)))
        print(f"Also saved as CSV: {csv_path}")
    except Exception as e:
        print(f"[Warning] Could not save CSV format: {e}")

if __name__=='__main__':
    main()
