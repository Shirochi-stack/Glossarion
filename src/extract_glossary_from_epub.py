# extract_glossary_from_epub.py
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
import tempfile
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
        if check_stop_fn and check_stop_fn():  # Add safety check for None
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

def create_client_with_multi_key_support(api_key, model, output_dir, config):
    """Create a UnifiedClient with multi API key support if enabled"""
    
    # Check if multi API key mode is enabled
    use_multi_keys = config.get('use_multi_api_keys', False)
    
    # Set environment variables for UnifiedClient to pick up
    if use_multi_keys and 'multi_api_keys' in config and config['multi_api_keys']:
        print("🔑 Multi API Key mode enabled for glossary extraction")
        
        # Set environment variables that UnifiedClient will read
        os.environ['USE_MULTI_API_KEYS'] = '1'
        os.environ['MULTI_API_KEYS'] = json.dumps(config['multi_api_keys'])
        os.environ['FORCE_KEY_ROTATION'] = '1' if config.get('force_key_rotation', True) else '0'
        os.environ['ROTATION_FREQUENCY'] = str(config.get('rotation_frequency', 1))
        
        print(f"   • Keys configured: {len(config['multi_api_keys'])}")
        print(f"   • Force rotation: {config.get('force_key_rotation', True)}")
        print(f"   • Rotation frequency: every {config.get('rotation_frequency', 1)} request(s)")
    else:
        # Ensure multi-key mode is disabled in environment
        os.environ['USE_MULTI_API_KEYS'] = '0'
        
    # Create UnifiedClient normally - it will check environment variables
    return UnifiedClient(api_key=api_key, model=model, output_dir=output_dir)
    
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
    check_interval = 0.1
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

# Threading locks for atomic glossary saves
_glossary_json_lock = threading.Lock()
_glossary_csv_lock = threading.Lock()
_progress_lock = threading.Lock()

def set_stop_flag(value):
    """Set the global stop flag"""
    global _stop_requested
    _stop_requested = value
    
    # When clearing the stop flag, also clear the multi-key environment variable
    if not value:
        os.environ['TRANSLATION_CANCELLED'] = '0'
        
        # Also clear UnifiedClient global flag
        try:
            import unified_api_client
            if hasattr(unified_api_client, 'UnifiedClient'):
                unified_api_client.UnifiedClient._global_cancelled = False
        except:
            pass

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
        import threading
        
        class CallbackWriter:
            def __init__(self, callback):
                self.callback = callback
                self.buffer = ""
                self.main_thread = threading.main_thread()
                
            def write(self, text):
                if text.strip():
                    # The callback (append_log) is already thread-safe - it handles QTimer internally
                    # So we can call it directly from any thread
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

def get_custom_entry_types():
    """Get custom entry types configuration from environment"""
    try:
        types_json = os.getenv('GLOSSARY_CUSTOM_ENTRY_TYPES', '{}')
        result = json.loads(types_json)
        # If empty, return defaults
        if not result:
            return {
                'character': {'enabled': True, 'has_gender': True},
                'term': {'enabled': True, 'has_gender': False}
            }
        return result
    except:
        # Default configuration
        return {
            'character': {'enabled': True, 'has_gender': True},
            'term': {'enabled': True, 'has_gender': False}
        }

def save_glossary_json(glossary: List[Dict], output_path: str):
    """Save glossary in the new simple format with automatic sorting by type"""
    global _glossary_json_lock
    
    # Acquire lock to prevent concurrent writes
    with _glossary_json_lock:
        # Get custom types for sorting order
        custom_types = get_custom_entry_types()
        
        # Create sorting order: character=0, term=1, others alphabetically starting from 2
        type_order = {'character': 0, 'term': 1}
        other_types = sorted([t for t in custom_types.keys() if t not in ['character', 'term']])
        for i, t in enumerate(other_types):
            type_order[t] = i + 2
        
        # Sort glossary by type order, then by raw_name
        sorted_glossary = sorted(glossary, key=lambda x: (
            type_order.get(x.get('type', 'term'), 999),  # Unknown types go last
            x.get('raw_name', '').lower()
        ))
        
        # Use atomic write to prevent corruption during parallel saves
        try:
            # Create temp file in the same directory for atomic rename
            output_dir = os.path.dirname(output_path) or '.'
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=output_dir, delete=False, suffix='.tmp') as temp_f:
                temp_path = temp_f.name
                json.dump(sorted_glossary, temp_f, ensure_ascii=False, indent=2)
                temp_f.flush()
                os.fsync(temp_f.fileno())  # Ensure data is written to disk
            
            # Atomic rename
            try:
                if os.path.exists(output_path):
                    os.remove(output_path)
                os.rename(temp_path, output_path)
            except Exception as e:
                # If rename fails, try to clean up temp file
                if os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except:
                        pass
                raise
        except Exception as e:
            print(f"[Warning] Atomic write failed for JSON: {e}. Attempting direct write...")
            try:
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(sorted_glossary, f, ensure_ascii=False, indent=2)
            except Exception as e2:
                print(f"[Error] Failed to save glossary JSON: {e2}")

def save_glossary_csv(glossary: List[Dict], output_path: str):
    """Save glossary in CSV or token-efficient format based on environment variable"""
    global _glossary_csv_lock
    import csv
    
    with _glossary_csv_lock:
        csv_path = output_path.replace('.json', '.csv')
        custom_types = get_custom_entry_types()
        type_order = {'character': 0, 'term': 1}
        other_types = sorted([t for t in custom_types.keys() if t not in ['character', 'term']])
        for i, t in enumerate(other_types):
            type_order[t] = i + 2
        
        sorted_glossary = sorted(glossary, key=lambda x: (
            type_order.get(x.get('type', 'term'), 999),
            x.get('raw_name', '').lower()
        ))
        
        use_legacy_format = os.getenv('GLOSSARY_USE_LEGACY_CSV', '0') == '1'
        csv_dir = os.path.dirname(csv_path) or '.'
        
        try:
            if use_legacy_format:
                with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=csv_dir, delete=False, newline='', suffix='.tmp') as temp_f:
                    temp_path = temp_f.name
                    writer = csv.writer(temp_f)
                    header = ['type', 'raw_name', 'translated_name', 'gender']
                    custom_fields_json = os.getenv('GLOSSARY_CUSTOM_FIELDS', '[]')
                    try:
                        custom_fields = json.loads(custom_fields_json)
                        header.extend(custom_fields)
                    except:
                        custom_fields = []
                    writer.writerow(header)
                    for entry in sorted_glossary:
                        entry_type = entry.get('type', 'term')
                        type_config = custom_types.get(entry_type, {})
                        row = [entry_type, entry.get('raw_name', ''), entry.get('translated_name', '')]
                        if type_config.get('has_gender', False):
                            row.append(entry.get('gender', ''))
                        for field in custom_fields:
                            row.append(entry.get(field, ''))
                        expected_fields = 4 + len(custom_fields)
                        while len(row) > expected_fields and row[-1] == '':
                            row.pop()
                        while len(row) < 3:
                            row.append('')
                        writer.writerow(row)
                    temp_f.flush()
                    os.fsync(temp_f.fileno())  # Ensure data is written to disk
                
                try:
                    if os.path.exists(csv_path):
                        os.remove(csv_path)
                    os.rename(temp_path, csv_path)
                except Exception as e:
                    if os.path.exists(temp_path):
                        try:
                            os.remove(temp_path)
                        except:
                            pass
                    raise
                print(f"✅ Saved legacy CSV format: {csv_path}")
            
            else:
                grouped_entries = {}
                for entry in sorted_glossary:
                    entry_type = entry.get('type', 'term')
                    if entry_type not in grouped_entries:
                        grouped_entries[entry_type] = []
                    grouped_entries[entry_type].append(entry)
                
                custom_fields_json = os.getenv('GLOSSARY_CUSTOM_FIELDS', '[]')
                try:
                    custom_fields = json.loads(custom_fields_json)
                except:
                    custom_fields = []
                
                with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=csv_dir, delete=False, suffix='.tmp') as temp_f:
                    temp_path = temp_f.name
                    
                    # Write column header
                    column_headers = ['translated_name', 'raw_name']
                    # Add gender if any type supports it
                    has_gender = any(type_config.get('has_gender', False) for type_config in custom_types.values())
                    if has_gender:
                        column_headers.append('gender')
                    if custom_fields:
                        column_headers.extend(custom_fields)
                    temp_f.write(f"Glossary Columns: {', '.join(column_headers)}\n\n")
                    for entry_type in sorted(grouped_entries.keys(), key=lambda x: type_order.get(x, 999)):
                        entries = grouped_entries[entry_type]
                        type_config = custom_types.get(entry_type, {})
                        section_name = entry_type.upper() + 'S' if not entry_type.upper().endswith('S') else entry_type.upper()
                        temp_f.write(f"=== {section_name} ===\n")
                        for entry in entries:
                            raw_name = entry.get('raw_name', '')
                            translated_name = entry.get('translated_name', '')
                            line = f"* {translated_name} ({raw_name})"
                            if type_config.get('has_gender', False):
                                gender = entry.get('gender', '')
                                if gender and gender != 'Unknown':
                                    line += f" [{gender}]"
                            custom_field_parts = []
                            for field in custom_fields:
                                value = entry.get(field, '').strip()
                                if value:
                                    if field.lower() in ['description', 'notes', 'details']:
                                        line += f": {value}"
                                    else:
                                        custom_field_parts.append(f"{field}: {value}")
                            if custom_field_parts:
                                line += f" ({', '.join(custom_field_parts)})"
                            temp_f.write(line + "\n")
                        temp_f.write("\n")
                    temp_f.flush()
                    os.fsync(temp_f.fileno())  # Ensure data is written to disk
                
                try:
                    if os.path.exists(csv_path):
                        os.remove(csv_path)
                    os.rename(temp_path, csv_path)
                except Exception as e:
                    if os.path.exists(temp_path):
                        try:
                            os.remove(temp_path)
                        except:
                            pass
                    raise
                print(f"✅ Saved token-efficient glossary: {csv_path}")
                type_counts = {}
                for entry_type in grouped_entries:
                    type_counts[entry_type] = len(grouped_entries[entry_type])
                total = sum(type_counts.values())
                print(f"   Total entries: {total}")
                for entry_type, count in type_counts.items():
                    print(f"   - {entry_type}: {count} entries")
        
        except Exception as e:
            print(f"[Warning] Atomic write failed for CSV: {e}. Attempting direct write...")
            try:
                if use_legacy_format:
                    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
                        writer = csv.writer(f)
                        header = ['type', 'raw_name', 'translated_name', 'gender']
                        try:
                            custom_fields = json.loads(os.getenv('GLOSSARY_CUSTOM_FIELDS', '[]'))
                            header.extend(custom_fields)
                        except:
                            custom_fields = []
                        writer.writerow(header)
                        for entry in sorted_glossary:
                            entry_type = entry.get('type', 'term')
                            type_config = custom_types.get(entry_type, {})
                            row = [entry_type, entry.get('raw_name', ''), entry.get('translated_name', '')]
                            if type_config.get('has_gender', False):
                                row.append(entry.get('gender', ''))
                            for field in custom_fields:
                                row.append(entry.get(field, ''))
                            writer.writerow(row)
                else:
                    grouped_entries = {}
                    for entry in sorted_glossary:
                        entry_type = entry.get('type', 'term')
                        if entry_type not in grouped_entries:
                            grouped_entries[entry_type] = []
                        grouped_entries[entry_type].append(entry)
                    with open(csv_path, 'w', encoding='utf-8') as f:
                        # Write column header
                        custom_fields_json = os.getenv('GLOSSARY_CUSTOM_FIELDS', '[]')
                        try:
                            custom_fields_list = json.loads(custom_fields_json)
                        except:
                            custom_fields_list = []
                        column_headers = ['translated_name', 'raw_name']
                        # Add gender if any type supports it
                        has_gender = any(type_config.get('has_gender', False) for type_config in custom_types.values())
                        if has_gender:
                            column_headers.append('gender')
                        if custom_fields_list:
                            column_headers.extend(custom_fields_list)
                        f.write(f"Glossary Columns: {', '.join(column_headers)}\n\n")
                        for entry_type in sorted(grouped_entries.keys(), key=lambda x: type_order.get(x, 999)):
                            entries = grouped_entries[entry_type]
                            type_config = custom_types.get(entry_type, {})
                            section_name = entry_type.upper() + 'S' if not entry_type.upper().endswith('S') else entry_type.upper()
                            f.write(f"=== {section_name} ===\n")
                            for entry in entries:
                                raw_name = entry.get('raw_name', '')
                                translated_name = entry.get('translated_name', '')
                                line = f"* {translated_name} ({raw_name})"
                                if type_config.get('has_gender', False):
                                    gender = entry.get('gender', '')
                                    if gender and gender != 'Unknown':
                                        line += f" [{gender}]"
                                custom_field_parts = []
                                for field in custom_fields:
                                    value = entry.get(field, '').strip()
                                    if value:
                                        if field.lower() in ['description', 'notes', 'details']:
                                            line += f": {value}"
                                        else:
                                            custom_field_parts.append(f"{field}: {value}")
                                if custom_field_parts:
                                    line += f" ({', '.join(custom_field_parts)})"
                                f.write(line + "\n")
                            f.write("\n")
            except Exception as e2:
                print(f"[Error] Failed to save CSV: {e2}")
            
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
    """Parse API response to extract glossary entries - handles custom types"""
    entries = []
    
    # Get enabled types from custom configuration
    custom_types = get_custom_entry_types()
    enabled_types = [t for t, cfg in custom_types.items() if cfg.get('enabled', True)]
    
    # First try JSON parsing
    try:
        # Clean up response text
        cleaned_text = response_text.strip()
        
        # Remove markdown code blocks if present
        if '```json' in cleaned_text or '```' in cleaned_text:
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
                for item in data:
                    if isinstance(item, dict):
                        # Check if entry type is enabled
                        entry_type = item.get('type', '').lower()
                        
                        # Handle legacy format where type is the key
                        if not entry_type:
                            for type_name in enabled_types:
                                if type_name in item:
                                    entry_type = type_name
                                    fixed_entry = {
                                        'type': type_name,
                                        'raw_name': item.get(type_name, ''),
                                        'translated_name': item.get('translated_name', '')
                                    }
                                    
                                    # Add gender if type supports it
                                    if custom_types.get(type_name, {}).get('has_gender', False):
                                        fixed_entry['gender'] = item.get('gender', 'Unknown')
                                    
                                    # Copy other fields
                                    for k, v in item.items():
                                        if k not in [type_name, 'translated_name', 'gender', 'type', 'raw_name']:
                                            fixed_entry[k] = v
                                    
                                    entries.append(fixed_entry)
                                    break
                        else:
                            # Standard format with type field
                            if entry_type in enabled_types:
                                entries.append(item)
                
                return entries
                
            elif isinstance(data, dict):
                # Handle single entry
                entry_type = data.get('type', '').lower()
                if entry_type in enabled_types:
                    return [data]
                
                # Check for wrapper
                for key in ['entries', 'glossary', 'characters', 'terms', 'data']:
                    if key in data and isinstance(data[key], list):
                        return parse_api_response(json.dumps(data[key]))
                
                return []
                
    except (json.JSONDecodeError, AttributeError) as e:
        print(f"[Debug] JSON parsing failed: {e}")
        pass
    
    # CSV-like format parsing
    lines = response_text.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        
        # Skip header lines
        if 'type' in line.lower() and 'raw_name' in line.lower():
            continue
        
        # Parse CSV
        parts = []
        current_part = []
        in_quotes = False
        
        for char in line + ',':
            if char == '"':
                in_quotes = not in_quotes
            elif char == ',' and not in_quotes:
                parts.append(''.join(current_part).strip())
                current_part = []
            else:
                current_part.append(char)
        
        if parts and parts[-1] == '':
            parts = parts[:-1]
        
        if len(parts) >= 3:
            entry_type = parts[0].lower()
            
            # Check if type is enabled
            if entry_type not in enabled_types:
                continue
            
            entry = {
                'type': entry_type,
                'raw_name': parts[1],
                'translated_name': parts[2]
            }
            
            # Add gender if type supports it and it's provided
            type_config = custom_types.get(entry_type, {})
            if type_config.get('has_gender', False) and len(parts) > 3 and parts[3]:
                entry['gender'] = parts[3]
            elif type_config.get('has_gender', False):
                entry['gender'] = 'Unknown'
            
            # Add any custom fields
            custom_fields_json = os.getenv('GLOSSARY_CUSTOM_FIELDS', '[]')
            try:
                custom_fields = json.loads(custom_fields_json)
                start_idx = 4  # Always 4, not conditional
                for i, field in enumerate(custom_fields):
                    if len(parts) > start_idx + i:
                        field_value = parts[start_idx + i]
                        if field_value:  # Only add if not empty
                            entry[field] = field_value
            except:
                pass
            
            entries.append(entry)
    
    return entries

def validate_extracted_entry(entry):
    """Validate that extracted entry has required fields and enabled type"""
    if 'type' not in entry:
        return False
    
    # Check if type is enabled
    custom_types = get_custom_entry_types()
    entry_type = entry.get('type', '').lower()
    
    if entry_type not in custom_types:
        return False
    
    if not custom_types[entry_type].get('enabled', True):
        return False
    
    # Must have raw_name and translated_name
    if 'raw_name' not in entry or not entry['raw_name']:
        return False
    if 'translated_name' not in entry or not entry['translated_name']:
        return False
    
    return True

def build_prompt(chapter_text: str) -> tuple:
    """Build the extraction prompt with custom types - returns (system_prompt, user_prompt)"""
    custom_prompt = os.getenv('GLOSSARY_SYSTEM_PROMPT', '').strip()
    
    if not custom_prompt:
        # If no custom prompt, create a default
        custom_prompt = """Extract all character names and important terms from the text.

{fields}

Only include entries that appear in the text.
Return the data in the exact format specified above."""
    
    # Check if the prompt contains {fields} placeholder
    if '{fields}' in custom_prompt:
        # Get enabled types
        custom_types = get_custom_entry_types()
        
        enabled_types = [(t, cfg) for t, cfg in custom_types.items() if cfg.get('enabled', True)]
        
        # Get custom fields
        custom_fields_json = os.getenv('GLOSSARY_CUSTOM_FIELDS', '[]')
        try:
            custom_fields = json.loads(custom_fields_json)
        except:
            custom_fields = []
        
        # Build fields specification based on what the prompt expects
        # Check if the prompt mentions CSV or JSON to determine format
        if 'CSV' in custom_prompt.upper():
            # CSV format
            fields_spec = []
            
            # Show the header format
            header_parts = ['type', 'raw_name', 'translated_name', 'gender']
            if custom_fields:
                header_parts.extend(custom_fields)
            fields_spec.append(','.join(header_parts))
            
            # Show examples for each type
            for type_name, type_config in enabled_types:
                example_parts = [type_name, '<name in original language>', '<English translation>']
                
                # Add gender field
                if type_config.get('has_gender', False):
                    example_parts.append('<Male/Female/Unknown>')
                else:
                    example_parts.append('')  # Empty for non-character types
                
                # Add custom field placeholders
                for field in custom_fields:
                    example_parts.append(f'<{field} value>')
                
                fields_spec.append(','.join(example_parts))
            
            fields_str = '\n'.join(fields_spec)
        else:
            # JSON format (default)
            fields_spec = []
            fields_spec.append("Extract entities and return as a JSON array.")
            fields_spec.append("Each entry must be a JSON object with these exact fields:")
            fields_spec.append("")
            
            for type_name, type_config in enabled_types:
                fields_spec.append(f"For {type_name}s:")
                fields_spec.append(f'  "type": "{type_name}" (required)')
                fields_spec.append('  "raw_name": the name in original language/script (required)')
                fields_spec.append('  "translated_name": English translation or romanization (required)')
                if type_config.get('has_gender', False):
                    fields_spec.append('  "gender": "Male", "Female", or "Unknown" (required for characters)')
                fields_spec.append("")
            
            # Add custom fields info
            if custom_fields:
                fields_spec.append("Additional custom fields to include:")
                for field in custom_fields:
                    fields_spec.append(f'  "{field}": appropriate value')
                fields_spec.append("")
            
            # Add example
            if enabled_types:
                fields_spec.append("Example output format:")
                fields_spec.append('[')
                examples = []
                if 'character' in [t[0] for t in enabled_types]:
                    example = '  {"type": "character", "raw_name": "田中太郎", "translated_name": "Tanaka Taro", "gender": "Male"'
                    for field in custom_fields:
                        example += f', "{field}": "example value"'
                    example += '}'
                    examples.append(example)
                if 'term' in [t[0] for t in enabled_types]:
                    example = '  {"type": "term", "raw_name": "東京駅", "translated_name": "Tokyo Station"'
                    for field in custom_fields:
                        example += f', "{field}": "example value"'
                    example += '}'
                    examples.append(example)
                fields_spec.append(',\n'.join(examples))
                fields_spec.append(']')
            
            fields_str = '\n'.join(fields_spec)
        
        # Replace {fields} placeholder
        system_prompt = custom_prompt.replace('{fields}', fields_str)
    else:
        # No {fields} placeholder - use the prompt as-is
        system_prompt = custom_prompt
    
    # Remove any {chapter_text} placeholders from system prompt
    system_prompt = system_prompt.replace('{chapter_text}', '')
    system_prompt = system_prompt.replace('{{chapter_text}}', '')
    system_prompt = system_prompt.replace('{text}', '')
    system_prompt = system_prompt.replace('{{text}}', '')
    
    # Strip any trailing "Text:" or similar
    system_prompt = system_prompt.rstrip()
    if system_prompt.endswith('Text:'):
        system_prompt = system_prompt[:-5].rstrip()
    
    # User prompt is just the chapter text
    user_prompt = chapter_text
    
    return (system_prompt, user_prompt)


def skip_duplicate_entries(glossary):
    """
    Skip entries with duplicate raw names and translated names using 2-pass deduplication.
    
    Pass 1: Remove entries with similar raw names (fuzzy matching)
    Pass 2: Remove entries with identical translated names (exact matching)
    
    Returns deduplicated list maintaining first occurrence of each unique entry.
    """
    # Try to use RapidFuzz for speed, fallback to difflib
    try:
        from rapidfuzz import fuzz
        use_rapidfuzz = True
    except ImportError:
        import difflib
        use_rapidfuzz = False
    
    # Get configuration
    fuzzy_threshold = float(os.getenv('GLOSSARY_FUZZY_THRESHOLD', '0.9'))
    # GLOSSARY_DEDUPE_TRANSLATIONS: "1" = enable Pass 2 (remove entries with identical translations)
    #                              : "0" = disable Pass 2 (only remove entries with similar raw names)
    dedupe_translations = os.getenv('GLOSSARY_DEDUPE_TRANSLATIONS', '1') == '1'
    
    original_count = len(glossary)
    print(f"[Dedup] Starting 2-pass deduplication with {original_count} entries...")
    
    # Show which method we're using
    if use_rapidfuzz:
        print(f"[Dedup] Using RapidFuzz (C++ speed) with threshold {fuzzy_threshold:.2f}")
    else:
        print(f"[Dedup] Using difflib (fallback) with threshold {fuzzy_threshold:.2f}")
    
    if dedupe_translations:
        print(f"[Dedup] Pass 2 (translated name deduplication): ENABLED")
    else:
        print(f"[Dedup] Pass 2 (translated name deduplication): DISABLED")
    
    # PASS 1: Raw name deduplication (existing fuzzy matching logic)
    print(f"[Dedup] 🔄 PASS 1: Raw name deduplication...")
    pass1_results = _skip_raw_name_duplicates(glossary, fuzzy_threshold, use_rapidfuzz)
    pass1_removed = original_count - len(pass1_results)
    print(f"[Dedup] ✅ PASS 1 complete: {pass1_removed} duplicates removed ({len(pass1_results)} remaining)")
    
    # PASS 2: Translated name deduplication (if enabled)
    if dedupe_translations:
        print(f"[Dedup] 🔄 PASS 2: Translated name deduplication...")
        final_results = _skip_translated_name_duplicates(pass1_results)
        pass2_removed = len(pass1_results) - len(final_results)
        print(f"[Dedup] ✅ PASS 2 complete: {pass2_removed} duplicates removed ({len(final_results)} remaining)")
        total_removed = pass1_removed + pass2_removed
    else:
        final_results = pass1_results
        total_removed = pass1_removed
        print(f"[Dedup] ⏭️ PASS 2 skipped (translation deduplication disabled)")
    
    if total_removed > 0:
        print(f"⏭️ Total skipped: {total_removed} duplicate entries")
        print(f"✅ Total kept: {len(final_results)} unique entries")
    
    return final_results


def _skip_raw_name_duplicates(glossary, fuzzy_threshold, use_rapidfuzz):
    """Pass 1: Remove entries with similar raw names using fuzzy matching"""
    if use_rapidfuzz:
        from rapidfuzz import fuzz
    else:
        import difflib
    
    seen_raw_names = []  # List of (cleaned_name, original_entry) tuples
    deduplicated = []
    skipped_count = 0
    
    for entry in glossary:
        # Get raw_name and clean it
        raw_name = entry.get('raw_name', '')
        if not raw_name:
            continue
            
        # Remove honorifics for comparison (unless disabled)
        cleaned_name = remove_honorifics(raw_name)
        
        # Check for fuzzy matches with seen names using advanced multi-algorithm approach
        is_duplicate = False
        best_score = 0.0
        best_match = None
        
        for seen_clean, seen_original in seen_raw_names:
            # Try multiple algorithms for better accuracy
            scores = []
            
            if use_rapidfuzz:
                # RapidFuzz: Multiple methods
                basic = fuzz.ratio(cleaned_name.lower(), seen_clean.lower()) / 100.0
                scores.append(basic)
                
                # Token sort (handles word order: "Kim Sang" vs "Sang Kim")
                try:
                    token_sort = fuzz.token_sort_ratio(cleaned_name.lower(), seen_clean.lower()) / 100.0
                    scores.append(token_sort)
                except:
                    pass
                
                # Partial ratio (substring matching: "김상현" in "김상현님")
                try:
                    partial = fuzz.partial_ratio(cleaned_name.lower(), seen_clean.lower()) / 100.0
                    scores.append(partial)
                except:
                    pass
            else:
                # Fallback to difflib
                basic = difflib.SequenceMatcher(None, cleaned_name.lower(), seen_clean.lower()).ratio()
                scores.append(basic)
            
            # Try Jaro-Winkler (better for names)
            try:
                import jellyfish
                jaro = jellyfish.jaro_winkler_similarity(cleaned_name, seen_clean)
                scores.append(jaro)
            except ImportError:
                pass
            
            # Take the best score from all algorithms
            similarity = max(scores) if scores else 0.0
            
            if similarity >= fuzzy_threshold:
                if similarity > best_score:
                    best_score = similarity
                    best_match = seen_original
                is_duplicate = True
        
        if is_duplicate:
            # Find the existing entry to compare field counts
            existing_entry = None
            for i, existing in enumerate(deduplicated):
                if existing.get('raw_name') == best_match:
                    existing_entry = existing
                    existing_index = i
                    break
            
            if existing_entry:
                # Count fields in both entries
                current_field_count = len([v for v in entry.values() if v and str(v).strip()])
                existing_field_count = len([v for v in existing_entry.values() if v and str(v).strip()])
                
                # If current entry has more fields, replace the existing one
                if current_field_count > existing_field_count:
                    # Replace existing entry
                    deduplicated[existing_index] = entry
                    # Update seen_raw_names
                    for j, (seen_clean, seen_original) in enumerate(seen_raw_names):
                        if seen_original == best_match:
                            seen_raw_names[j] = (cleaned_name, entry.get('raw_name', ''))
                            break
                    skipped_count += 1
                    if skipped_count <= 10:
                        print(f"[Skip] Pass 1: Replacing {best_match} ({existing_field_count} fields) with {raw_name} ({current_field_count} fields) - {best_score*100:.1f}% match, more detailed entry")
                else:
                    # Keep existing entry
                    skipped_count += 1
                    if skipped_count <= 10:
                        extra_info = f" ({current_field_count} vs {existing_field_count} fields)" if current_field_count != existing_field_count else ""
                        print(f"[Skip] Pass 1: {raw_name} (cleaned: {cleaned_name}) - {best_score*100:.1f}% match with {best_match}{extra_info}")
            else:
                # Fallback if we can't find the existing entry
                skipped_count += 1
                if skipped_count <= 10:
                    print(f"[Skip] Pass 1: {raw_name} (cleaned: {cleaned_name}) - {best_score*100:.1f}% match with {best_match}")
        else:
            # Add to seen list and keep the entry
            seen_raw_names.append((cleaned_name, entry.get('raw_name', '')))
            deduplicated.append(entry)
    
    return deduplicated


def _skip_translated_name_duplicates(glossary):
    """Pass 2: Remove entries with identical translated names"""
    seen_translations = {}  # translated_name.lower() -> (raw_name, entry)
    deduplicated = []
    skipped_count = 0
    
    for entry in glossary:
        raw_name = entry.get('raw_name', '')
        translated_name = entry.get('translated_name', '')
        translated_lower = translated_name.lower().strip()
        
        # Skip empty translations
        if not translated_lower:
            deduplicated.append(entry)
            continue
        
        # Check if we've seen this translation before
        if translated_lower in seen_translations:
            existing_raw, existing_entry = seen_translations[translated_lower]
            existing_translated = existing_entry.get('translated_name', translated_name)
            
            # Count fields in both entries (more fields = higher priority)
            current_field_count = len([v for v in entry.values() if v and str(v).strip()])
            existing_field_count = len([v for v in existing_entry.values() if v and str(v).strip()])
            
            # If current entry has more fields, replace the existing one
            if current_field_count > existing_field_count:
                # Remove existing entry from deduplicated list
                deduplicated = [e for e in deduplicated if e != existing_entry]
                # Replace with current entry
                seen_translations[translated_lower] = (raw_name, entry)
                deduplicated.append(entry)
                skipped_count += 1
                if skipped_count <= 10:  # Only log first few
                    print(f"[Skip] Pass 2: Replacing '{existing_raw}' -> '{existing_translated}' ({existing_field_count} fields) with '{raw_name}' -> '{translated_name}' ({current_field_count} fields) - more detailed entry")
            else:
                # Keep existing entry (has same or more fields)
                skipped_count += 1
                if skipped_count <= 10:  # Only log first few
                    extra_info = f" ({current_field_count} vs {existing_field_count} fields)" if current_field_count != existing_field_count else ""
                    print(f"[Skip] Pass 2: Removing '{raw_name}' -> '{translated_name}' (duplicate translation of '{existing_raw}' -> '{existing_translated}'){extra_info}")
        else:
            # New translation, keep it
            seen_translations[translated_lower] = (raw_name, entry)
            deduplicated.append(entry)
    
    return deduplicated


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
                
            # Get system and user prompts
            system_prompt, user_prompt = build_prompt(chap)

            # Build messages correctly with system and user prompts
            if not contextual_enabled:
                msgs = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            else:
                msgs = [{"role": "system", "content": system_prompt}] \
                     + trim_context_history(history, ctx_limit, rolling_window) \
                     + [{"role": "user", "content": user_prompt}]

            
            # Submit to thread pool
            future = executor.submit(
                process_single_chapter_api_call,
                idx, chap, msgs, client, temp, mtoks, check_stop, chunk_timeout
            )
            futures[future] = (idx, chap)
        
        # Process results with better cancellation
        for future in as_completed(futures):  # Removed timeout - let futures complete
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
                # Ensure chap is added to result here if not already present
                if 'chap' not in result:
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
    
    # Sort results by chapter index
    results.sort(key=lambda x: x['idx'])
    return results

def process_single_chapter_api_call(idx: int, chap: str, msgs: List[Dict], 
                                  client: UnifiedClient, temp: float, mtoks: int,
                                  stop_check_fn, chunk_timeout: int = None) -> Dict:
    """Process a single chapter API call with thread-safe payload handling"""
    
    # APPLY INTERRUPTIBLE THREADING DELAY FIRST
    thread_delay = float(os.getenv("THREAD_SUBMISSION_DELAY_SECONDS", "0.5"))
    if thread_delay > 0:
        # Check if we need to wait (same logic as unified_api_client)
        if hasattr(client, '_thread_submission_lock') and hasattr(client, '_last_thread_submission_time'):
            with client._thread_submission_lock:
                current_time = time.time()
                time_since_last = current_time - client._last_thread_submission_time
                
                if time_since_last < thread_delay:
                    sleep_time = thread_delay - time_since_last
                    thread_name = threading.current_thread().name
                    
                    # PRINT BEFORE THE DELAY STARTS
                    print(f"🧵 [{thread_name}] Applying thread delay: {sleep_time:.1f}s for Chapter {idx+1}")
                    
                    # Interruptible sleep - check stop flag every 0.1 seconds
                    elapsed = 0
                    check_interval = 0.1
                    while elapsed < sleep_time:
                        if stop_check_fn():
                            print(f"🛑 Threading delay interrupted by stop flag")
                            raise UnifiedClientError("Glossary extraction stopped by user during threading delay")
                        
                        sleep_chunk = min(check_interval, sleep_time - elapsed)
                        time.sleep(sleep_chunk)
                        elapsed += sleep_chunk
                
                client._last_thread_submission_time = time.time()
                if not hasattr(client, '_thread_submission_count'):
                    client._thread_submission_count = 0
                client._thread_submission_count += 1
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

        # Handle the response - it might be a tuple or a string
        if raw is None:
            print(f"⚠️ API returned None for chapter {idx+1}")
            return {
                'idx': idx,
                'data': [],
                'resp': "",
                'chap': chap,
                'error': "API returned None"
            }

        if isinstance(raw, tuple):
            resp = raw[0] if raw[0] is not None else ""
        elif isinstance(raw, str):
            resp = raw
        elif hasattr(raw, 'content'):
            resp = raw.content if raw.content is not None else ""
        elif hasattr(raw, 'text'):
            resp = raw.text if raw.text is not None else ""
        else:
            resp = str(raw) if raw is not None else ""

        # Ensure resp is never None
        if resp is None:
            resp = ""
        
        # Save the raw response in thread-safe location
        response_file = os.path.join(thread_dir, f"chapter_{idx+1}_response.txt")
        with open(response_file, "w", encoding="utf-8", errors="replace") as f:
            f.write(resp)
        
        # Parse response using the new parser
        data = parse_api_response(resp)
        
        # More detailed debug logging
        print(f"[BATCH] Chapter {idx+1} - Raw response length: {len(resp)} chars")
        print(f"[BATCH] Chapter {idx+1} - Parsed {len(data)} entries before validation")
        
        # Filter out invalid entries
        valid_data = []
        for entry in data:
            if validate_extracted_entry(entry):
                # Clean the raw_name
                if 'raw_name' in entry:
                    entry['raw_name'] = entry['raw_name'].strip()
                valid_data.append(entry)
            else:
                print(f"[BATCH] Chapter {idx+1} - Invalid entry: {entry}")
        
        elapsed = time.time() - start_time
        print(f"[BATCH] Completed Chapter {idx+1} in {elapsed:.1f}s at {time.strftime('%H:%M:%S')} - Extracted {len(valid_data)} valid entries")
        
        return {
            'idx': idx,
            'data': valid_data,
            'resp': resp,
            'chap': chap,  # Include the chapter text in the result
            'error': None
        }
            
    except UnifiedClientError as e:
        print(f"[Error] API call interrupted/failed for chapter {idx+1}: {e}")
        return {
            'idx': idx,
            'data': [],
            'resp': "",
            'chap': chap,  # Include chapter even on error
            'error': str(e)
        }
    except Exception as e:
        print(f"[Error] Unexpected error for chapter {idx+1}: {e}")
        import traceback
        print(f"[Error] Traceback: {traceback.format_exc()}")
        return {
            'idx': idx,
            'data': [],
            'resp': "",
            'chap': chap,  # Include chapter even on error
            'error': str(e)
        }

# Update main function to support batch processing:
def main(log_callback=None, stop_callback=None):
    # Declare global variables at the very start of the function
    global _skipped_chapters
    
    # Redirect print/logs to callback if provided
    if log_callback:
        set_output_redirect(log_callback)
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
        
        args = parser.parse_args()
        epub_path = args.epub
    else:
        # GUI mode - get from environment
        epub_path = os.getenv("EPUB_PATH", "")
        if not epub_path and len(sys.argv) > 1:
            epub_path = sys.argv[1]
        
        # Create args object for GUI mode
        import types
        args = types.SimpleNamespace()
        args.epub = epub_path
        args.output = os.getenv("OUTPUT_PATH", "glossary.json")
        args.config = os.getenv("CONFIG_PATH", "config.json")

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
        f"{file_base}_glossary_progress.json"
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
    client = create_client_with_multi_key_support(api_key, model, out, config)
    
    # Check for batch mode
    batch_enabled = os.getenv("BATCH_TRANSLATION", "0") == "1"
    batch_size = int(os.getenv("BATCH_SIZE", "5"))
    conservative_batching = os.getenv("CONSERVATIVE_BATCHING", "0") == "1"
    
    print(f"[DEBUG] BATCH_TRANSLATION = {os.getenv('BATCH_TRANSLATION')} (enabled: {batch_enabled})")
    print(f"[DEBUG] BATCH_SIZE = {batch_size}")
    print(f"[DEBUG] CONSERVATIVE_BATCHING = {os.getenv('CONSERVATIVE_BATCHING')} (enabled: {conservative_batching})")
    
    if batch_enabled:
        print(f"🚀 Glossary batch mode enabled with size: {batch_size}")
        print(f"📑 Note: Glossary extraction uses direct batching (not affected by conservative batching setting)")
    
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
    format_parts = ["type", "raw_name", "translated_name", "gender"]
    custom_fields_json = os.getenv('GLOSSARY_CUSTOM_FIELDS', '[]')
    try:
        custom_fields = json.loads(custom_fields_json)
        if custom_fields:
            format_parts.extend(custom_fields)
    except:
        pass
    print(f"📑 Glossary Format: Simple ({', '.join(format_parts)})")
    
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
        # BATCH MODE: Process in batches with per-entry saving
        total_batches = (len(chapters_to_process) + batch_size - 1) // batch_size
        
        for batch_num in range(total_batches):
            # Check for stop at the beginning of each batch
            if check_stop():
                print(f"❌ Glossary extraction stopped at batch {batch_num+1}")
                # Apply deduplication before stopping
                if glossary:
                    print("🔀 Applying deduplication and sorting before exit...")
                    glossary[:] = skip_duplicate_entries(glossary)
                    
                    # Sort glossary
                    custom_types = get_custom_entry_types()
                    type_order = {'character': 0, 'term': 1}
                    other_types = sorted([t for t in custom_types.keys() if t not in ['character', 'term']])
                    for i, t in enumerate(other_types):
                        type_order[t] = i + 2
                    glossary.sort(key=lambda x: (
                        type_order.get(x.get('type', 'term'), 999),
                        x.get('raw_name', '').lower()
                    ))
                    
                    save_progress(completed, glossary, history)
                    save_glossary_json(glossary, os.path.join(glossary_dir, os.path.basename(args.output)))
                    save_glossary_csv(glossary, os.path.join(glossary_dir, os.path.basename(args.output)))
                    print(f"✅ Saved {len(glossary)} deduplicated entries before exit")
                return
            
            # Get current batch
            batch_start = batch_num * batch_size
            batch_end = min(batch_start + batch_size, len(chapters_to_process))
            current_batch = chapters_to_process[batch_start:batch_end]
            
            print(f"\n🔄 Processing Batch {batch_num+1}/{total_batches} (Chapters: {[idx+1 for idx, _ in current_batch]})")
            print(f"[BATCH] Submitting {len(current_batch)} chapters for parallel processing...")
            batch_start_time = time.time()
            
            # Process batch in parallel BUT handle results as they complete
            temp = float(os.getenv("GLOSSARY_TEMPERATURE") or config.get('temperature', 0.1))
            env_max_output = os.getenv("MAX_OUTPUT_TOKENS")
            if env_max_output and env_max_output.isdigit():
                mtoks = int(env_max_output)
            else:
                mtoks = config.get('max_tokens', 4196)
            
            batch_entry_count = 0
            stopped_early = False
            
            with ThreadPoolExecutor(max_workers=len(current_batch)) as executor:
                futures = {}
                
                # Submit all chapters in the batch
                for idx, chap in current_batch:
                    if check_stop():
                        stopped_early = True
                        break
                        
                    # Get system and user prompts
                    system_prompt, user_prompt = build_prompt(chap)
                    
                    # Build messages
                    if not contextual_enabled:
                        msgs = [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ]
                    else:
                        msgs = [{"role": "system", "content": system_prompt}] \
                             + trim_context_history(history, ctx_limit, rolling_window) \
                             + [{"role": "user", "content": user_prompt}]
                    
                    # Submit to thread pool
                    future = executor.submit(
                        process_single_chapter_api_call,
                        idx, chap, msgs, client, temp, mtoks, check_stop, chunk_timeout
                    )
                    futures[future] = (idx, chap)
                    # Small yield to keep GUI responsive when submitting many tasks
                    if idx % 5 == 0:
                        time.sleep(0.001)
                    # Small yield to keep GUI responsive when submitting many tasks
                    if idx % 5 == 0:
                        time.sleep(0.001)
                
                # Process results AS THEY COMPLETE, not all at once
                for future in as_completed(futures):
                    if check_stop():
                        print("🛑 Stop detected - cancelling all pending operations...")
                        stopped_early = True
                        cancelled = cancel_all_futures(list(futures.keys()))
                        if cancelled > 0:
                            print(f"✅ Cancelled {cancelled} pending API calls")
                        executor.shutdown(wait=False)
                        break
                    
                    idx, chap = futures[future]
                    
                    try:
                        result = future.result(timeout=0.5)
                        
                        # Process this chapter's results immediately
                        data = result.get('data', [])
                        resp = result.get('resp', '')
                        error = result.get('error')
                        
                        if error:
                            print(f"[Chapter {idx+1}] Error: {error}")
                            completed.append(idx)
                            continue
                        
                        # Process and save entries IMMEDIATELY as each chapter completes
                        if data and len(data) > 0:
                            total_ent = len(data)
                            batch_entry_count += total_ent
                            
                            for eidx, entry in enumerate(data, start=1):
                                elapsed = time.time() - start
                                
                                # Get entry info
                                entry_type = entry.get("type", "?")
                                raw_name = entry.get("raw_name", "?")
                                trans_name = entry.get("translated_name", "?")
                                
                                print(f'[Chapter {idx+1}/{total_chapters}] [{eidx}/{total_ent}] ({elapsed:.1f}s elapsed) → {entry_type}: {raw_name} ({trans_name})')
                                
                                # Add entry immediately WITHOUT deduplication
                                glossary.append(entry)
                                
                                # Save immediately after EACH entry
                                save_progress(completed, glossary, history)
                                save_glossary_json(glossary, os.path.join(glossary_dir, os.path.basename(args.output)))
                                save_glossary_csv(glossary, os.path.join(glossary_dir, os.path.basename(args.output)))
                        
                        completed.append(idx)
                        
                        # Add to history if contextual is enabled
                        if contextual_enabled and resp and chap:
                            system_prompt, user_prompt = build_prompt(chap)
                            history.append({"user": user_prompt, "assistant": resp})
                        
                    except Exception as e:
                        if "stopped by user" in str(e).lower():
                            print(f"✅ Chapter {idx+1} stopped by user")
                        else:
                            print(f"Error processing chapter {idx+1}: {e}")
                        completed.append(idx)
            
            batch_elapsed = time.time() - batch_start_time
            print(f"[BATCH] Batch {batch_num+1} completed in {batch_elapsed:.1f}s total")
            
            # After batch completes, apply deduplication and sorting (only if not stopped early)
            if batch_entry_count > 0 and not stopped_early:
                print(f"\n🔀 Applying deduplication and sorting after batch {batch_num+1}/{total_batches}")
                original_size = len(glossary)
                
                # Apply deduplication to entire glossary
                glossary[:] = skip_duplicate_entries(glossary)
                
                # Sort glossary by type and name
                custom_types = get_custom_entry_types()
                type_order = {'character': 0, 'term': 1}
                other_types = sorted([t for t in custom_types.keys() if t not in ['character', 'term']])
                for i, t in enumerate(other_types):
                    type_order[t] = i + 2
                
                glossary.sort(key=lambda x: (
                    type_order.get(x.get('type', 'term'), 999),
                    x.get('raw_name', '').lower()
                ))
                
                deduplicated_size = len(glossary)
                removed = original_size - deduplicated_size
                
                if removed > 0:
                    print(f"✅ Removed {removed} duplicates (fuzzy threshold: {os.getenv('GLOSSARY_FUZZY_THRESHOLD', '0.90')})")
                print(f"📊 Glossary size: {deduplicated_size} unique entries")
                
                # Save final deduplicated and sorted glossary
                save_progress(completed, glossary, history)
                save_glossary_json(glossary, os.path.join(glossary_dir, os.path.basename(args.output)))
                save_glossary_csv(glossary, os.path.join(glossary_dir, os.path.basename(args.output)))
            
            # Print batch summary
            if batch_entry_count > 0:
                print(f"\n📊 Batch {batch_num+1}/{total_batches} Summary:")
                print(f"   • Chapters processed: {len(current_batch)}")
                print(f"   • Total entries extracted: {batch_entry_count}")
                print(f"   • Glossary size: {len(glossary)} unique entries")
            
            # If stopped early, deduplicate once and exit
            if stopped_early:
                if glossary:
                    print(f"\n🔀 Deduplicating {len(glossary)} entries before exit...")
                    glossary[:] = skip_duplicate_entries(glossary)
                    
                    custom_types = get_custom_entry_types()
                    type_order = {'character': 0, 'term': 1}
                    other_types = sorted([t for t in custom_types.keys() if t not in ['character', 'term']])
                    for i, t in enumerate(other_types):
                        type_order[t] = i + 2
                    glossary.sort(key=lambda x: (
                        type_order.get(x.get('type', 'term'), 999),
                        x.get('raw_name', '').lower()
                    ))
                    
                    save_progress(completed, glossary, history)
                    save_glossary_json(glossary, os.path.join(glossary_dir, os.path.basename(args.output)))
                    save_glossary_csv(glossary, os.path.join(glossary_dir, os.path.basename(args.output)))
                    print(f"✅ Saved {len(glossary)} deduplicated entries before exit")
                return
            
            # Handle context history
            if contextual_enabled:
                if not rolling_window and len(history) >= ctx_limit and ctx_limit > 0:
                    print(f"🔄 Resetting glossary context (reached {ctx_limit} chapter limit)")
                    history = []
                    prog['context_history'] = []
            
            # Add delay between batches (but not after the last batch)
            if batch_num < total_batches - 1:
                print(f"\n⏱️  Waiting {api_delay}s before next batch...")
                if not interruptible_sleep(api_delay, check_stop, 0.1):
                    print(f"❌ Glossary extraction stopped during delay")
                    # Apply deduplication before stopping
                    if glossary:
                        print("🔀 Applying deduplication and sorting before exit...")
                        glossary[:] = skip_duplicate_entries(glossary)
                        
                        # Sort glossary
                        custom_types = get_custom_entry_types()
                        type_order = {'character': 0, 'term': 1}
                        other_types = sorted([t for t in custom_types.keys() if t not in ['character', 'term']])
                        for i, t in enumerate(other_types):
                            type_order[t] = i + 2
                        glossary.sort(key=lambda x: (
                            type_order.get(x.get('type', 'term'), 999),
                            x.get('raw_name', '').lower()
                        ))
                        
                        save_progress(completed, glossary, history)
                        save_glossary_json(glossary, os.path.join(glossary_dir, os.path.basename(args.output)))
                        save_glossary_csv(glossary, os.path.join(glossary_dir, os.path.basename(args.output)))
                        print(f"✅ Saved {len(glossary)} deduplicated entries before exit")
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
                    # Track skipped chapters for summary (don't print individually)
                    if '_skipped_chapters' not in globals():
                        _skipped_chapters = []
                    is_text_chapter = hasattr(chap, 'filename') and chap.get('filename', '').endswith('.txt')
                    terminology = "Section" if is_text_chapter else "Chapter"
                    _skipped_chapters.append((chapter_num, terminology))
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
                # Get system and user prompts from build_prompt
                system_prompt, user_prompt = build_prompt(chap)
                
                if not contextual_enabled:
                    # No context at all
                    msgs = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ]
                else:
                    # Use context with trim_context_history handling the mode
                    msgs = [{"role": "system", "content": system_prompt}] \
                         + trim_context_history(history, ctx_limit, rolling_window) \
                         + [{"role": "user", "content": user_prompt}]
                
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
                    system_tokens = chapter_splitter.count_tokens(system_prompt)
                    context_tokens = sum(chapter_splitter.count_tokens(m["content"]) for m in trim_context_history(history, ctx_limit, rolling_window))
                    safety_margin = 1000
                    available_tokens = token_limit - system_tokens - context_tokens - safety_margin
                    
                    # Since glossary extraction works with plain text, wrap it in a simple HTML structure
                    chapter_html = f"<html><body><p>{chap.replace(chr(10)+chr(10), '</p><p>')}</p></body></html>"
                    
                    # Use ChapterSplitter to split the chapter
                    # No filename passed as this is EPUB content (not plain text files)
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
                        
                        # Get system and user prompts for chunk
                        chunk_system_prompt, chunk_user_prompt = build_prompt(chunk_text)

                        # Build chunk messages
                        if not contextual_enabled:
                            chunk_msgs = [
                                {"role": "system", "content": chunk_system_prompt},
                                {"role": "user", "content": chunk_user_prompt}
                            ]
                        else:
                            chunk_msgs = [{"role": "system", "content": chunk_system_prompt}] \
                                       + trim_context_history(history, ctx_limit, rolling_window) \
                                       + [{"role": "user", "content": chunk_user_prompt}]

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
                        if chunk_raw is None:
                            print(f"❌ API returned None for chunk {chunk_idx}")
                            continue

                        # Handle different response types
                        if isinstance(chunk_raw, tuple):
                            chunk_resp = chunk_raw[0] if chunk_raw[0] is not None else ""
                        elif isinstance(chunk_raw, str):
                            chunk_resp = chunk_raw
                        elif hasattr(chunk_raw, 'content'):
                            chunk_resp = chunk_raw.content if chunk_raw.content is not None else ""
                        elif hasattr(chunk_raw, 'text'):
                            chunk_resp = chunk_raw.text if chunk_raw.text is not None else ""
                        else:
                            print(f"❌ Unexpected response type for chunk {chunk_idx}: {type(chunk_raw)}")
                            chunk_resp = str(chunk_raw) if chunk_raw is not None else ""

                        # Ensure resp is a string
                        if not isinstance(chunk_resp, str):
                            print(f"⚠️ Converting non-string response to string for chunk {chunk_idx}")
                            chunk_resp = str(chunk_resp) if chunk_resp is not None else ""

                        # Check if response is empty
                        if not chunk_resp or chunk_resp.strip() == "":
                            print(f"⚠️ Empty response for chunk {chunk_idx}, skipping...")
                            continue
                        
                        # Save chunk response with thread-safe location
                        thread_name = threading.current_thread().name
                        thread_id = threading.current_thread().ident
                        thread_dir = os.path.join("Payloads", "glossary", f"{thread_name}_{thread_id}")
                        os.makedirs(thread_dir, exist_ok=True)
                        
                        with open(os.path.join(thread_dir, f"chunk_response_chap{idx+1}_chunk{chunk_idx}.txt"), "w", encoding="utf-8", errors="replace") as f:
                            f.write(chunk_resp)
                        
                        # Extract data from chunk
                        chunk_resp_data = parse_api_response(chunk_resp)

                        if not chunk_resp_data:
                            print(f"[Warning] No data found in chunk {chunk_idx}, skipping...")
                            continue

                        # The parse_api_response already returns parsed data, no need to parse again
                        try:
                            # Filter out invalid entries directly from chunk_resp_data
                            valid_chunk_data = []
                            for entry in chunk_resp_data:
                                if validate_extracted_entry(entry):
                                    # Clean the raw_name
                                    if 'raw_name' in entry:
                                        entry['raw_name'] = entry['raw_name'].strip()
                                    valid_chunk_data.append(entry)
                                else:
                                    print(f"[Debug] Skipped invalid entry in chunk {chunk_idx}: {entry}")
                            
                            chapter_glossary_data.extend(valid_chunk_data)
                            print(f"✅ Chunk {chunk_idx}/{total_chunks}: extracted {len(valid_chunk_data)} entries")
                            
                            # Add chunk to history if contextual
                            if contextual_enabled:
                                history.append({"user": chunk_user_prompt, "assistant": chunk_resp})

                        except Exception as e:
                            print(f"[Warning] Error processing chunk {chunk_idx} data: {e}")
                            continue
                        
                        # Add delay between chunks (but not after last chunk)
                        if chunk_idx < total_chunks:
                            print(f"⏱️  Waiting {api_delay}s before next chunk...")
                            if not interruptible_sleep(api_delay, check_stop, 0.1):
                                print(f"❌ Glossary extraction stopped during chunk delay")
                                return
                    
                    # Use the collected data from all chunks
                    data = chapter_glossary_data
                    resp = ""  # Combined response not needed for progress tracking
                    print(f"✅ Chapter {idx+1} processed in {len(chunks)} chunks, total entries: {len(data)}")
                    
                else:
                    # Original single-chapter processing
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
                    
                    # Handle response
                    if raw is None:
                        print(f"❌ API returned None for chapter {idx+1}")
                        continue

                    # Handle different response types
                    if isinstance(raw, tuple):
                        resp = raw[0] if raw[0] is not None else ""
                    elif isinstance(raw, str):
                        resp = raw
                    elif hasattr(raw, 'content'):
                        resp = raw.content if raw.content is not None else ""
                    elif hasattr(raw, 'text'):
                        resp = raw.text if raw.text is not None else ""
                    else:
                        print(f"❌ Unexpected response type for chapter {idx+1}: {type(raw)}")
                        resp = str(raw) if raw is not None else ""

                    # Ensure resp is a string
                    if not isinstance(resp, str):
                        print(f"⚠️ Converting non-string response to string for chapter {idx+1}")
                        resp = str(resp) if resp is not None else ""

                    # NULL CHECK before checking if response is empty
                    if resp is None:
                        print(f"⚠️ Response is None for chapter {idx+1}, skipping...")
                        continue

                    # Check if response is empty
                    if not resp or resp.strip() == "":
                        print(f"⚠️ Empty response for chapter {idx+1}, skipping...")
                        continue

                    # Save the raw response with thread-safe location
                    thread_name = threading.current_thread().name
                    thread_id = threading.current_thread().ident
                    thread_dir = os.path.join("Payloads", "glossary", f"{thread_name}_{thread_id}")
                    os.makedirs(thread_dir, exist_ok=True)
                    
                    with open(os.path.join(thread_dir, f"response_chap{idx+1}.txt"), "w", encoding="utf-8", errors="replace") as f:
                        f.write(resp)

                    # Parse response using the new parser
                    try:
                        data = parse_api_response(resp)
                    except Exception as e:
                        print(f"❌ Error parsing response for chapter {idx+1}: {e}")
                        print(f"   Response preview: {resp[:200] if resp else 'None'}...")
                        continue
                    
                    # Filter out invalid entries
                    valid_data = []
                    for entry in data:
                        if validate_extracted_entry(entry):
                            # Clean the raw_name
                            if 'raw_name' in entry:
                                entry['raw_name'] = entry['raw_name'].strip()
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
                if contextual_enabled and 'resp' in locals() and resp:
                    history.append({"user": user_prompt, "assistant": resp})
                    
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
                import traceback
                print(f"Full traceback: {traceback.format_exc()}")
                # Check for stop even after error
                if check_stop():
                    print(f"❌ Glossary extraction stopped after error in chapter {idx+1}")
                    return
    
    # Print skip summary if any chapters were skipped
    if '_skipped_chapters' in globals() and _skipped_chapters:
        skipped = _skipped_chapters
        print(f"\n📊 Skipped {len(skipped)} chapters outside range {range_start}-{range_end}")
        if len(skipped) <= 10:
            chapter_list = ', '.join([f"{term} {num}" for num, term in skipped])
            print(f"   Skipped: {chapter_list}")
        else:
            chapter_nums = [num for num, _ in skipped]
            print(f"   Range: {min(chapter_nums)} to {max(chapter_nums)}")
        # Clear the list
        _skipped_chapters = []
    
    print(f"\nDone. Glossary saved to {args.output}")
    
    # Also save as CSV format for compatibility
    try:
        csv_output = args.output.replace('.json', '.csv')
        csv_path = os.path.join(glossary_dir, os.path.basename(csv_output))
        save_glossary_csv(glossary, os.path.join(glossary_dir, os.path.basename(args.output)))
        print(f"Also saved as CSV: {csv_path}")
    except Exception as e:
        print(f"[Warning] Could not save CSV format: {e}")

def save_progress(completed: List[int], glossary: List[Dict], context_history: List[Dict]):
    """Save progress to JSON file"""
    global _progress_lock
    
    # Acquire lock to prevent concurrent writes
    with _progress_lock:
        progress_data = {
            "completed": completed,
            "glossary": glossary,
            "context_history": context_history
        }
        
        try:
            # Use atomic write with proper temp file handling
            progress_dir = os.path.dirname(PROGRESS_FILE) or '.'
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=progress_dir, delete=False, suffix='.tmp') as temp_f:
                temp_path = temp_f.name
                json.dump(progress_data, temp_f, ensure_ascii=False, indent=2)
                temp_f.flush()
                os.fsync(temp_f.fileno())  # Ensure data is written to disk
            
            # Atomic rename
            try:
                if os.path.exists(PROGRESS_FILE):
                    os.remove(PROGRESS_FILE)
                os.rename(temp_path, PROGRESS_FILE)
            except Exception as e:
                if os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except:
                        pass
                raise
            
        except Exception as e:
            print(f"[Warning] Failed to save progress: {e}")
            # Try direct write as fallback
            try:
                with open(PROGRESS_FILE, 'w', encoding='utf-8') as f:
                    json.dump(progress_data, f, ensure_ascii=False, indent=2)
            except Exception as e2:
                print(f"[Error] Could not save progress: {e2}")
            
if __name__=='__main__':
    main()