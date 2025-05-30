import os
import json
import argparse
import zipfile
import time
import sys
import tiktoken

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
    # 1) take only the last `limit` entries
    recent = history[-limit:]

    # 2) convert each entry into a user+assistant message pair
    trimmed = []
    for entry in recent:
        trimmed.append({"role": "user",      "content": entry["user"]})
        trimmed.append({"role": "assistant", "content": entry["assistant"]})
    return trimmed

def build_prompt(chapter_text: str) -> str:
    return f"""
Output exactly a JSON array of objects and nothing else.
You are a glossary extractor for Korean, Japanese, or Chinese novels.
- Extract character information (e.g., name, traits), locations (countries, regions, cities), and translate them into English (romanization or equivalent).
- Romanize all untranslated honorifics and suffixes (e.g., 님 to '-nim', さん to '-san').
- all output must be in english, unless specified otherwise
For each character, provide JSON fields:
- original_name: name in the original script
- name: English/romanized name
- gender
- title (with romanized suffix)
- group_affiliation
- traits
- how_they_refer_to_others (mapping with romanized suffix)
- locations: list of place names mentioned (inlude the original language in brackets)
Sort by appearance order; respond with a JSON array only.

Text:
{chapter_text}
"""

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

def merge_glossary_entries(glossary):
    merged = {}
    for entry in glossary:
        key = entry['original_name']
        if key not in merged:
            merged[key] = entry.copy()
            # initial normalize locations in appearance order
            merged[key]['locations'] = dedupe_keep_order(
                entry.get('locations') or [], []
            )
        else:
            for field in ['locations', 'traits', 'group_affiliation']:
                old = merged[key].get(field) or []
                new = entry.get(field) or []
                merged[key][field] = dedupe_keep_order(old, new)

            # how_they_refer_to_others stays the same…
            old_map = merged[key].get('how_they_refer_to_others', {}) or {}
            new_map = entry.get('how_they_refer_to_others', {}) or {}
            for k, v in new_map.items():
                if v is not None and k not in old_map:
                    old_map[k] = v
            merged[key]['how_they_refer_to_others'] = old_map

    # strip out any None fields exactly as before…
    for entry in merged.values():
        for field in ['title','group_affiliation','traits','locations','gender']:
            if entry.get(field) is None:
                entry.pop(field, None)
        # only sanitize how_they_refer_to_others if it's actually a dict
        htr = entry.get('how_they_refer_to_others')
        if isinstance(htr, dict):
            entry['how_they_refer_to_others'] = {
                k: v for k, v in htr.items() if v is not None
            }
        else:
            # drop it entirely if it was null or something else
            entry.pop('how_they_refer_to_others', None)

    return list(merged.values())

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
    client = UnifiedClient(model=config['model'], api_key=config['api_key'])
    model = config.get('model', 'gpt-4.1-mini')
    temp = config.get('temperature', 0.3)
    env_max_output = os.getenv("MAX_OUTPUT_TOKENS")
    if env_max_output and env_max_output.isdigit():
        mtoks = int(env_max_output)
        print(f"[DEBUG] Output Token Limit: {mtoks} (from GUI)")
    else:
        mtoks = config.get('max_tokens', 4196)
        print(f"[DEBUG] Output Token Limit: {mtoks} (from config)")
    sys_prompt = config.get('system_prompt', 'You are a helpful assistant.')
    ctx_limit = config.get('context_limit_chapters', 3)

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
        
        try:
            msgs = [{"role":"system","content":sys_prompt}] \
                 + trim_context_history(history, ctx_limit) \
                 + [{"role":"user","content":build_prompt(chap)}]

            total_tokens = sum(count_tokens(m["content"]) for m in msgs)
            
            # READ THE TOKEN LIMIT RIGHT HERE, RIGHT NOW
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
            
            # Run API call in a separate thread with timeout checking
            import threading
            import queue
            
            result_queue = queue.Queue()
            api_error = None
            
            def api_call():
                try:
                    result = client.send(msgs, temperature=temp, max_tokens=mtoks)
                    result_queue.put(result)
                except Exception as e:
                    result_queue.put(e)
            
            api_thread = threading.Thread(target=api_call)
            api_thread.daemon = True
            api_thread.start()
            
            # Check for stop every 0.5 seconds while waiting for API
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
                
            # if send() returned (text, finish_reason), pull out the text
            resp = raw[0] if isinstance(raw, tuple) else raw

            # Save the raw response in case you need to inspect it
            os.makedirs("Payloads", exist_ok=True)
            # after
            with open(f"Payloads/failed_response_chap{idx+1}.txt", "w", encoding="utf-8", errors="replace") as f:
                f.write(resp)

            # Extract the JSON array itself (strip any leading/trailing prose)
            m = re.search(r"\[.*\]", resp, re.DOTALL)
            if not m:
                print(f"[Warning] Couldn't find JSON array in chapter {idx+1}, saving raw…")
                continue

            try:
                data = json.loads(m.group(0))
            except json.JSONDecodeError as e:
                print(f"[Error] Invalid JSON format in chapter {idx+1}: {e}")
                continue
            
            json_str = m.group(0) if m else resp

            # Parse *only* the cleaned-up JSON
            try:
                data = json.loads(json_str)
                total_ent = len(data)
                # log each entry _with_ chapter number
                for eidx, entry in enumerate(data, start=1):
                    # Check for stop during entry processing
                    if check_stop():
                        print(f"❌ Glossary extraction stopped during entry processing for chapter {idx+1}")
                        return
                        
                    elapsed = time.time() - start
                    # Fixed the calculation to avoid division by zero
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
                
            #merge entries as before
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
