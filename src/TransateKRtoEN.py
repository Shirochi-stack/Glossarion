# -*- coding: utf-8 -*-
import json
import logging
import shutil
import threading
import queue
import os, sys, io, zipfile, time, re, mimetypes, subprocess, tiktoken
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
from collections import Counter
from unified_api_client import UnifiedClient, UnifiedClientError

# optional: turn on HTTP‐level debugging in the OpenAI client
logging.basicConfig(level=logging.DEBUG)

# Fix for PyInstaller - handle stdout reconfigure more carefully
try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='ignore')
except AttributeError:
    if sys.stdout is None:
        devnull = open(os.devnull, "wb")
        sys.stdout = io.TextIOWrapper(devnull, encoding='utf-8', errors='ignore')
    elif hasattr(sys.stdout, 'buffer'):
        try:
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='ignore')
        except:
            pass

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

def set_output_redirect(log_callback=None):
    """Redirect print statements to a callback function for GUI integration"""
    if log_callback:
        class CallbackWriter:
            def __init__(self, callback):
                self.callback = callback
                
            def write(self, text):
                if text.strip():
                    self.callback(text.strip())
                    
            def flush(self):
                pass
                
        sys.stdout = CallbackWriter(log_callback)

# Load or initialize history between runs
def load_history(payloads_dir):
    hist_path = os.path.join(payloads_dir, "translation_history.json")
    if os.path.exists(hist_path):
        with open(hist_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_history(history, payloads_dir):
    hist_path = os.path.join(payloads_dir, "translation_history.json")
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

def get_instructions(lang):
    """Get language-specific translation instructions"""
    if lang == "japanese":
        return (
            "- Retain Japanese suffixes like -san, -sama, -chan, -kun\n"
            "- Preserve onomatopoeia and speech tone\n"
            "- Use Romaji for untranslatable sounds\n"
            "- Preserve all HTML tags and image references"
        )
    elif lang == "chinese":
        return (
            "- Preserve Chinese tone, family terms, and idioms\n"
            "- Do not add un-present honorifics\n"
            "- Preserve slang and tone\n"
            "- Preserve all HTML tags and image references"
        )
    else:  # korean
        return (
            "- Retain Korean suffixes like -nim, -ssi, oppa, hyung\n"
            "- Preserve dialects, speech tone, and slang\n"
            "- Convert onomatopoeia to Romaji\n"
            "- Preserve all HTML tags and image references"
        )

def extract_epub_metadata(zf):
    """Extract metadata from EPUB file"""
    meta = {}
    for n in zf.namelist():
        if n.lower().endswith('.opf'):
            soup = BeautifulSoup(zf.read(n), 'xml')
            for t in ['title','creator','language']:
                e = soup.find(t)
                if e: meta[t] = e.get_text(strip=True)
            break
    return meta

def extract_chapters(zf):
    """Extract chapters from EPUB file"""
    chaps = []
    for name in zf.namelist():
        if name.lower().endswith(('.xhtml','.html')):
            raw = zf.read(name)
            soup = BeautifulSoup(raw, 'html.parser')
            title_tag = soup.find(['h1','h2','title'])
            title = title_tag.get_text(strip=True) if title_tag else name
            m = re.search(r'chapter[\W_]*(\d+)', name, re.IGNORECASE) or re.search(r'\b(\d+)\b', title)
            if not m: continue
            full_body_html = soup.body.decode_contents()
            chaps.append({
                "num": int(m.group(1)),
                "title": title,
                "body": full_body_html
            })
    return sorted(chaps, key=lambda x: x["num"])
    
def save_glossary(output_dir, chapters, instructions):
    """Generate and save glossary from chapters"""
    samples = []
    for c in chapters:
        samples.append(c["body"])
    names, suffixes = [], set()
    for txt in samples:
        for nm in re.findall(r"\b[A-Z][a-z]{2,20}\b", txt):
            names.append(nm)
        for s in re.findall(r"\b\w+-(?:nim|ssi)\b|\boppa\b|\bhyung\b", txt, re.I):
            suffixes.add(s)
    gloss = {
        "Characters": list(set(names)),
        "Honorifics/Suffixes": sorted(suffixes),
        "Instructions": instructions
    }
    with open(os.path.join(output_dir, "glossary.json"), 'w', encoding='utf-8') as f:
        json.dump(gloss, f, ensure_ascii=False, indent=2)

def send_with_interrupt(messages, client, temperature, max_tokens, stop_check_fn):
    """Send API request with interrupt capability"""
    result_queue = queue.Queue()
    
    def api_call():
        try:
            result = client.send(messages, temperature=temperature, max_tokens=max_tokens)
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
            result = result_queue.get(timeout=check_interval)
            if isinstance(result, Exception):
                raise result
            return result
        except queue.Empty:
            if stop_check_fn():
                raise UnifiedClientError("Translation stopped by user")
            elapsed += check_interval
    
    raise UnifiedClientError(f"API call timed out after {timeout} seconds")

def remove_header_artifacts(chapters):
    """Remove Gemini header artifacts from first chapter"""
    if not chapters:
        return []
        
    first = chapters[0]["body"]
    soup = BeautifulSoup(first, "html.parser")
    removed_tags = []

    # Simplified Gemini filler pattern
    gemini_intro_re = re.compile(
        r"^(okay|sure|understood|of course|got it|here.*?s)\b.*\b(translate|translation)\b",
        re.IGNORECASE
    )

    # Remove Gemini filler from second+ <h1>
    h1_tags = soup.find_all("h1")
    if len(h1_tags) > 1:
        for tag in h1_tags[1:]:
            text = tag.get_text(strip=True)
            if gemini_intro_re.search(text):
                removed_tags.append(f"<h1>: {text[:60]}...")
                tag.decompose()

    # Remove Gemini filler from <p> or <div>
    for tag in soup.find_all(["p", "div"]):
        text = tag.get_text(strip=True)
        if gemini_intro_re.search(text):
            removed_tags.append(f"<{tag.name}>: {text[:60]}...")
            tag.decompose()

    # Remove leaked JSON blocks
    text_preview = soup.get_text(strip=True)
    if text_preview.lstrip().startswith("{") and text_preview.rstrip().endswith("}"):
        try:
            json.loads(text_preview)
            soup.clear()
            removed_tags.append("- Removed leaked JSON block in header")
        except json.JSONDecodeError:
            pass

    chapters[0]["body"] = str(soup)
    return removed_tags

def parse_token_limit(env_value):
    """Parse token limit from environment variable"""
    if not env_value or env_value.strip() == "":
        return None, "unlimited"
    
    env_value = env_value.strip()
    if env_value.lower() == "unlimited":
        return None, "unlimited"
    
    if env_value.isdigit() and int(env_value) > 0:
        limit = int(env_value)
        return limit, str(limit)
    
    # Default fallback
    return 1000000, "1000000 (default)"

def build_system_prompt(user_prompt, glossary_path, instructions):
    """Build the system prompt with glossary"""
    if user_prompt:
        system = user_prompt
        if os.path.exists(glossary_path):
            with open(glossary_path, "r", encoding="utf-8") as gf:
                entries = json.load(gf)
            glossary_block = json.dumps(entries, ensure_ascii=False, indent=2)
            system += (
                "\n\nUse the following glossary entries exactly as given:\n"
                f"{glossary_block}"
            )
    elif os.path.exists(glossary_path):
        with open(glossary_path, "r", encoding="utf-8") as gf:
            entries = json.load(gf)
        glossary_block = json.dumps(entries, ensure_ascii=False, indent=2)
        system = (
            "You are a professional translator. " + instructions + "\n\n"
            "Use the following glossary entries exactly as given:\n"
            f"{glossary_block}"
        )
    else:
        system = f"You are a professional translator. {instructions}"
    
    return system

def main(log_callback=None, stop_callback=None):
    """Main translation function"""
    if log_callback:
        set_output_redirect(log_callback)
    
    # Set up stop checking
    def check_stop():
        if stop_callback and stop_callback():
            print("❌ Translation stopped by user request.")
            return True
        return is_stop_requested()
    
    # Parse all environment variables
    MODEL = os.getenv("MODEL", "gemini-1.5-flash")
    EPUB_PATH = os.getenv("EPUB_PATH", "default.epub")
    TRANSLATION_LANG = os.getenv("TRANSLATION_LANG", "korean").lower()
    CONTEXTUAL = os.getenv("CONTEXTUAL", "1") == "1"
    DELAY = int(os.getenv("SEND_INTERVAL_SECONDS", "2"))
    SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", "").strip()
    REMOVE_HEADER = os.getenv("REMOVE_HEADER", "0") == "1"
    TEMP = float(os.getenv("TRANSLATION_TEMPERATURE", "0.3"))
    HIST_LIMIT = int(os.getenv("TRANSLATION_HISTORY_LIMIT", "20"))
    MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "8192"))
    
    # Parse chapter range
    rng = os.getenv("CHAPTER_RANGE", "")
    if rng and re.match(r"^\d+\s*-\s*\d+$", rng):
        start, end = map(int, rng.split("-", 1))
    else:
        start, end = None, None
    
    # Get instructions for the language
    instructions = get_instructions(TRANSLATION_LANG)
    
    # Set up tokenizer
    try:
        enc = tiktoken.encoding_for_model(MODEL)
    except:
        enc = tiktoken.get_encoding("cl100k_base")
    
    # Get API key
    API_KEY = (os.getenv("API_KEY") or 
               os.getenv("OPENAI_API_KEY") or 
               os.getenv("OPENAI_OR_Gemini_API_KEY") or
               os.getenv("GEMINI_API_KEY"))

    if not API_KEY:
        print("❌ Error: Set API_KEY, OPENAI_API_KEY, or OPENAI_OR_Gemini_API_KEY in your environment.")
        return

    print(f"[DEBUG] Found API key: {API_KEY[:10]}...")
    print(f"[DEBUG] Using model = {MODEL}")
    print(f"[DEBUG] Max output tokens = {MAX_OUTPUT_TOKENS}")

    # Initialize client
    client = UnifiedClient(model=MODEL, api_key=API_KEY)
        
    # Set up paths
    epub_path = sys.argv[1] if len(sys.argv) > 1 else EPUB_PATH
    epub_base = os.path.splitext(os.path.basename(epub_path))[0]
    out = epub_base
    os.makedirs(out, exist_ok=True)
    print(f"[DEBUG] Created output folder → {out}")

    # Set output directory in environment
    os.environ["EPUB_OUTPUT_DIR"] = out
    payloads_dir = out
    
    # Purge old translation history on startup
    history_file = os.path.join(payloads_dir, "translation_history.json")
    if os.path.exists(history_file):
        os.remove(history_file)
        print(f"[DEBUG] Purged translation history → {history_file}")
        
    # Load or init progress file
    PROGRESS_FILE = os.path.join(payloads_dir, "translation_progress.json")
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r", encoding="utf-8") as pf:
            prog = json.load(pf)
    else:
        prog = {"completed": []}    

    def save_progress():
        with open(PROGRESS_FILE, "w", encoding="utf-8") as pf:
            json.dump(prog, pf, ensure_ascii=False, indent=2)

    # Check for stop before starting
    if check_stop():
        return

    # Extract EPUB contents
    with zipfile.ZipFile(epub_path, 'r') as zf:
        metadata = extract_epub_metadata(zf)
        chapters = extract_chapters(zf)

        # Remove header artifacts if enabled
        if REMOVE_HEADER and chapters:
            removed_tags = remove_header_artifacts(chapters)
            if removed_tags:
                removal_log_path = os.path.join(out, "removal.txt")
                with open(removal_log_path, "a", encoding="utf-8") as logf:
                    logf.write(f"{chapters[0]['title']} (Chapter {chapters[0]['num']})\n")
                    for entry in removed_tags:
                        logf.write(f"- {entry}\n")
                    logf.write("\n")

        # Extract images
        imgdir = os.path.join(out, "images")
        os.makedirs(imgdir, exist_ok=True)
        for n in zf.namelist():
            if n.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.svg')):
                with open(os.path.join(imgdir, os.path.basename(n)), 'wb') as f:
                    f.write(zf.read(n))

    # Check for stop after file processing
    if check_stop():
        return

    # Write metadata
    with open(os.path.join(out, "metadata.json"), 'w', encoding='utf-8') as mf:
        json.dump(metadata, mf, ensure_ascii=False, indent=2)
        
    # Handle glossary
    manual_gloss = os.getenv("MANUAL_GLOSSARY")
    if manual_gloss and os.path.isfile(manual_gloss):
        shutil.copy(manual_gloss, os.path.join(out, "glossary.json"))
    else:
        save_glossary(out, chapters, instructions)

    # Build system prompt
    glossary_path = os.path.join(out, "glossary.json")
    system = build_system_prompt(SYSTEM_PROMPT, glossary_path, instructions)
    base_msg = [{"role": "system", "content": system}]
    
    total_chapters = len(chapters)
    
    # Process each chapter
    for idx, c in enumerate(chapters):
        # Check for stop at the beginning of each chapter
        if check_stop():
            print(f"❌ Translation stopped at chapter {idx+1}")
            return
            
        chap_num = c["num"]

        # Apply chapter range filter
        if start is not None and not (start <= chap_num <= end):
            continue

        # Skip already completed chapters
        if idx in prog["completed"]:
            print(f"[SKIP] Chapter #{idx+1} (EPUB-num {chap_num}) already done, skipping.")
            continue

        print(f"🔄 Processing Chapter {idx+1}/{total_chapters}: {c['title']}")

        user_prompt = c["body"] 
        history = load_history(payloads_dir)
        
        # Build messages with context
        if CONTEXTUAL:
            trimmed = history[-HIST_LIMIT*2:]
        else:
            trimmed = []
        msgs = base_msg + trimmed + [{"role": "user", "content": user_prompt}]

        while True:
            # Check for stop before API call
            if check_stop():
                print(f"❌ Translation stopped during chapter {idx+1}")
                return
                
            try:
                # Parse token limit just before checking
                _tok_env = os.getenv("MAX_INPUT_TOKENS", "1000000").strip()
                max_tokens_limit, budget_str = parse_token_limit(_tok_env)
                
                # Calculate token usage
                total_tokens = sum(len(enc.encode(m["content"])) for m in msgs)
                print(f"[DEBUG] Prompt tokens = {total_tokens} / {budget_str}")
                
                # Check if over token limit
                if max_tokens_limit is not None and total_tokens > max_tokens_limit:
                    print(f"⚠️ Chapter {idx+1} exceeds token limit: {total_tokens} > {max_tokens_limit}")
                    print(f"⚠️ Skipping chapter {idx+1} due to token limit")
                    
                    # Mark as completed and save placeholder
                    prog["completed"].append(idx)
                    save_progress()
                    
                    safe_title = re.sub(r'\W+', '_', c['title'])[:40]
                    fname = f"response_{c['num']:03d}_{safe_title}.html"
                    with open(os.path.join(out, fname), 'w', encoding='utf-8') as f:
                        f.write(f"<h1>Chapter {c['num']}: {c['title']}</h1>\n")
                        f.write(f"<p><em>Chapter skipped: Exceeded token limit ({total_tokens} tokens > {max_tokens_limit} limit)</em></p>\n")
                    
                    break  # Exit retry loop
                
                # Send request with interrupt capability
                result, finish_reason = send_with_interrupt(
                    msgs, client, TEMP, MAX_OUTPUT_TOKENS, check_stop
                )
                
                if finish_reason == "length":
                    print(f"[WARN] Output was truncated!")
                    
                # Remove header artifacts if enabled
                if REMOVE_HEADER:
                    lines = result.splitlines(True)
                    intro_re = re.compile(
                        r'^(?:okay|sure|understood|of course|got it|here.*?s)[^\n]*\b(?:translate|translation)\b',
                        re.IGNORECASE
                    )
                    while lines and intro_re.match(lines[0]):
                        lines.pop(0)
                    result = "".join(lines)
                    
                # Update history
                history = load_history(payloads_dir)
                old_len = len(history)
                history = history[-HIST_LIMIT * 2:]
                history_trimmed = len(history) < old_len
                
                if history_trimmed:
                    print(f"[DBG] Trimmed translation history from {old_len} to {len(history)} entries.")

                history.append({"role": "user", "content": user_prompt})
                history.append({"role": "assistant", "content": result})
                save_history(history, payloads_dir)

                # Handle rolling summary if enabled
                if history_trimmed and os.getenv("USE_ROLLING_SUMMARY", "0") == "1":
                    if check_stop():
                        print(f"❌ Translation stopped during summary generation for chapter {idx+1}")
                        return
                        
                    # Generate summary
                    recent_entries = [h for h in history[-2:] if h["role"] == "assistant"]
                    summary_prompt = (
                        "Summarize the key events, tone, and terminology used in the following translation.\n"
                        "The summary will be used to maintain consistency in the next chapter.\n\n"
                        + "\n\n".join(e["content"] for e in recent_entries)
                    )

                    summary_msgs = [
                        {"role": "system", "content": "You are a summarizer."},
                        {"role": "user", "content": summary_prompt}
                    ]
                    
                    summary_resp, _ = send_with_interrupt(
                        summary_msgs, client, TEMP, MAX_OUTPUT_TOKENS, check_stop
                    )

                    # Save summary
                    summary_file = os.path.join(out, "rolling_summary.txt")
                    with open(summary_file, "w", encoding="utf-8") as sf:
                        sf.write(summary_resp.strip())

                    # Inject summary into base prompt
                    base_msg.insert(1, {
                        "role": os.getenv("SUMMARY_ROLE", "user"),
                        "content": (
                            "Here is a concise summary of the previous chapter. "
                            "Use this to ensure accurate tone, terminology, and character continuity:\n\n"
                            f"{summary_resp.strip()}"
                        )
                    })

                # Delay between API calls
                for i in range(DELAY):
                    if check_stop():
                        print(f"❌ Translation stopped during delay after chapter {idx+1}")
                        return
                    time.sleep(1)
                break

            except UnifiedClientError as e:
                error_msg = str(e)
                if "stopped by user" in error_msg:
                    print("❌ Translation stopped by user during API call")
                    return
                elif "timed out" in error_msg:
                    print(f"⚠️ {error_msg}, retrying...")
                    continue
                elif getattr(e, "http_status", None) == 429:
                    print("⚠️ Rate limited, sleeping 60s…")
                    for i in range(60):
                        if check_stop():
                            print("❌ Translation stopped during rate limit wait")
                            return
                        time.sleep(1)
                else:
                    raise

        # Check for stop before writing file
        if check_stop():
            print(f"❌ Translation stopped before saving chapter {idx+1}")
            return

        # Save translated chapter
        safe_title = re.sub(r'\W+', '_', c['title'])[:40]
        fname = f"response_{c['num']:03d}_{safe_title}.html"

        # Clean up code fences
        cleaned = re.sub(r"^```(?:html)?\s*", "", result, flags=re.MULTILINE)
        cleaned = re.sub(r"\s*```$", "", cleaned, flags=re.MULTILINE)

        # Write HTML file
        with open(os.path.join(out, fname), 'w', encoding='utf-8') as f:
            f.write(f"<h1>Chapter {c['num']}: {c['title']}</h1>\n" + cleaned)
        
        final_title = c['title'] or safe_title
        print(f"[Chapter {idx+1}/{total_chapters}] ✅ Saved Chapter {c['num']}: {final_title}")
        
        # Record completion
        prog["completed"].append(idx)
        save_progress()

    # Check for stop before building EPUB
    if check_stop():
        print("❌ Translation stopped before building EPUB")
        return

    # Build final EPUB
    print("📘 Building final EPUB…")
    try:
        from epub_converter import fallback_compile_epub
        fallback_compile_epub(out, log_callback=log_callback)
        print("✅ All done: your final EPUB is in", out)
    except Exception as e:
        print("❌ EPUB build failed:", e)

    # Signal completion to GUI
    print("TRANSLATION_COMPLETE_SIGNAL")

if __name__ == "__main__":
    main()
