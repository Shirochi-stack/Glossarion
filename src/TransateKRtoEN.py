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

# Import the new modules
from history_manager import HistoryManager
from chapter_splitter import ChapterSplitter

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

# Remove old load_history and save_history functions since we'll use HistoryManager

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
    """Extract chapters from EPUB file with multi-language support"""
    chaps = []
      # Get all HTML files and sort them to maintain order
    html_files = sorted([name for name in zf.namelist() 
                        if name.lower().endswith(('.xhtml', '.html'))])
    
    print(f"[DEBUG] Processing {len(html_files)} HTML files from EPUB")
    
    # Process EVERY HTML file
    for idx, name in enumerate(html_files):
        try:
            raw = zf.read(name)
            soup = BeautifulSoup(raw, 'html.parser')
            
            # Get body content
            if soup.body:
                full_body_html = soup.body.decode_contents()
                body_text = soup.body.get_text(strip=True)
            else:
                full_body_html = str(soup)
                body_text = soup.get_text(strip=True)
            
            # Skip only if completely empty
            if not body_text.strip():
                print(f"[DEBUG] Skipping empty file: {name}")
                continue
            
            # Always use index as chapter number to avoid skipping
            # This ensures we don't miss any content
            chapter_num = idx + 1
            
            # Try to get a better title
            title = None
            
            # Method 1: Look for headers
            for header_tag in ['h1', 'h2', 'h3', 'title']:
                header = soup.find(header_tag)
                if header:
                    title = header.get_text(strip=True)
                    if title:
                        break
            
            # Method 2: Use filename
            if not title:
                # Remove extension and path
                base_name = os.path.splitext(os.path.basename(name))[0]
                title = base_name.replace('_', ' ').replace('-', ' ')
            
            # Method 3: Fallback
            if not title or title.lower() in ['text', 'document', 'html']:
                title = f"Chapter {chapter_num}"
            
            # Create chapter entry
            chaps.append({
                "num": chapter_num,
                "title": title[:100],  # Limit title length
                "body": full_body_html,
                "filename": name
            })
            
        except Exception as e:
            print(f"[ERROR] Failed to process {name}: {e}")
            # Even on error, add a placeholder so we don't skip
            chaps.append({
                "num": idx + 1,
                "title": f"Chapter {idx + 1} (Error)",
                "body": f"<p>Error loading chapter from {name}</p>",
                "filename": name
            })
    
    print(f"[DEBUG] Extracted {len(chaps)} chapters")
    return chaps
    # Multi-language chapter patterns
    chapter_patterns = [
        # English
        (r'chapter[\W_]*(\d+)', re.IGNORECASE),
        (r'ch[\W_]*(\d+)', re.IGNORECASE),
        (r'part[\W_]*(\d+)', re.IGNORECASE),
        
        # Chinese
        (r'第\s*(\d+)\s*[章节話话回]', 0),  # 第1章, 第1节, 第1話, 第1话, 第1回
        (r'第\s*([一二三四五六七八九十百千]+)\s*[章节話话回]', 0),  # 第一章, etc
        (r'(\d+)[章节話话回]', 0),  # 1章, 1节, etc
        
        # Japanese
        (r'第\s*(\d+)\s*話', 0),  # 第1話
        (r'第\s*(\d+)\s*章', 0),  # 第1章
        (r'その\s*(\d+)', 0),     # その1
        (r'話\s*(\d+)', 0),       # 話1
        (r'第\s*(\d+)\s*部', 0),  # 第1部
        
        # Korean
        (r'제\s*(\d+)\s*[장화권부]', 0),  # 제1장, 제1화, 제1권, 제1부
        (r'(\d+)\s*[장화권부]', 0),       # 1장, 1화, etc
        
        # Generic number patterns (as fallback)
        (r'^(\d+)\.?\s*$', re.MULTILINE),  # Just numbers at line start
        (r'^\s*(\d+)\s*[-–—]\s*', re.MULTILINE),  # "1 - Title"
        (r'_(\d+)\.x?html?$', re.IGNORECASE),  # filename_1.html
        (r'[\W_](\d{2,4})[\W_]', 0),  # Any 2-4 digit number surrounded by non-word chars
    ]
    
    # Chinese number conversion
    chinese_nums = {
        '一': 1, '二': 2, '三': 3, '四': 4, '五': 5,
        '六': 6, '七': 7, '八': 8, '九': 9, '十': 10,
        '十一': 11, '十二': 12, '十三': 13, '十四': 14, '十五': 15,
        '十六': 16, '十七': 17, '十八': 18, '十九': 19, '二十': 20,
        '三十': 30, '四十': 40, '五十': 50, '六十': 60,
        '七十': 70, '八十': 80, '九十': 90, '百': 100,
        '一百': 100, '两百': 200, '三百': 300, '四百': 400,
        '五百': 500, '六百': 600, '七百': 700, '八百': 800,
        '九百': 900, '千': 1000
    }
    
    def convert_chinese_number(cn_num):
        """Convert Chinese number to integer"""
        if cn_num in chinese_nums:
            return chinese_nums[cn_num]
        
        # Handle compound numbers like 二十一 (21)
        if '十' in cn_num:
            parts = cn_num.split('十')
            if len(parts) == 2:
                tens = chinese_nums.get(parts[0], 1) if parts[0] else 1
                ones = chinese_nums.get(parts[1], 0) if parts[1] else 0
                return tens * 10 + ones
        
        # Handle hundreds
        if '百' in cn_num:
            parts = cn_num.split('百')
            hundreds = chinese_nums.get(parts[0], 1) if parts[0] else 1
            remainder = parts[1] if len(parts) > 1 and parts[1] else ''
            if remainder:
                return hundreds * 100 + convert_chinese_number(remainder)
            return hundreds * 100
            
        return None
    
    # Track found chapters to avoid duplicates
    found_chapters = {}
    
    # First, try to detect chapter pattern from filenames
    all_files = [n for n in zf.namelist() if n.lower().endswith(('.xhtml', '.html'))]
    detected_pattern = None
    
    # Check if files follow a numeric pattern
    for pattern, flags in chapter_patterns[:10]:  # Check first 10 patterns
        matches = 0
        for fname in all_files[:10]:  # Sample first 10 files
            if re.search(pattern, fname, flags):
                matches += 1
        if matches > len(all_files[:10]) * 0.5:  # If >50% match
            detected_pattern = (pattern, flags)
            print(f"[DEBUG] Detected chapter pattern in filenames: {pattern}")
            break
    
    # Process each file
    for idx, name in enumerate(all_files):
        if not name.lower().endswith(('.xhtml', '.html')):
            continue
            
        try:
            raw = zf.read(name)
            soup = BeautifulSoup(raw, 'html.parser')
            
            # Skip if no body content
            if not soup.body:
                continue
                
            body_text = soup.body.get_text(strip=True)
            if len(body_text) < 100:  # Skip very short files (likely TOC, etc)
                continue
            
            # Try to extract chapter number
            chapter_num = None
            chapter_title = None
            
            # Method 1: Check filename first
            for pattern, flags in chapter_patterns:
                m = re.search(pattern, name, flags)
                if m:
                    try:
                        # Try to convert to number
                        num_str = m.group(1)
                        if num_str.isdigit():
                            chapter_num = int(num_str)
                        else:
                            # Try Chinese number conversion
                            chapter_num = convert_chinese_number(num_str)
                        
                        if chapter_num:
                            break
                    except:
                        continue
            
            # Method 2: Check content headers
            if not chapter_num:
                # Look for chapter markers in headers
                for header in soup.find_all(['h1', 'h2', 'h3', 'title']):
                    header_text = header.get_text(strip=True)
                    if not header_text:
                        continue
                        
                    for pattern, flags in chapter_patterns:
                        m = re.search(pattern, header_text, flags)
                        if m:
                            try:
                                num_str = m.group(1)
                                if num_str.isdigit():
                                    chapter_num = int(num_str)
                                else:
                                    chapter_num = convert_chinese_number(num_str)
                                
                                if chapter_num:
                                    chapter_title = header_text
                                    break
                            except:
                                continue
                    
                    if chapter_num:
                        break
            
            # Method 3: Check first few paragraphs for chapter markers
            if not chapter_num:
                first_texts = []
                for elem in soup.find_all(['p', 'div'])[:5]:  # Check first 5 elements
                    text = elem.get_text(strip=True)
                    if text:
                        first_texts.append(text)
                
                for text in first_texts:
                    for pattern, flags in chapter_patterns:
                        m = re.search(pattern, text, flags)
                        if m:
                            try:
                                num_str = m.group(1)
                                if num_str.isdigit():
                                    chapter_num = int(num_str)
                                else:
                                    chapter_num = convert_chinese_number(num_str)
                                
                                if chapter_num:
                                    break
                            except:
                                continue
                    
                    if chapter_num:
                        break
            
            # Method 4: If no chapter number found, use file index
            if not chapter_num and detected_pattern:
                # If we detected a pattern but this file doesn't match,
                # it might be a prologue/epilogue
                continue
            elif not chapter_num:
                # Use file index as chapter number (1-based)
                chapter_num = idx + 1
                print(f"[DEBUG] No chapter marker found in {name}, using index {chapter_num}")
            
            # Skip if we already have this chapter number
            if chapter_num in found_chapters:
                print(f"[DEBUG] Duplicate chapter {chapter_num} found in {name}, skipping")
                continue
            
            # Get title if not already set
            if not chapter_title:
                # Try to get title from headers
                title_elem = soup.find(['h1', 'h2', 'h3', 'title'])
                if title_elem:
                    chapter_title = title_elem.get_text(strip=True)
                else:
                    chapter_title = f"Chapter {chapter_num}"
            
            # Get full body HTML
            full_body_html = soup.body.decode_contents() if soup.body else str(soup)
            
            chapter_info = {
                "num": chapter_num,
                "title": chapter_title,
                "body": full_body_html,
                "filename": name
            }
            
            chaps.append(chapter_info)
            found_chapters[chapter_num] = True
            
        except Exception as e:
            print(f"[WARNING] Error processing {name}: {e}")
            continue
    
    # Sort by chapter number
    chaps.sort(key=lambda x: x["num"])
    
    # Debug output
    print(f"[DEBUG] Found {len(chaps)} chapters out of {len(all_files)} HTML files")
    if chaps:
        print(f"[DEBUG] Chapter range: {chaps[0]['num']} to {chaps[-1]['num']}")
        print(f"[DEBUG] First few chapters: {[c['title'][:30] + '...' for c in chaps[:5]]}")
    
    return chaps
    
def save_glossary(output_dir, chapters, instructions, language="korean"):
    """Generate and save glossary from chapters"""
    samples = []
    for c in chapters:
        samples.append(c["body"])
    
    names = []
    suffixes = set()
    
    for txt in samples:
        # Extract names (works for all languages)
        for nm in re.findall(r"\b[A-Z][a-z]{2,20}\b", txt):
            names.append(nm)
        
        # Language-specific suffix/honorific patterns
        if language == "korean":
            # Korean suffixes
            for s in re.findall(r"\b\w+[-~]?(?:nim|ssi|ah|ya|ie|hyung|noona|unnie|oppa|sunbae|hoobae|gun|yang)\b", txt, re.I):
                suffixes.add(s)
        
        elif language == "japanese":
            # Japanese honorifics and suffixes
            for s in re.findall(r"\b\w+[-~]?(?:san|sama|chan|kun|senpai|kouhai|sensei|dono|tan|chin|bo|rin|pyon)\b", txt, re.I):
                suffixes.add(s)
            # Also catch standalone honorifics
            for s in re.findall(r"\b(?:Onii|Onee|Oji|Oba|Nii|Nee)[-~]?(?:san|sama|chan|kun)?\b", txt, re.I):
                suffixes.add(s)
        
        elif language == "chinese":
            # Chinese titles and honorifics
            # Common suffixes/titles in pinyin or English
            for s in re.findall(r"\b\w+[-~]?(?:ge|gege|jie|jiejie|di|didi|mei|meimei|xiong|da|xiao|lao|shao)\b", txt, re.I):
                suffixes.add(s)
            # Titles
            for s in re.findall(r"\b(?:Shizun|Shifu|Daozhang|Gongzi|Guniang|Xiaojie|Furen|Niangniang)\b", txt, re.I):
                suffixes.add(s)
            # Family terms
            for s in re.findall(r"\b(?:A-|Ah-)?(?:Niang|Die|Ba|Ma|Ye|Nai|Gong|Po)\b", txt, re.I):
                suffixes.add(s)
    
    # Remove duplicates and sort
    unique_names = list(set(names))
    sorted_suffixes = sorted(suffixes)
    
    # Build glossary with language-specific labels
    gloss = {
        "Characters": unique_names,
        "Instructions": instructions
    }
    
    # Add language-specific sections
    if language == "korean":
        gloss["Korean_Honorifics_Suffixes"] = sorted_suffixes
    elif language == "japanese":
        gloss["Japanese_Honorifics_Suffixes"] = sorted_suffixes
    elif language == "chinese":
        gloss["Chinese_Titles_Terms"] = sorted_suffixes
    
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
    # Check if system prompt is disabled
    if os.getenv("DISABLE_SYSTEM_PROMPT", "0") == "1":
        # Return user prompt as-is when system prompt is disabled
        return user_prompt if user_prompt else ""
    
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
    
    # Initialize HistoryManager and ChapterSplitter
    history_manager = HistoryManager(payloads_dir)
    chapter_splitter = ChapterSplitter(model_name=MODEL)
    
    # Purge old translation history on startup
    history_file = os.path.join(payloads_dir, "translation_history.json")
    if os.path.exists(history_file):
        os.remove(history_file)
        print(f"[DEBUG] Purged translation history → {history_file}")
        
    # Load or init progress file - UPDATED to include chunk tracking
    PROGRESS_FILE = os.path.join(payloads_dir, "translation_progress.json")
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r", encoding="utf-8") as pf:
            prog = json.load(pf)
        # Ensure chapter_chunks exists in old progress files
        if "chapter_chunks" not in prog:
            prog["chapter_chunks"] = {}
    else:
        prog = {"completed": [], "chapter_chunks": {}}

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
    disable_auto_glossary = os.getenv("DISABLE_AUTO_GLOSSARY", "0") == "1"

    if manual_gloss and os.path.isfile(manual_gloss):
        # Use manual glossary if provided
        shutil.copy(manual_gloss, os.path.join(out, "glossary.json"))
        print("📑 Using manual glossary")
    elif not disable_auto_glossary:
        # Generate automatic glossary only if not disabled
        save_glossary(out, chapters, instructions, TRANSLATION_LANG)  # Pass language
        print("📑 Generated automatic glossary")
    else:
        # Don't create any glossary file when disabled
        print("📑 Automatic glossary disabled - no glossary will be used")

    # Build system prompt
    glossary_path = os.path.join(out, "glossary.json")
    system = build_system_prompt(SYSTEM_PROMPT, glossary_path, instructions)
    base_msg = [{"role": "system", "content": system}]
    
    total_chapters = len(chapters)
    
    # First pass: Count total chunks needed
    print("📊 Calculating total chunks needed...")
    total_chunks_needed = 0
    chunks_per_chapter = {}
    
    for idx, c in enumerate(chapters):
        chap_num = c["num"]
        
        # Apply chapter range filter
        if start is not None and not (start <= chap_num <= end):
            continue
            
        # Skip already completed chapters
        if idx in prog["completed"]:
            chunks_per_chapter[idx] = 0
            continue
        
        # Parse token limit for counting
        _tok_env = os.getenv("MAX_INPUT_TOKENS", "1000000").strip()
        max_tokens_limit, _ = parse_token_limit(_tok_env)
        
        # Calculate available tokens
        system_tokens = chapter_splitter.count_tokens(system)
        history_tokens = HIST_LIMIT * 2 * 1000
        safety_margin = 1000
        
        if max_tokens_limit is not None:
            available_tokens = max_tokens_limit - system_tokens - history_tokens - safety_margin
            chunks = chapter_splitter.split_chapter(c["body"], available_tokens)
        else:
            chunks = [(c["body"], 1, 1)]
        
        # Count chunks needed for this chapter
        chapter_key = str(idx)
        if chapter_key in prog.get("chapter_chunks", {}):
            # Count only remaining chunks
            completed_chunks = len(prog["chapter_chunks"][chapter_key].get("completed", []))
            chunks_needed = len(chunks) - completed_chunks
            chunks_per_chapter[idx] = max(0, chunks_needed)
        else:
            chunks_per_chapter[idx] = len(chunks)
        
        total_chunks_needed += chunks_per_chapter[idx]
    
    print(f"📊 Total chunks to translate: {total_chunks_needed}")
    
    # Print chapter breakdown if there are multi-chunk chapters
    multi_chunk_chapters = [(idx, count) for idx, count in chunks_per_chapter.items() if count > 1]
    if multi_chunk_chapters:
        print("📄 Chapters requiring multiple chunks:")
        for idx, chunk_count in multi_chunk_chapters:
            chap = chapters[idx]
            print(f"   • Chapter {idx+1} ({chap['title'][:30]}...): {chunk_count} chunks")
    
    # Track timing for ETA calculation
    translation_start_time = time.time()
    chunks_completed = 0
    
    # Process each chapter with chunk counting
    current_chunk_number = 0
    
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

        print(f"\n🔄 Processing Chapter {idx+1}/{total_chapters}: {c['title']}")

        # Parse token limit
        _tok_env = os.getenv("MAX_INPUT_TOKENS", "1000000").strip()
        max_tokens_limit, budget_str = parse_token_limit(_tok_env)
        
        # Calculate available tokens for content
        system_tokens = chapter_splitter.count_tokens(system)
        
        # Estimate tokens for history (rough approximation)
        history_tokens = HIST_LIMIT * 2 * 1000  # Assume ~1000 tokens per history entry
        
        # Safety margin
        safety_margin = 1000
        
        # Determine if we need to split the chapter
        if max_tokens_limit is not None:
            available_tokens = max_tokens_limit - system_tokens - history_tokens - safety_margin
            chunks = chapter_splitter.split_chapter(c["body"], available_tokens)
        else:
            # No limit, process as single chunk
            chunks = [(c["body"], 1, 1)]
        
        print(f"📄 Chapter will be processed in {len(chunks)} chunk(s)")
        
        # Show token information if split was needed
        if len(chunks) > 1:
            chapter_tokens = chapter_splitter.count_tokens(c["body"])
            print(f"   ℹ️ Chapter size: {chapter_tokens:,} tokens (limit: {available_tokens:,} tokens per chunk)")
        else:
            chapter_tokens = chapter_splitter.count_tokens(c["body"])
            if max_tokens_limit is not None:
                print(f"   ℹ️ Chapter size: {chapter_tokens:,} tokens (within limit of {available_tokens:,} tokens)")
        
        # Track translated chunks for this chapter
        chapter_key = str(idx)
        if chapter_key not in prog["chapter_chunks"]:
            prog["chapter_chunks"][chapter_key] = {
                "total": len(chunks),
                "completed": [],
                "chunks": {}
            }
        
        # Update total chunks if different (in case of re-run with different settings)
        prog["chapter_chunks"][chapter_key]["total"] = len(chunks)
        
        translated_chunks = []
        
        # Process each chunk
        for chunk_html, chunk_idx, total_chunks in chunks:
            # Check if this chunk was already translated
            if chunk_idx in prog["chapter_chunks"][chapter_key]["completed"]:
                # Load previously translated chunk
                saved_chunk = prog["chapter_chunks"][chapter_key]["chunks"].get(str(chunk_idx))
                if saved_chunk:
                    translated_chunks.append((saved_chunk, chunk_idx, total_chunks))
                    print(f"  [SKIP] Chunk {chunk_idx}/{total_chunks} already translated")
                    continue
            
            if check_stop():
                print(f"❌ Translation stopped during chapter {idx+1}, chunk {chunk_idx}")
                return
            
            current_chunk_number += 1
            
            # Calculate progress and ETA
            progress_percent = (current_chunk_number / total_chunks_needed) * 100
            
            # Calculate ETA if we have completed at least one chunk
            if chunks_completed > 0:
                elapsed_time = time.time() - translation_start_time
                avg_time_per_chunk = elapsed_time / chunks_completed
                remaining_chunks = total_chunks_needed - current_chunk_number + 1
                eta_seconds = remaining_chunks * avg_time_per_chunk
                
                # Format ETA
                eta_hours = int(eta_seconds // 3600)
                eta_minutes = int((eta_seconds % 3600) // 60)
                eta_str = f"{eta_hours}h {eta_minutes}m" if eta_hours > 0 else f"{eta_minutes}m"
            else:
                eta_str = "calculating..."
            
            if total_chunks > 1:
                print(f"  🔄 Translating chunk {chunk_idx}/{total_chunks} (Overall: {current_chunk_number}/{total_chunks_needed} - {progress_percent:.1f}% - ETA: {eta_str})")
            else:
                print(f"  🔄 Translating chapter (Overall: {current_chunk_number}/{total_chunks_needed} - {progress_percent:.1f}% - ETA: {eta_str})")
            
            # Add chunk context to prompt if multi-chunk
            if total_chunks > 1:
                user_prompt = f"[PART {chunk_idx}/{total_chunks}]\n{chunk_html}"
            else:
                user_prompt = chunk_html
            
            # Load history using thread-safe manager
            history = history_manager.load_history()
            
            # Build messages with context
            if CONTEXTUAL:
                trimmed = history[-HIST_LIMIT*2:]
            else:
                trimmed = []
                
            # Build messages - handle case where base_msg might be empty
            if base_msg:
                msgs = base_msg + trimmed + [{"role": "user", "content": user_prompt}]
            else:
                # No system message, start with history or user message
                if trimmed:
                    msgs = trimmed + [{"role": "user", "content": user_prompt}]
                else:
                    msgs = [{"role": "user", "content": user_prompt}]

            while True:
                # Check for stop before API call
                if check_stop():
                    print(f"❌ Translation stopped during chapter {idx+1}")
                    return
                    
                try:
                    # Calculate actual token usage
                    total_tokens = sum(chapter_splitter.count_tokens(m["content"]) for m in msgs)
                    print(f"    [DEBUG] Chunk {chunk_idx}/{total_chunks} tokens = {total_tokens:,} / {budget_str}")
                    
                    # Send request with interrupt capability
                    result, finish_reason = send_with_interrupt(
                        msgs, client, TEMP, MAX_OUTPUT_TOKENS, check_stop
                    )
                    
                    if finish_reason == "length":
                        print(f"    [WARN] Output was truncated!")
                        
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
                    
                    # Remove chunk markers if present
                    result = re.sub(r'\[PART \d+/\d+\]\s*', '', result, flags=re.IGNORECASE)
                    
                    # Save chunk result
                    translated_chunks.append((result, chunk_idx, total_chunks))
                    
                    # Update progress for this chunk
                    prog["chapter_chunks"][chapter_key]["completed"].append(chunk_idx)
                    prog["chapter_chunks"][chapter_key]["chunks"][str(chunk_idx)] = result
                    save_progress()
                    
                    # Increment completed chunks counter
                    chunks_completed += 1
                        
                    # Update history using thread-safe manager
                    history = history_manager.load_history()
                    old_len = len(history)
                    history = history[-HIST_LIMIT * 2:]
                    history_trimmed = len(history) < old_len
                    
                    if history_trimmed:
                        print(f"    [DBG] Trimmed translation history from {old_len} to {len(history)} entries.")

                    history.append({"role": "user", "content": user_prompt})
                    history.append({"role": "assistant", "content": result})
                    history_manager.save_history(history)

                    # Handle rolling summary if enabled and history was trimmed
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

                    # Delay between chunks/API calls
                    if chunk_idx < total_chunks:
                        for i in range(DELAY):
                            if check_stop():
                                print("❌ Translation stopped during delay")
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

        # Check for stop before merging and saving
        if check_stop():
            print(f"❌ Translation stopped before saving chapter {idx+1}")
            return

        # Merge all chunks back together
        if len(translated_chunks) > 1:
            print(f"  📎 Merging {len(translated_chunks)} chunks...")
            # Sort by chunk index to ensure correct order
            translated_chunks.sort(key=lambda x: x[1])
            # Merge the HTML content
            merged_result = chapter_splitter.merge_translated_chunks(translated_chunks)
        else:
            merged_result = translated_chunks[0][0] if translated_chunks else ""

        # Save translated chapter
        safe_title = re.sub(r'\W+', '_', c['title'])[:40]
        fname = f"response_{c['num']:03d}_{safe_title}.html"

        # Clean up code fences
        cleaned = re.sub(r"^```(?:html)?\s*", "", merged_result, flags=re.MULTILINE)
        cleaned = re.sub(r"\s*```$", "", cleaned, flags=re.MULTILINE)

        # Write HTML file without adding extra header
        # The translator should preserve the original chapter structure
        with open(os.path.join(out, fname), 'w', encoding='utf-8') as f:
            f.write(cleaned)
        
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
        
        # Print final statistics
        total_time = time.time() - translation_start_time
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = int(total_time % 60)
        
        print(f"\n📊 Translation Statistics:")
        print(f"   • Total chunks processed: {chunks_completed}")
        print(f"   • Total time: {hours}h {minutes}m {seconds}s")
        if chunks_completed > 0:
            avg_time = total_time / chunks_completed
            print(f"   • Average time per chunk: {avg_time:.1f} seconds")
            
    except Exception as e:
        print("❌ EPUB build failed:", e)

    # Signal completion to GUI
    print("TRANSLATION_COMPLETE_SIGNAL")

if __name__ == "__main__":
    main()
