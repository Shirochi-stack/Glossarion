# -*- coding: utf-8 -*-
import json
import logging
import shutil
# optional: turn on HTTP‐level debugging in the OpenAI client
logging.basicConfig(level=logging.DEBUG)

import os, sys, io, zipfile, time, json, re, textwrap, mimetypes, subprocess, tiktoken
import ebooklib
import re
MODEL            = os.getenv("MODEL", "gemini-1.5-flash")
_tok_env = os.getenv("MAX_INPUT_TOKENS", "").strip()
REMOVE_HEADER = os.getenv("REMOVE_HEADER", "0") == "1"
if _tok_env.isdigit():
    MAX_INPUT_TOKENS = int(_tok_env)
    _budget_str = str(MAX_INPUT_TOKENS)
else:
    MAX_INPUT_TOKENS = None       # signal “unlimited”
    _budget_str = "∞"             # or "unlimited"
try:
    # for any OpenAI model or if tiktoken knows it
    enc = tiktoken.encoding_for_model(MODEL)
except KeyError:
    # fallback for Gemini (or any unknown name)
    enc = tiktoken.get_encoding("cl100k_base")

print(f"[DEBUG] Using model = {MODEL}")
print(f"[DEBUG] Input token budget = {MAX_INPUT_TOKENS}")
# needed for ITEM_DOCUMENT
from ebooklib import epub
from bs4 import BeautifulSoup
from collections import Counter
from unified_api_client import UnifiedClient
from unified_api_client import UnifiedClientError

# Load or initialize history between runs
def load_history():
    hist_path = os.path.join(payloads_dir, "translation_history.json")
    if os.path.exists(hist_path):
        with open(hist_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_history(history):
    hist_path = os.path.join(payloads_dir, "translation_history.json")
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

# Read limits from env
TEMP = float(os.getenv("TRANSLATION_TEMPERATURE", 0.3))
HIST_LIMIT = int(os.getenv("TRANSLATION_HISTORY_LIMIT", 20))

# base output folder (from env or “.”)
output_dir = os.getenv("EPUB_OUTPUT_DIR", ".")

# make sure the Payloads sub-folder exists
payloads_dir = os.path.join(output_dir, "Payloads")
os.makedirs(payloads_dir, exist_ok=True)
log_path = os.path.join(payloads_dir, "openai_http_debug.log")
file_handler = logging.FileHandler(log_path, encoding="utf-8")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter("%(message)s"))


# attach it to each library logger and prevent console propagation
for logger_name in ("openai._base_client", "httpcore", "httpx"):
    lib_logger = logging.getLogger(logger_name)
    lib_logger.setLevel(logging.DEBUG)
    lib_logger.addHandler(file_handler)
    lib_logger.propagate = False

# ensure UTF-8 flush
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
print = lambda *a, **k: __builtins__.print(*a, **{**k, "flush": True})

# ---------- CONFIG ----------
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    print("❌ Error: Set OPENAI_OR_Gemini_API_KEY in your environment.")
    sys.exit(1)
client = UnifiedClient(model=MODEL, api_key=API_KEY)
EPUB_PATH       = os.getenv("EPUB_PATH", "default.epub")
MODEL           = os.getenv("MODEL", "gpt-4.1-nano")
TRANSLATION_LANG= os.getenv("TRANSLATION_LANG", "korean").lower()
CONTEXTUAL      = os.getenv("CONTEXTUAL", "1") == "1"
DELAY           = int(os.getenv("SEND_INTERVAL_SECONDS", "2"))
SYSTEM_PROMPT   = os.getenv("SYSTEM_PROMPT", "").strip()
rng = os.getenv("CHAPTER_RANGE", "")
if rng and re.match(r"^\d+\s*-\s*\d+$", rng):
    start, end = map(int, rng.split("-", 1))
else:
    start, end = None, None


# ---------- INSTRUCTIONS ----------
if TRANSLATION_LANG == "japanese":
    INSTR = (
        "- Retain Japanese suffixes like -san, -sama, -chan, -kun\n"
        "- Preserve onomatopoeia and speech tone\n"
        "- Use Romaji for untranslatable sounds\n"
        "- Preserve all HTML tags and image references"
    )
elif TRANSLATION_LANG == "chinese":
    INSTR = (
        "- Preserve Chinese tone, family terms, and idioms\n"
        "- Do not add un-present honorifics\n"
        "- Preserve slang and tone\n"
        "- Preserve all HTML tags and image references"
    )
else:
    INSTR = (
        "- Retain Korean suffixes like -nim, -ssi, oppa, hyung\n"
        "- Preserve dialects, speech tone, and slang\n"
        "- Convert onomatopoeia to Romaji\n"
        "- Preserve all HTML tags and image references"
    )

# ---------- HELPERS ----------
def extract_epub_metadata(zf):
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
    

def save_glossary(output_dir, chapters):
    samples = []
    for c in chapters:
        samples.append(c["body"])
    names, suffixes, titles = [], set(), set()
    for txt in samples:
        for nm in re.findall(r"\b[A-Z][a-z]{2,20}\b", txt):
            names.append(nm)
        for s in re.findall(r"\b\w+-(?:nim|ssi)\b|\boppa\b|\bhyung\b", txt, re.I):
            suffixes.add(s)
    gloss = {
        "Characters": list(set(names)),
        "Honorifics/Suffixes": sorted(suffixes),
        "Instructions": INSTR
    }
    with open(os.path.join(output_dir, "glossary.json"), 'w', encoding='utf-8') as f:
        json.dump(gloss, f, ensure_ascii=False, indent=2)

def send(messages, max_tokens=None):
    if max_tokens is None:
        max_tokens = int(os.getenv("MAX_OUTPUT_TOKENS", "8192"))
    logging.debug(f"[DEBUG] Sending request with max_tokens={max_tokens}")
    return client.send(messages, temperature=TEMP, max_tokens=max_tokens)
    # Build the exact payload we're about to send
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": TEMP
    }

    # Determine where to save the payload log
    output_dir = os.getenv("EPUB_OUTPUT_DIR", ".")
    os.makedirs(output_dir, exist_ok=True)
    # after
    path = os.path.join(payloads_dir, "chat_payload.json")
    with open(path, "a", encoding="utf-8") as f:
        f.write("=== DEBUG: ChatGPT payload ===\n")
        f.write(json.dumps(payload, ensure_ascii=False, indent=2))
        f.write("\n=== END DEBUG ===\n\n")

    # ——— ALSO print it to stdout so the GUI will show it ———
    #print("=== DEBUG: ChatGPT payload ===")
    #print(json.dumps(payload, ensure_ascii=False, indent=2))
    #print("=== END DEBUG ===\n")

    # Actually send the request
    if max_tokens is None:
    # Read from env, fallback to 8192 if not set or invalid
        try:
            max_tokens = int(os.getenv("MAX_OUTPUT_TOKENS", "8192"))
        except ValueError:
            max_tokens = 8192
    logging.debug(f"[DEBUG] Sending request with max_tokens={max_tokens}")
    
    return client.send(messages, temperature=TEMP, max_tokens=max_tokens)
    print(f"[DEBUG] Sending request with max_tokens={max_tokens}")
    # alias for rolling‐summary logic
send_to_model = lambda msgs: send(msgs)[0]



# ---------- MAIN ----------
def main():
    epub_path = sys.argv[1]
    epub_base = os.path.splitext(os.path.basename(EPUB_PATH))[0]
    base_out  = epub_base

    # ─── always use a folder named exactly after the EPUB ───
    out = base_out
    os.makedirs(out, exist_ok=True)
    print(f"[DEBUG] Created output folder → {out}")

    # now we can set the env var correctly
    os.environ["EPUB_OUTPUT_DIR"] = out
        
    # ─── override payloads_dir so history lives inside `out` ───
    global payloads_dir
    payloads_dir = out
    
    # ─── PURGE old translation history on startup ───
    history_file = os.path.join(payloads_dir, "translation_history.json")
    if os.path.exists(history_file):
        os.remove(history_file)
        print(f"[DEBUG] Purged translation history → {history_file}")
        
    # ─── load or init our own per-EPUB progress file ───
    PROGRESS_FILE = os.path.join(payloads_dir, "translation_progress.json")
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r", encoding="utf-8") as pf:
            prog = json.load(pf)
    else:
        prog = {"completed": []}    

    # 1) unpack and collect
    with zipfile.ZipFile(EPUB_PATH, 'r') as zf:
        metadata = extract_epub_metadata(zf)
        chapters = extract_chapters(zf)

        if REMOVE_HEADER and chapters:
            first = chapters[0]["body"]
            soup = BeautifulSoup(first, "html.parser")

            removed_tags = []

            # Gemini filler pattern
            gemini_intro_re = re.compile(
                r"^(okay|sure|understood|of course|got it|here(?:'|’)s)\b.*\b(translate|translation)\b",
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

            # Remove Gemini filler from <p> or <div> if present
            for tag in soup.find_all(["p", "div"]):
                text = tag.get_text(strip=True)
                if gemini_intro_re.search(text):
                    removed_tags.append(f"<{tag.name}>: {text[:60]}...")
                    tag.decompose()

            # Preserve JSON block removal as-is
            text_preview = soup.get_text(strip=True)
            if text_preview.lstrip().startswith("{") and text_preview.rstrip().endswith("}"):
                try:
                    json.loads(text_preview)
                    soup.clear()
                    removed_tags.append("- Removed leaked JSON block in header")
                except json.JSONDecodeError:
                    pass

            chapters[0]["body"] = str(soup)

            # Log what we removed
            if removed_tags:
                removal_log_path = os.path.join(out, "removal.txt")
                with open(removal_log_path, "a", encoding="utf-8") as logf:
                    logf.write(f"{chapters[0]['title']} (Chapter {chapters[0]['num']})\n")
                    for entry in removed_tags:
                        logf.write(f"- {entry}\n")
                    logf.write("\n")


        # images
        imgdir = os.path.join(out, "images")
        os.makedirs(imgdir, exist_ok=True)
        for n in zf.namelist():
            if n.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.svg')):
                with open(os.path.join(imgdir, os.path.basename(n)), 'wb') as f:
                    f.write(zf.read(n))


    # write meta.json
    with open(os.path.join(out,"metadata.json"),'w',encoding='utf-8') as mf:
        json.dump(metadata, mf, ensure_ascii=False, indent=2)
        
    # — manual-glossary override —
    manual_gloss = os.getenv("MANUAL_GLOSSARY")
    if manual_gloss and os.path.isfile(manual_gloss):
        shutil.copy(manual_gloss, os.path.join(out, "glossary.json"))
    else:
        save_glossary(out, chapters)


    # 2) build system prompt with optional manual glossary
    glossary_path = os.path.join(out, "glossary.json")
    user_prompt  = SYSTEM_PROMPT

    if user_prompt:
        # 1) always use whatever you typed in the GUI…
        system = user_prompt
        # 2) then append your manual (or auto-generated) glossary if it exists
        if os.path.exists(glossary_path):
            with open(glossary_path, "r", encoding="utf-8") as gf:
                entries = json.load(gf)
            glossary_block = json.dumps(entries, ensure_ascii=False, indent=2)
            system += (
                "\n\nUse the following glossary entries exactly as given:\n"
                f"{glossary_block}"
            )

    elif os.path.exists(glossary_path):
        # No GUI prompt, but glossary.json is present
        with open(glossary_path, "r", encoding="utf-8") as gf:
            entries = json.load(gf)
        glossary_block = json.dumps(entries, ensure_ascii=False, indent=2)
        system = (
            "You are a professional translator. " + INSTR + "\n\n"
            "Use the following glossary entries exactly as given:\n"
            f"{glossary_block}"
        )

    else:
        # Neither user prompt nor glossary → fallback to default
        system = SYSTEM_PROMPT or (
            "You are a professional translator. " + INSTR + "\n\n"
            f"Glossary:\n{json.dumps(json.load(open(os.path.join(out,'glossary.json'))), ensure_ascii=False, indent=2)}"
        )

    base_msg = [{"role":"system","content":system}]
    history  = []
    def save_progress():
        with open(PROGRESS_FILE, "w", encoding="utf-8") as pf:
            json.dump(prog, pf, ensure_ascii=False, indent=2)
    
    total_chapters = len(chapters)
    
    for idx, c in enumerate(chapters):
        chap_num = c["num"]

        # 1) apply the user’s range filter
        if start is not None and not (start <= chap_num <= end):
            continue

        # 2) skip chapters already done
        if idx in prog["completed"]:
            print(f"[SKIP] Chapter #{idx+1} (EPUB-num {chap_num}) already done, skipping.")
            continue

        user_prompt = c["body"] 
        history = load_history()
        trimmed     = history[-HIST_LIMIT*2:] if CONTEXTUAL else []
        msgs        = base_msg + trimmed + [{"role":"user","content": user_prompt}]

        # Trim history
        if CONTEXTUAL:
            max_msgs = HIST_LIMIT * 2
            trimmed = history[-max_msgs:]
        else:
            trimmed = []
        msgs = base_msg + trimmed + [{"role":"user","content":user_prompt}]

        while True:
            try:
               
                #print(">>> FULL MSGS:", json.dumps(msgs, ensure_ascii=False, indent=2))
                # after building msgs, dump token usage vs. budget
                total_tokens = sum(len(enc.encode(m["content"])) for m in msgs)
                print(f"[DEBUG] Prompt tokens = {total_tokens} / {MAX_INPUT_TOKENS}")
                result, finish_reason = send(msgs)
                if finish_reason == "length":
                    print(f"[WARN] Output was truncated at {total_tokens} tokens!")
                # ─── If REMOVE_HEADER is on, also strip Gemini’s “Okay, I will translate…” line ───
                if REMOVE_HEADER:
                    # split into lines, preserving newline chars
                    lines = result.splitlines(True)

                    # compile regex to catch Gemini intros like:
                    # “Okay, I will translate…”, “Sure, here’s the translation…”, etc.
                    intro_re = re.compile(
                        r'^(?:okay|sure|understood|of course|got it|here(?:\'|’)s)[^\n]*\b(?:translate|translation)\b',
                        re.IGNORECASE
                    )

                    # strip off any leading intro lines
                    while lines and intro_re.match(lines[0]):
                        lines.pop(0)

                    result = "".join(lines)
                    
                    
                # Load and trim history
                history = load_history()
                old_len = len(history)
                history = history[-HIST_LIMIT * 2:]
                history_trimmed = len(history) < old_len
                if history_trimmed:
                    print(f"[DBG] Trimmed translation history from {old_len} to {len(history)} entries.")

                # Append latest user and assistant messages
                history.append({"role": "user", "content": user_prompt})
                history.append({"role": "assistant", "content": result})

                # Save updated history
                save_history(history)

                # Use rolling summary if trimming occurred and enabled
                if history_trimmed and os.getenv("USE_ROLLING_SUMMARY", "0") == "1":
                    # Summarize ONLY the most recent chapter (last assistant message)
                    recent_entries = [h for h in history[-2:] if h["role"] == "assistant"]

                    summary_prompt = (
                        "Summarize the key events, tone, and terminology used in the following translation.\n"
                        "The summary will be used to maintain consistency in the next chapter.\n\n"
                        + "\n\n".join(e["content"] for e in recent_entries)
                    )

                    summary_resp = send_to_model([
                        {"role": "system", "content": "You are a summarizer."},
                        {"role": "user", "content": summary_prompt}
                    ])

                    # Clean and purge both old files
                    summary_file = os.path.join(out, "rolling_summary.txt")
                    history_file = os.path.join(out, "translation_history.json")

                    if os.path.exists(history_file):
                        try:
                            os.remove(history_file)
                            print("[DBG] translation_history.json purged due to rolling summary.")
                        except Exception as e:
                            print(f"[WARN] Failed to delete history file: {e}")

                    if os.path.exists(summary_file):
                        try:
                            os.remove(summary_file)
                            print("[DBG] Old rolling_summary.txt removed before regeneration.")
                        except Exception as e:
                            print(f"[WARN] Failed to delete old summary: {e}")

                    with open(summary_file, "w", encoding="utf-8") as sf:
                        sf.write(summary_resp.strip())

                    # Inject new summary into prompt
                    trimmed = []  # Reset context window to empty
                    base_msg.insert(1, {
                        "role": os.getenv("SUMMARY_ROLE", "user"),
                        "content": (
                            "Here is a concise summary of the previous chapter. "
                            "Use this to ensure accurate tone, terminology, and character continuity:\n\n"
                            f"{summary_resp.strip()}"
                        )
                    })


                # Pause after success, then exit retry loop
                time.sleep(DELAY)
                break

            except UnifiedClientError as e:
                if getattr(e, "http_status", None) == 429:
                    print("⚠️ Rate limited, sleeping 60s…")
                    time.sleep(60)
                else:
                    raise

        # Now write out the HTML file
        safe_title = re.sub(r'\W+', '_', c['title'])[:40]
        fname = f"response_{c['num']:03d}_{safe_title}.html"

        # strip any ``` or ```html fences from the model output
        cleaned = re.sub(r"^```(?:html)?\s*", "", result, flags=re.MULTILINE)
        cleaned = re.sub(r"\s*```$", "", cleaned, flags=re.MULTILINE)

        # write only the cleaned HTML
        with open(os.path.join(out, fname), 'w', encoding='utf-8') as f:
            f.write(f"<h1>Chapter {c['num']}: {c['title']}</h1>\n" + cleaned)
        final_title = c['title'] or safe_title
        print(f"[Chapter {idx+1}/{total_chapters}] ✅ Saved Chapter {c['num']}: {final_title}")
        # ─── record that this chapter is done and save progress ───
        prog["completed"].append(idx)
        save_progress()

    # 3) final EPUB via EPUB compiler
    print("📘 Building final EPUB…")
    epubc = os.path.join(getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__))),
                            "epub_converter.py")
    try:
        subprocess.run([sys.executable, epubc, out], check=True)
        print("✅ All done: your final EPUB is in", out)
    except subprocess.CalledProcessError as e:
        print("❌ EPUB build failed:", e)

    # signal to GUI
    print("TRANSLATION_COMPLETE_SIGNAL")

if __name__ == "__main__":
    main()
