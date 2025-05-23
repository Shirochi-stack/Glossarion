# -*- coding: utf-8 -*-
import json
import logging
import shutil
# optional: turn on HTTP‐level debugging in the OpenAI client
logging.basicConfig(level=logging.DEBUG)

import os, sys, io, zipfile, time, json, re, textwrap, mimetypes, subprocess
import ebooklib                                        # needed for ITEM_DOCUMENT
from ebooklib import epub
from bs4 import BeautifulSoup
from collections import Counter
from unified_api_client import UnifiedClient


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

EPUB_PATH       = os.getenv("EPUB_PATH", "default.epub")
MODEL           = os.getenv("MODEL", "gpt-4.1-nano")
TRANSLATION_LANG= os.getenv("TRANSLATION_LANG", "korean").lower()
CONTEXTUAL      = os.getenv("CONTEXTUAL", "1") == "1"
DELAY           = int(os.getenv("SEND_INTERVAL_SECONDS", "2"))
SYSTEM_PROMPT   = os.getenv("SYSTEM_PROMPT", "").strip()

client = UnifiedClient(model=MODEL, api_key=API_KEY)

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

def send(messages):
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
    return client.send(messages, temperature=TEMP, max_tokens=12000)



# ---------- MAIN ----------
def main():
    base_out = "output"
    # make a unique output dir
    out = base_out
    i = 1
    while os.path.exists(out):
        out = f"{base_out}{i}"
        i += 1
    os.makedirs(out)

    # 1) unpack and collect
    with zipfile.ZipFile(EPUB_PATH, 'r') as zf:
        metadata = extract_epub_metadata(zf)
        chapters = extract_chapters(zf)
        # images
        imgdir = os.path.join(out, "images")
        os.makedirs(imgdir, exist_ok=True)
        for n in zf.namelist():
            if n.lower().endswith(('.png','.jpg','.jpeg','.gif','.svg')):
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


    for c in chapters:
        user_prompt = c["body"] 
        history = load_history()

        # Trim history
        if CONTEXTUAL:
            max_msgs = HIST_LIMIT * 2
            trimmed = history[-max_msgs:]
        else:
            trimmed = []
        msgs = base_msg + trimmed + [{"role":"user","content":user_prompt}]

        while True:
            try:
                result = send(msgs)

                # Record this turn and persist it
                history.append({"role":"user",      "content": user_prompt})
                history.append({"role":"assistant", "content": result})
                save_history(history)

                # Pause after success, then exit retry loop
                time.sleep(DELAY)
                break

            except OpenAIError as e:
                if getattr(e, "http_status", None) == 429:
                    print("⚠️ Rate limited, sleeping 60s…")
                    time.sleep(60)
                else:
                    raise

        # Now write out the HTML file
        safe_title = re.sub(r'\W+', '_', c['title'])[:40]
        fname = f"response_{c['num']:03d}_{safe_title}.html"
        with open(os.path.join(out, fname), 'w', encoding='utf-8') as f:
            f.write(f"<h1>Chapter {c['num']}: {c['title']}</h1>\n" + result)
        print(f"✅ Saved chapter {c['num']}")

    # 3) final EPUB via fallback compiler
    print("📘 Building final EPUB…")
    fallback = os.path.join(getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__))),
                            "epub_fallback_compiler_with_cover_portable.py")
    try:
        subprocess.run([sys.executable, fallback, out], check=True)
        print("✅ All done: your final EPUB is in", out)
    except subprocess.CalledProcessError as e:
        print("❌ EPUB build failed:", e)

    # signal to GUI
    print("TRANSLATION_COMPLETE_SIGNAL")

if __name__ == "__main__":
    main()
