import os
import json
import argparse
import zipfile
from ebooklib import epub
from bs4 import BeautifulSoup
from unified_api_client import UnifiedClient
from typing import List, Dict
import re
PROGRESS_FILE = "glossary_progress.json"

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

def merge_glossary_entries(glossary):
    merged = {}

    for entry in glossary:
        key = entry['original_name']
        if key not in merged:
            merged[key] = entry.copy()
            if 'locations' in merged[key]:
                merged[key]['locations'] = list(set(x.lower() for x in merged[key].get('locations') or []))
        else:
            # Merge fields with normalization
            for field in ['locations', 'traits', 'group_affiliation']:
                old = merged[key].get(field)
                new = entry.get(field)

                if isinstance(old, str):
                    old = [old]
                if isinstance(new, str):
                    new = [new]

                old = set(x.lower() if isinstance(x, str) else x for x in (old or []))
                new = set(x.lower() if isinstance(x, str) else x for x in (new or []))
                merged[key][field] = list(old.union(new))

            # Merge how_they_refer_to_others as a union of mappings
            old_map = merged[key].get('how_they_refer_to_others', {}) or {}
            new_map = entry.get('how_they_refer_to_others', {}) or {}
            for k, v in new_map.items():
                if v is not None and k not in old_map:
                    old_map[k] = v
            merged[key]['how_they_refer_to_others'] = old_map

    # Remove null fields globally
    for entry in merged.values():
        for field in ['title', 'group_affiliation', 'traits', 'locations', 'gender']:
            if entry.get(field) is None:
                entry.pop(field, None)

        if 'how_they_refer_to_others' in entry:
            entry['how_they_refer_to_others'] = {
                k: v for k, v in entry['how_they_refer_to_others'].items() if v is not None
            }

    return list(merged.values())


    
def main():
    parser = argparse.ArgumentParser(description="Extract character glossary from EPUB via ChatGPT")
    parser.add_argument('--epub', required=True)
    parser.add_argument('--output', default='glossary.json')
    parser.add_argument('--config', default='config.json')
    args = parser.parse_args()
    # ensure we have a Glossary subfolder next to the JSON/MD outputs
    glossary_dir = os.path.join(os.path.dirname(args.output), "Glossary")
    os.makedirs(glossary_dir, exist_ok=True)
    # override the module‐level PROGRESS_FILE
    global PROGRESS_FILE
    PROGRESS_FILE = os.path.join(glossary_dir, "glossary_progress.json")


    config = load_config(args.config)
    client = UnifiedClient(model=config['model'], api_key=config['api_key'])
    model = config.get('model', 'gpt-4.1-mini')
    temp = config.get('temperature', 0.3)
    mtoks = config.get('max_tokens', 12000)
    sys_prompt = config.get('system_prompt', 'You are a helpful assistant.')
    ctx_limit = config.get('context_limit_chapters', 3)

    chapters = extract_chapters_from_epub(args.epub)
    if not chapters:
        print("No chapters found. Exiting.")
        return

    prog = load_progress()
    completed = prog['completed']
    glossary = prog['glossary']
    history = prog['context_history']

    for idx, chap in enumerate(chapters):
        if idx in completed:
            print(f"Skipping chapter {idx+1} (already processed)")
            continue
        print(f"Chapter {idx+1}/{len(chapters)}...")
        try:
            msgs = trim_context_history(history, ctx_limit) + [
                {"role":"system","content":sys_prompt},
                {"role":"user","content":build_prompt(chap)}
            ]
            resp = client.send(msgs, temperature=temp, max_tokens=mtoks)

            # Save the raw response in case you need to inspect it
            os.makedirs("Payloads", exist_ok=True)
            with open(f"Payloads/failed_response_chap{idx+1}.txt", "w", encoding="utf-8") as f:
                f.write(resp)

            # Extract the JSON array itself (strip any leading/trailing prose)
            m = re.search(r"\[.*\]", resp, re.DOTALL)
            json_str = m.group(0) if m else resp

            # Parse *only* the cleaned-up JSON
            try:
                data = json.loads(json_str)
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

        except Exception as e:
            print(f"Error at chapter {idx+1}: {e}")

    print(f"Done. Glossary saved to {args.output}")

if __name__=='__main__':
    main()
