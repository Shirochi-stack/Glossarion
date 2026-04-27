# rpgmaker_handler.py - GTool: RPG Maker Translation Engine
# Detects RPG Maker version, extracts translatable text, sends through
# Glossarion's translation pipeline, and patches translations back.
import os
import sys
import json
import re
import shutil
import time
import hashlib
from typing import Dict, List, Tuple, Optional, Callable

# RPG Maker MV/MZ event command codes that contain translatable text
_DIALOG_CODES = {401, 405}  # Show Text, Show Scrolling Text
_CHOICE_CODE = 102           # Show Choices
_NAME_CODE = 320             # Change Name (rare)
_NICKNAME_CODE = 324         # Change Nickname

# Data files that contain translatable fields
_DB_FILES = [
    "Actors.json", "Armors.json", "Classes.json", "CommonEvents.json",
    "Enemies.json", "Items.json", "Skills.json", "States.json",
    "System.json", "Weapons.json",
]

# Fields to extract from database entries
_TRANSLATABLE_FIELDS = ["name", "description", "message1", "message2",
                        "message3", "message4", "nickname", "profile", "note"]

# System.json specific translatable paths
_SYSTEM_FIELDS = ["gameTitle"]
_SYSTEM_TERMS_FIELDS = [
    "basic", "commands", "params", "messages"
]


class RPGMakerVersion:
    MV = "mv"
    MZ = "mz"
    VX_ACE = "vxace"
    XP = "xp"
    UNKNOWN = "unknown"


def detect_version(game_dir: str) -> Tuple[str, str]:
    """Detect RPG Maker version and return (version, data_dir_path).

    Checks for signature files/folders near the game executable.
    """
    # MV: www/data/ or www/js/
    www_data = os.path.join(game_dir, "www", "data")
    if os.path.isdir(www_data):
        # Check for MZ vs MV - MZ has effekseer or rmmz in js
        js_dir = os.path.join(game_dir, "www", "js")
        if os.path.isdir(js_dir):
            for f in os.listdir(js_dir):
                if "rmmz" in f.lower():
                    return RPGMakerVersion.MZ, www_data
        return RPGMakerVersion.MV, www_data

    # MZ (newer layout): data/ directly alongside exe
    direct_data = os.path.join(game_dir, "data")
    if os.path.isdir(direct_data):
        # Verify it has RPG Maker JSON files
        for f in os.listdir(direct_data):
            if f.endswith(".json") and f in _DB_FILES + ["Map001.json"]:
                # Check for MZ indicators
                js_dir = os.path.join(game_dir, "js")
                if os.path.isdir(js_dir):
                    for jf in os.listdir(js_dir):
                        if "rmmz" in jf.lower():
                            return RPGMakerVersion.MZ, direct_data
                return RPGMakerVersion.MV, direct_data

    # VX Ace: Data/*.rvdata2
    vxa_data = os.path.join(game_dir, "Data")
    if os.path.isdir(vxa_data):
        for f in os.listdir(vxa_data):
            if f.endswith(".rvdata2"):
                return RPGMakerVersion.VX_ACE, vxa_data
            if f.endswith(".rxdata"):
                return RPGMakerVersion.XP, vxa_data

    return RPGMakerVersion.UNKNOWN, ""


def _is_translatable(text: str) -> bool:
    """Check if a string is worth translating (non-empty, has real text)."""
    if not text or not text.strip():
        return False
    # Skip pure numbers, control codes, filenames
    stripped = text.strip()
    if stripped.isdigit():
        return False
    if re.match(r'^[A-Za-z0-9_./\\]+\.(png|ogg|mp3|wav|m4a)$', stripped):
        return False
    # Skip very short strings that are just punctuation
    if len(stripped) <= 1 and not stripped.isalpha():
        return False
    return True


def _extract_event_strings(commands: list) -> List[Tuple[int, str]]:
    """Extract translatable strings from RPG Maker event command list.

    Returns list of (index, text) tuples.
    """
    results = []
    if not commands:
        return results
    for i, cmd in enumerate(commands):
        if not isinstance(cmd, dict):
            continue
        code = cmd.get("code", 0)
        params = cmd.get("parameters", [])
        if code in _DIALOG_CODES and params:
            text = params[0] if isinstance(params[0], str) else ""
            if _is_translatable(text):
                results.append((i, text))
        elif code == _CHOICE_CODE and params:
            choices = params[0] if isinstance(params[0], list) else []
            for ci, choice in enumerate(choices):
                if isinstance(choice, str) and _is_translatable(choice):
                    results.append((i, choice))
    return results


def extract_map_strings(data_dir: str, log: Callable = print
                        ) -> Dict[str, Dict[str, str]]:
    """Extract all translatable strings from Map*.json files.

    Returns {filename: {key: original_text}} where key encodes location.
    """
    all_strings = {}
    map_files = sorted([f for f in os.listdir(data_dir)
                        if re.match(r'^Map\d+\.json$', f)])
    log(f"🗺️ Found {len(map_files)} map files")

    for mf in map_files:
        path = os.path.join(data_dir, mf)
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            log(f"⚠️ Failed to read {mf}: {e}")
            continue

        strings = {}
        # Map display name
        display_name = data.get("displayName", "")
        if _is_translatable(display_name):
            strings["displayName"] = display_name

        # Events
        events = data.get("events", [])
        for ev_idx, event in enumerate(events):
            if not event or not isinstance(event, dict):
                continue
            ev_name = event.get("name", "")
            if _is_translatable(ev_name):
                strings[f"event_{ev_idx}_name"] = ev_name

            pages = event.get("pages", [])
            for pg_idx, page in enumerate(pages):
                if not page or not isinstance(page, dict):
                    continue
                cmds = page.get("list", [])
                for cmd_idx, text in _extract_event_strings(cmds):
                    key = f"event_{ev_idx}_page_{pg_idx}_cmd_{cmd_idx}"
                    strings[key] = text

        if strings:
            all_strings[mf] = strings
            log(f"   📄 {mf}: {len(strings)} strings")

    return all_strings


def extract_db_strings(data_dir: str, log: Callable = print
                       ) -> Dict[str, Dict[str, str]]:
    """Extract translatable strings from database JSON files."""
    all_strings = {}

    for db_file in _DB_FILES:
        path = os.path.join(data_dir, db_file)
        if not os.path.exists(path):
            continue
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            log(f"⚠️ Failed to read {db_file}: {e}")
            continue

        strings = {}

        if db_file == "System.json":
            # System has a unique structure
            for field in _SYSTEM_FIELDS:
                val = data.get(field, "")
                if _is_translatable(val):
                    strings[field] = val
            # Terms
            terms = data.get("terms", {})
            for section in _SYSTEM_TERMS_FIELDS:
                section_data = terms.get(section, {})
                if isinstance(section_data, list):
                    for idx, val in enumerate(section_data):
                        if isinstance(val, str) and _is_translatable(val):
                            strings[f"terms_{section}_{idx}"] = val
                elif isinstance(section_data, dict):
                    for k, v in section_data.items():
                        if isinstance(v, str) and _is_translatable(v):
                            strings[f"terms_{section}_{k}"] = v
        elif db_file == "CommonEvents.json":
            # CommonEvents is an array of events
            if isinstance(data, list):
                for idx, event in enumerate(data):
                    if not event or not isinstance(event, dict):
                        continue
                    ev_name = event.get("name", "")
                    if _is_translatable(ev_name):
                        strings[f"ce_{idx}_name"] = ev_name
                    cmds = event.get("list", [])
                    for cmd_idx, text in _extract_event_strings(cmds):
                        strings[f"ce_{idx}_cmd_{cmd_idx}"] = text
        else:
            # Standard DB array (Actors, Items, etc.)
            if isinstance(data, list):
                for idx, entry in enumerate(data):
                    if not entry or not isinstance(entry, dict):
                        continue
                    for field in _TRANSLATABLE_FIELDS:
                        val = entry.get(field, "")
                        if isinstance(val, str) and _is_translatable(val):
                            strings[f"{idx}_{field}"] = val

        if strings:
            all_strings[db_file] = strings
            log(f"   📦 {db_file}: {len(strings)} strings")

    return all_strings


def extract_all(game_dir: str, log: Callable = print
                ) -> Tuple[str, str, Dict[str, Dict[str, str]]]:
    """Full extraction pipeline. Returns (version, data_dir, all_strings)."""
    version, data_dir = detect_version(game_dir)
    if version == RPGMakerVersion.UNKNOWN:
        log("❌ Could not detect RPG Maker version")
        return version, "", {}

    log(f"🎮 Detected RPG Maker {version.upper()}")
    log(f"📁 Data directory: {data_dir}")

    if version in (RPGMakerVersion.MV, RPGMakerVersion.MZ):
        db_strings = extract_db_strings(data_dir, log)
        map_strings = extract_map_strings(data_dir, log)
        all_strings = {**db_strings, **map_strings}

        total = sum(len(v) for v in all_strings.values())
        log(f"📊 Total extractable strings: {total}")
        return version, data_dir, all_strings
    else:
        log(f"⚠️ {version} support coming soon - currently MV/MZ only")
        return version, data_dir, {}


def create_translation_file(game_dir: str, all_strings: Dict,
                            log: Callable = print) -> str:
    """Save extracted strings to a translation JSON file."""
    out_dir = os.path.join(game_dir, "GTool_Translation")
    os.makedirs(out_dir, exist_ok=True)

    trans_path = os.path.join(out_dir, "translation_map.json")
    # Format: {file: {key: {"original": text, "translated": ""}}}
    trans_data = {}
    for filename, strings in all_strings.items():
        trans_data[filename] = {}
        for key, text in strings.items():
            trans_data[filename][key] = {
                "original": text,
                "translated": ""
            }

    with open(trans_path, 'w', encoding='utf-8') as f:
        json.dump(trans_data, f, ensure_ascii=False, indent=2)

    total = sum(len(v) for v in trans_data.values())
    log(f"💾 Saved translation map: {trans_path} ({total} entries)")
    return trans_path


def build_translation_chunks(all_strings: Dict, max_chars: int = 4000
                             ) -> List[Dict]:
    """Group extracted strings into chunks for batch translation.

    Each chunk is sent as one API request (like a "chapter").
    Returns list of dicts with 'file', 'keys', 'texts' fields.
    """
    chunks = []
    current_chunk = {"file": "", "keys": [], "texts": []}
    current_len = 0

    for filename, strings in all_strings.items():
        for key, text in strings.items():
            text_len = len(text)
            if current_len + text_len > max_chars and current_chunk["keys"]:
                chunks.append(current_chunk)
                current_chunk = {"file": "", "keys": [], "texts": []}
                current_len = 0

            current_chunk["file"] = filename
            current_chunk["keys"].append(f"{filename}::{key}")
            current_chunk["texts"].append(text)
            current_len += text_len

    if current_chunk["keys"]:
        chunks.append(current_chunk)

    return chunks


def format_chunk_for_translation(chunk: Dict) -> str:
    """Format a chunk into a numbered list for the AI to translate."""
    lines = []
    for i, text in enumerate(chunk["texts"], 1):
        lines.append(f"[{i}] {text}")
    return "\n".join(lines)


def parse_translated_chunk(response: str, chunk: Dict) -> Dict[str, str]:
    """Parse numbered translation response back into key->translation map."""
    translations = {}
    keys = chunk["keys"]

    # Try to parse [N] format
    pattern = re.compile(r'^\[(\d+)\]\s*(.+)$', re.MULTILINE)
    matches = pattern.findall(response)

    if matches:
        for num_str, text in matches:
            idx = int(num_str) - 1
            if 0 <= idx < len(keys):
                translations[keys[idx]] = text.strip()
    else:
        # Fallback: split by lines and match 1:1
        lines = [l.strip() for l in response.strip().split('\n') if l.strip()]
        for i, line in enumerate(lines):
            if i < len(keys):
                # Strip any leading number markers
                clean = re.sub(r'^\d+[\.\)\]]\s*', '', line)
                translations[keys[i]] = clean

    return translations


def apply_translations(data_dir: str, trans_map_path: str,
                       log: Callable = print) -> bool:
    """Apply translations from the translation map back to game files.

    Creates backups before modifying any file.
    """
    try:
        with open(trans_map_path, 'r', encoding='utf-8') as f:
            trans_data = json.load(f)
    except Exception as e:
        log(f"❌ Failed to load translation map: {e}")
        return False

    backup_dir = os.path.join(os.path.dirname(trans_map_path), "originals_backup")
    os.makedirs(backup_dir, exist_ok=True)

    patched = 0
    for filename, entries in trans_data.items():
        file_path = os.path.join(data_dir, filename)
        if not os.path.exists(file_path):
            continue

        # Backup original
        backup_path = os.path.join(backup_dir, filename)
        if not os.path.exists(backup_path):
            shutil.copy2(file_path, backup_path)

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            log(f"⚠️ Failed to read {filename}: {e}")
            continue

        file_patched = 0
        for key, entry in entries.items():
            translated = entry.get("translated", "")
            if not translated:
                continue

            if filename.startswith("Map"):
                file_patched += _patch_map_entry(data, key, translated)
            elif filename == "System.json":
                file_patched += _patch_system_entry(data, key, translated)
            elif filename == "CommonEvents.json":
                file_patched += _patch_common_event(data, key, translated)
            else:
                file_patched += _patch_db_entry(data, key, translated)

        if file_patched > 0:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False)
            patched += file_patched
            log(f"   ✅ {filename}: {file_patched} strings patched")

    log(f"🎮 Total: {patched} strings patched into game files")
    return True


def _patch_map_entry(data: dict, key: str, translated: str) -> int:
    """Patch a single map entry."""
    if key == "displayName":
        data["displayName"] = translated
        return 1

    m = re.match(r'event_(\d+)_name', key)
    if m:
        idx = int(m.group(1))
        events = data.get("events", [])
        if idx < len(events) and events[idx]:
            events[idx]["name"] = translated
            return 1
        return 0

    m = re.match(r'event_(\d+)_page_(\d+)_cmd_(\d+)', key)
    if m:
        ev_idx, pg_idx, cmd_idx = int(m.group(1)), int(m.group(2)), int(m.group(3))
        try:
            cmd = data["events"][ev_idx]["pages"][pg_idx]["list"][cmd_idx]
            code = cmd.get("code", 0)
            if code in _DIALOG_CODES:
                cmd["parameters"][0] = translated
                return 1
            elif code == _CHOICE_CODE:
                # Find which choice matches - for now patch all
                pass
        except (IndexError, KeyError, TypeError):
            pass
    return 0


def _patch_system_entry(data: dict, key: str, translated: str) -> int:
    if key in _SYSTEM_FIELDS:
        data[key] = translated
        return 1
    m = re.match(r'terms_(\w+)_(.+)', key)
    if m:
        section, sub = m.group(1), m.group(2)
        terms = data.get("terms", {})
        section_data = terms.get(section, {})
        if isinstance(section_data, list):
            try:
                section_data[int(sub)] = translated
                return 1
            except (ValueError, IndexError):
                pass
        elif isinstance(section_data, dict):
            if sub in section_data:
                section_data[sub] = translated
                return 1
    return 0


def _patch_common_event(data: list, key: str, translated: str) -> int:
    m = re.match(r'ce_(\d+)_name', key)
    if m:
        idx = int(m.group(1))
        if idx < len(data) and data[idx]:
            data[idx]["name"] = translated
            return 1
        return 0
    m = re.match(r'ce_(\d+)_cmd_(\d+)', key)
    if m:
        ev_idx, cmd_idx = int(m.group(1)), int(m.group(2))
        try:
            cmd = data[ev_idx]["list"][cmd_idx]
            if cmd.get("code", 0) in _DIALOG_CODES:
                cmd["parameters"][0] = translated
                return 1
        except (IndexError, KeyError, TypeError):
            pass
    return 0


def _patch_db_entry(data: list, key: str, translated: str) -> int:
    m = re.match(r'(\d+)_(\w+)', key)
    if m:
        idx, field = int(m.group(1)), m.group(2)
        if idx < len(data) and data[idx] and isinstance(data[idx], dict):
            if field in data[idx]:
                data[idx][field] = translated
                return 1
    return 0


def get_progress_path(game_dir: str) -> str:
    return os.path.join(game_dir, "GTool_Translation", "progress.json")


def save_progress(game_dir: str, translated_keys: Dict[str, str]):
    """Save translation progress for resume capability."""
    path = get_progress_path(game_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(translated_keys, f, ensure_ascii=False, indent=2)


def load_progress(game_dir: str) -> Dict[str, str]:
    """Load previous translation progress."""
    path = get_progress_path(game_dir)
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def process_game(exe_path: str, log: Callable = print,
                 stop_check: Callable = None,
                 translate_func: Callable = None) -> bool:
    """Main entry point: extract, translate, and patch an RPG Maker game.

    Args:
        exe_path: Path to the game's .exe file
        log: Logging callback
        stop_check: Returns True if user requested stop
        translate_func: Callable(source_text, chunk_info) -> translated_text
    """
    game_dir = os.path.dirname(os.path.abspath(exe_path))
    log(f"🎮 GTool: Processing game at {game_dir}")

    # Detect version
    version, data_dir, all_strings = extract_all(game_dir, log)
    if not all_strings:
        log("❌ No translatable strings found")
        return False

    # Create translation file
    trans_path = create_translation_file(game_dir, all_strings, log)

    # Load previous progress
    progress = load_progress(game_dir)
    if progress:
        log(f"📋 Resuming: {len(progress)} strings already translated")

    # Build chunks
    chunks = build_translation_chunks(all_strings)
    log(f"📦 Split into {len(chunks)} translation chunks")

    if not translate_func:
        log("ℹ️ No translation function provided - extraction only")
        log(f"💾 Translation map saved to: {trans_path}")
        return True

    # Translate each chunk
    total_translated = 0
    for i, chunk in enumerate(chunks):
        if stop_check and stop_check():
            log("⏹️ Translation stopped by user")
            break

        # Skip already-translated keys
        untranslated_keys = []
        untranslated_texts = []
        for k, t in zip(chunk["keys"], chunk["texts"]):
            if k not in progress:
                untranslated_keys.append(k)
                untranslated_texts.append(t)

        if not untranslated_keys:
            continue

        sub_chunk = {"keys": untranslated_keys, "texts": untranslated_texts}
        source = format_chunk_for_translation(sub_chunk)

        log(f"🔄 Translating chunk {i+1}/{len(chunks)} "
            f"({len(untranslated_keys)} strings)...")

        try:
            response = translate_func(source, {
                "chunk_index": i,
                "total_chunks": len(chunks),
            })
            if response:
                parsed = parse_translated_chunk(response, sub_chunk)
                progress.update(parsed)
                total_translated += len(parsed)
                save_progress(game_dir, progress)
        except Exception as e:
            log(f"⚠️ Chunk {i+1} failed: {e}")

    log(f"✅ Translated {total_translated} strings total")

    # Update translation map with results
    try:
        with open(trans_path, 'r', encoding='utf-8') as f:
            trans_data = json.load(f)
        for full_key, translated in progress.items():
            parts = full_key.split("::", 1)
            if len(parts) == 2:
                fn, key = parts
                if fn in trans_data and key in trans_data[fn]:
                    trans_data[fn][key]["translated"] = translated
        with open(trans_path, 'w', encoding='utf-8') as f:
            json.dump(trans_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        log(f"⚠️ Failed to update translation map: {e}")

    # Apply translations to game files
    log("🔧 Applying translations to game files...")
    apply_translations(data_dir, trans_path, log)

    log("🎮 GTool: Game translation complete!")
    return True
