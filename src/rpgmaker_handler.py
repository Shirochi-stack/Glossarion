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
    VX = "vx"
    XP = "xp"
    RM2K3 = "rm2k3"
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

    # VX Ace / VX / XP: Data/*.rvdata2 or *.rvdata or *.rxdata
    vxa_data = os.path.join(game_dir, "Data")
    if os.path.isdir(vxa_data):
        files = os.listdir(vxa_data)
        for f in files:
            if f.endswith(".rvdata2"):
                return RPGMakerVersion.VX_ACE, vxa_data
        for f in files:
            if f.endswith(".rvdata"):
                return RPGMakerVersion.VX, vxa_data
        for f in files:
            if f.endswith(".rxdata"):
                return RPGMakerVersion.XP, vxa_data

    # RPG Maker 2000/2003: root dir contains RPG_RT.ldb
    for f in os.listdir(game_dir):
        if f.lower() == "rpg_rt.ldb":
            return RPGMakerVersion.RM2K3, game_dir

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


def _extract_event_strings(commands: list) -> List[Tuple[str, str]]:
    """Extract translatable strings from RPG Maker event command list.

    Groups consecutive code-401 lines into single message blocks so the AI
    translates each dialog box as one unit instead of line-by-line.
    Returns list of (key_suffix, text) tuples.
    """
    results = []
    if not commands:
        return results

    i = 0
    while i < len(commands):
        cmd = commands[i]
        if not isinstance(cmd, dict):
            i += 1
            continue
        code = cmd.get("code", 0)
        params = cmd.get("parameters", [])

        if code in _DIALOG_CODES:
            # Collect ALL consecutive lines of this dialog block
            block_start = i
            lines = []
            while i < len(commands):
                c = commands[i]
                if not isinstance(c, dict) or c.get("code", 0) != code:
                    break
                p = c.get("parameters", [])
                line = p[0] if p and isinstance(p[0], str) else ""
                lines.append(line)
                i += 1
            # Join lines with newline — translate as a single message
            full_text = "\n".join(lines)
            if _is_translatable(full_text):
                # Key encodes the range: cmd_{start}_{end}
                key = f"msg_{block_start}_{block_start + len(lines) - 1}"
                results.append((key, full_text))
            continue

        elif code == 101 and len(params) > 4:
            # MZ Show Text header — params[4] is the speaker name box
            speaker = params[4]
            if isinstance(speaker, str) and _is_translatable(speaker):
                results.append((f"speaker_{i}", speaker))

        elif code == _CHOICE_CODE and params:
            choices = params[0] if isinstance(params[0], list) else []
            for ci, choice in enumerate(choices):
                if isinstance(choice, str) and _is_translatable(choice):
                    results.append((f"choice_{i}_{ci}", choice))

        i += 1

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
                for key_suffix, text in _extract_event_strings(cmds):
                    key = f"event_{ev_idx}_page_{pg_idx}_{key_suffix}"
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
                    for key_suffix, text in _extract_event_strings(cmds):
                        strings[f"ce_{idx}_{key_suffix}"] = text
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


# ============================================================
# Ruby Marshal parser (pure Python) — for VX Ace / VX / XP
# ============================================================
import struct
import io

class RubyMarshalReader:
    """Minimal Ruby Marshal 4.8 reader.  Enough to extract RPG Maker objects."""

    def __init__(self, data: bytes):
        self._io = io.BytesIO(data)
        self._symbols = []
        self._objects = []

    # -- primitives --
    def _read_byte(self) -> int:
        b = self._io.read(1)
        if not b:
            raise EOFError
        return b[0]

    def _read_bytes(self, n: int) -> bytes:
        return self._io.read(n)

    def _read_fixnum(self) -> int:
        c = struct.unpack('b', self._io.read(1))[0]
        if c == 0:
            return 0
        if 5 < c:
            return c - 5
        if -5 <= c < 0:
            return c + 5
        if c > 0:
            x = 0
            for i in range(c):
                x |= self._read_byte() << (8 * i)
            return x
        c = -c
        x = -1
        for i in range(c):
            x &= ~(0xff << (8 * i))
            x |= self._read_byte() << (8 * i)
        return x

    def _read_raw_string(self) -> bytes:
        n = self._read_fixnum()
        return self._read_bytes(n)

    # -- top-level --
    def load(self):
        major = self._read_byte()
        minor = self._read_byte()
        return self._read_value()

    def _read_value(self):
        tag = chr(self._read_byte())

        if tag == '0':  # nil
            return None
        if tag == 'T':
            return True
        if tag == 'F':
            return False
        if tag == 'i':  # Fixnum
            return self._read_fixnum()
        if tag == 'f':  # Float
            s = self._read_raw_string()
            return float(s)
        if tag == ':':  # Symbol
            name = self._read_raw_string().decode('utf-8', errors='replace')
            self._symbols.append(name)
            return ('__sym__', name)
        if tag == ';':  # Symbol ref
            idx = self._read_fixnum()
            name = self._symbols[idx] if idx < len(self._symbols) else f'sym_{idx}'
            return ('__sym__', name)
        if tag == '"':  # Raw string
            s = self._read_raw_string()
            obj_id = len(self._objects)
            self._objects.append(s)
            return s
        if tag == 'I':  # Instance (usually string with encoding)
            obj = self._read_value()
            n_ivars = self._read_fixnum()
            encoding = 'utf-8'
            for _ in range(n_ivars):
                k = self._read_value()
                v = self._read_value()
                k_name = k[1] if isinstance(k, tuple) and k[0] == '__sym__' else str(k)
                if k_name == 'E' and v is True:
                    encoding = 'utf-8'
                elif k_name == 'encoding':
                    if isinstance(v, bytes):
                        encoding = v.decode('ascii', errors='replace')
            if isinstance(obj, bytes):
                try:
                    return obj.decode(encoding, errors='replace')
                except Exception:
                    return obj.decode('utf-8', errors='replace')
            return obj
        if tag == '[':  # Array
            n = self._read_fixnum()
            arr = []
            obj_id = len(self._objects)
            self._objects.append(arr)
            for _ in range(n):
                arr.append(self._read_value())
            return arr
        if tag == '{':  # Hash
            n = self._read_fixnum()
            h = {}
            obj_id = len(self._objects)
            self._objects.append(h)
            for _ in range(n):
                k = self._read_value()
                v = self._read_value()
                # Flatten symbol keys
                if isinstance(k, tuple) and k[0] == '__sym__':
                    k = k[1]
                h[k] = v
            return h
        if tag == 'o':  # Object
            klass = self._read_value()
            klass_name = klass[1] if isinstance(klass, tuple) and klass[0] == '__sym__' else str(klass)
            n_ivars = self._read_fixnum()
            obj = {'__class__': klass_name}
            obj_id = len(self._objects)
            self._objects.append(obj)
            for _ in range(n_ivars):
                k = self._read_value()
                v = self._read_value()
                k_name = k[1] if isinstance(k, tuple) and k[0] == '__sym__' else str(k)
                # Strip leading @ from instance var names
                if k_name.startswith('@'):
                    k_name = k_name[1:]
                obj[k_name] = v
            return obj
        if tag == '@':  # Object ref
            idx = self._read_fixnum()
            return self._objects[idx] if idx < len(self._objects) else None
        if tag == 'l':  # Bignum
            sign = chr(self._read_byte())
            n = self._read_fixnum()
            val = 0
            for i in range(n * 2):
                val |= self._read_byte() << (8 * i)
            return val if sign == '+' else -val
        if tag == 'u':  # User-defined (Table, Tone, Color, etc.) — skip data
            klass = self._read_value()
            data = self._read_raw_string()
            klass_name = klass[1] if isinstance(klass, tuple) and klass[0] == '__sym__' else str(klass)
            obj = {'__class__': klass_name, '__userdata__': True}
            self._objects.append(obj)
            return obj
        if tag == 'U':  # User marshal
            klass = self._read_value()
            val = self._read_value()
            return val
        if tag == 'e':  # Extended module
            mod = self._read_value()
            return self._read_value()
        if tag == 'C':  # Subclass of built-in
            klass = self._read_value()
            return self._read_value()

        # Unknown tag — try to skip gracefully
        return None


def _marshal_load(filepath: str):
    """Load a Ruby Marshal file and return the deserialized Python object."""
    with open(filepath, 'rb') as f:
        data = f.read()
    reader = RubyMarshalReader(data)
    return reader.load()


def _extract_marshal_strings(obj, prefix: str, strings: dict):
    """Recursively extract translatable strings from a deserialized Marshal object."""
    if obj is None:
        return
    if isinstance(obj, dict):
        cls = obj.get('__class__', '')
        # Skip non-data classes
        if obj.get('__userdata__'):
            return
        for field in _TRANSLATABLE_FIELDS + ['display_name']:
            val = obj.get(field)
            if isinstance(val, str) and _is_translatable(val):
                strings[f"{prefix}_{field}"] = val
        # Event commands — list of RPG::EventCommand objects
        cmd_list = obj.get('list')
        if isinstance(cmd_list, list):
            for ci, cmd in enumerate(cmd_list):
                if isinstance(cmd, dict) and cmd.get('__class__', '').endswith('EventCommand'):
                    code = cmd.get('code', 0)
                    params = cmd.get('parameters', [])
                    if code in _DIALOG_CODES and params:
                        text = params[0] if isinstance(params[0], str) else None
                        if text and _is_translatable(text):
                            strings[f"{prefix}_cmd_{ci}"] = text
                    elif code == _CHOICE_CODE and params:
                        choices = params[0] if isinstance(params[0], list) else []
                        for chi, ch in enumerate(choices):
                            if isinstance(ch, str) and _is_translatable(ch):
                                strings[f"{prefix}_cmd_{ci}_ch_{chi}"] = ch
        # Pages (for map events)
        pages = obj.get('pages')
        if isinstance(pages, list):
            for pi, page in enumerate(pages):
                if isinstance(page, dict):
                    _extract_marshal_strings(page, f"{prefix}_pg_{pi}", strings)
        # Events hash (for maps)
        events = obj.get('events')
        if isinstance(events, dict):
            for ev_id, ev in events.items():
                if isinstance(ev, dict):
                    ev_name = ev.get('name', '')
                    if isinstance(ev_name, str) and _is_translatable(ev_name):
                        strings[f"{prefix}_ev_{ev_id}_name"] = ev_name
                    _extract_marshal_strings(ev, f"{prefix}_ev_{ev_id}", strings)
    elif isinstance(obj, list):
        for idx, item in enumerate(obj):
            if isinstance(item, dict):
                _extract_marshal_strings(item, f"{prefix}_{idx}", strings)


def extract_marshal_data(data_dir: str, ext: str, log: Callable = print
                         ) -> Dict[str, Dict[str, str]]:
    """Extract translatable strings from Ruby Marshal data files (.rvdata2/.rvdata/.rxdata)."""
    all_strings = {}
    files = sorted([f for f in os.listdir(data_dir) if f.endswith(ext)])
    log(f"📦 Found {len(files)} {ext} files")

    # Database files to process
    db_names = ['Actors', 'Armors', 'Classes', 'CommonEvents', 'Enemies',
                'Items', 'Skills', 'States', 'System', 'Weapons']

    for fname in files:
        path = os.path.join(data_dir, fname)
        base = os.path.splitext(fname)[0]
        try:
            obj = _marshal_load(path)
        except Exception as e:
            log(f"   ⚠️ Failed to parse {fname}: {e}")
            continue

        strings = {}

        if base.startswith('Map') and base[3:].isdigit():
            # Map file
            if isinstance(obj, dict):
                dn = obj.get('display_name', '')
                if isinstance(dn, str) and _is_translatable(dn):
                    strings['display_name'] = dn
                _extract_marshal_strings(obj, 'map', strings)
        elif base in db_names:
            if isinstance(obj, list):
                for idx, entry in enumerate(obj):
                    if isinstance(entry, dict):
                        _extract_marshal_strings(entry, f"{idx}", strings)
            elif isinstance(obj, dict):
                _extract_marshal_strings(obj, base.lower(), strings)
        elif base == 'System':
            if isinstance(obj, dict):
                for field in ['game_title', 'title', 'currency_unit']:
                    val = obj.get(field)
                    if isinstance(val, str) and _is_translatable(val):
                        strings[field] = val
                # Words/terms
                words = obj.get('words') or obj.get('terms')
                if isinstance(words, dict):
                    for k, v in words.items():
                        if isinstance(v, str) and _is_translatable(v):
                            strings[f"terms_{k}"] = v
                        elif isinstance(v, list):
                            for vi, vv in enumerate(v):
                                if isinstance(vv, str) and _is_translatable(vv):
                                    strings[f"terms_{k}_{vi}"] = vv
        else:
            # Generic — try to extract from any structure
            if isinstance(obj, list):
                for idx, entry in enumerate(obj):
                    if isinstance(entry, dict):
                        _extract_marshal_strings(entry, f"{idx}", strings)

        if strings:
            all_strings[fname] = strings
            log(f"   📄 {fname}: {len(strings)} strings")

    return all_strings


# ============================================================
# RPG Maker 2000/2003 binary parser (.ldb / .lmu / .lmt)
# ============================================================

def _ber_read(f) -> int:
    """Read a BER-encoded variable-length integer."""
    result = 0
    shift = 0
    while True:
        b = f.read(1)
        if not b:
            raise EOFError
        byte = b[0]
        result |= (byte & 0x7F) << shift
        if not (byte & 0x80):
            break
        shift += 7
    return result


def _read_rm2k_string(f, encoding='shift_jis') -> str:
    """Read a BER-length-prefixed string."""
    length = _ber_read(f)
    raw = f.read(length)
    try:
        return raw.decode(encoding, errors='replace')
    except Exception:
        return raw.decode('utf-8', errors='replace')


def _skip_rm2k_chunk(f):
    """Skip a BER-length-prefixed chunk."""
    length = _ber_read(f)
    f.read(length)


def extract_rm2k3_ldb(filepath: str, log: Callable = print
                      ) -> Dict[str, Dict[str, str]]:
    """Extract translatable strings from RPG Maker 2000/2003 LDB (database) file."""
    all_strings = {}
    basename = os.path.basename(filepath)

    # Detect encoding — try shift_jis first (Japanese), then cp949 (Korean), gbk (Chinese)
    encoding = 'shift_jis'

    try:
        with open(filepath, 'rb') as f:
            # Read header string
            header = _read_rm2k_string(f, 'ascii')
            if 'RPG_RT' not in header and 'LcfDataBase' not in header:
                log(f"⚠️ {basename}: unexpected header '{header}'")
                return {}

            strings = {}
            # The LDB is a sequence of tagged sections
            # Each section: tag(BER) + length(BER) + data
            # Key sections: 0x0B=Actors, 0x0C=Skills, 0x0D=Items,
            #   0x0E=Enemies, 0x11=Classes, 0x12=System terms
            section_names = {
                0x0B: 'Actors', 0x0C: 'Skills', 0x0D: 'Items',
                0x0E: 'Enemies', 0x0F: 'Troops', 0x10: 'Terrains',
                0x11: 'States', 0x12: 'Animations', 0x13: 'ChipSets',
                0x14: 'Terms', 0x15: 'System', 0x16: 'Switches',
                0x17: 'Variables', 0x18: 'CommonEvents',
            }
            # String field tags within each DB entry (simplified)
            # Actors: 0x01=name, 0x02=title, 0x07=skill_name
            # Items: 0x01=name, 0x02=description
            # Skills: 0x01=name, 0x02=description
            string_tags = {0x01, 0x02, 0x03, 0x07}

            while True:
                try:
                    section_tag = _ber_read(f)
                except EOFError:
                    break

                section_len = _ber_read(f)
                section_end = f.tell() + section_len
                section_name = section_names.get(section_tag, f"section_{section_tag:02x}")

                if section_tag in (0x0B, 0x0C, 0x0D, 0x0E, 0x11):
                    # Array of entries — each entry is: index(BER) + entry_data
                    try:
                        count = _ber_read(f)
                        for _ in range(count):
                            entry_idx = _ber_read(f)
                            entry_len = _ber_read(f)
                            entry_end = f.tell() + entry_len
                            while f.tell() < entry_end:
                                try:
                                    field_tag = _ber_read(f)
                                    field_len = _ber_read(f)
                                    field_data = f.read(field_len)
                                    if field_tag in string_tags and field_data:
                                        try:
                                            text = field_data.decode(encoding, errors='replace')
                                            if _is_translatable(text):
                                                key = f"{section_name}_{entry_idx}_f{field_tag:02x}"
                                                strings[key] = text
                                        except Exception:
                                            pass
                                except EOFError:
                                    break
                    except Exception:
                        f.seek(section_end)
                elif section_tag == 0x14:
                    # Terms section — flat list of strings for menu text
                    try:
                        term_idx = 0
                        pos_start = f.tell()
                        while f.tell() < section_end:
                            try:
                                t_tag = _ber_read(f)
                                t_len = _ber_read(f)
                                t_data = f.read(t_len)
                                if t_data:
                                    try:
                                        text = t_data.decode(encoding, errors='replace')
                                        if _is_translatable(text):
                                            strings[f"Terms_{t_tag}"] = text
                                    except Exception:
                                        pass
                                term_idx += 1
                            except EOFError:
                                break
                    except Exception:
                        f.seek(section_end)
                else:
                    f.seek(section_end)

            if strings:
                all_strings[basename] = strings
                log(f"   📦 {basename}: {len(strings)} strings")

    except Exception as e:
        log(f"⚠️ Failed to parse {basename}: {e}")

    return all_strings


def extract_rm2k3_maps(game_dir: str, log: Callable = print
                       ) -> Dict[str, Dict[str, str]]:
    """Extract translatable strings from RPG Maker 2000/2003 LMU (map) files."""
    all_strings = {}
    map_files = sorted([f for f in os.listdir(game_dir)
                        if re.match(r'^Map\d+\.lmu$', f, re.IGNORECASE)])
    log(f"🗺️ Found {len(map_files)} LMU map files")

    encoding = 'shift_jis'

    for mf in map_files:
        path = os.path.join(game_dir, mf)
        strings = {}
        try:
            with open(path, 'rb') as f:
                header = _read_rm2k_string(f, 'ascii')
                # LMU structure: tagged fields, events section at tag 0x51
                file_size = os.path.getsize(path)
                while f.tell() < file_size:
                    try:
                        tag = _ber_read(f)
                        length = _ber_read(f)
                        chunk_end = f.tell() + length

                        if tag == 0x51:  # Events section
                            try:
                                ev_count = _ber_read(f)
                                for _ in range(ev_count):
                                    ev_id = _ber_read(f)
                                    ev_len = _ber_read(f)
                                    ev_end = f.tell() + ev_len
                                    # Parse event fields
                                    while f.tell() < ev_end:
                                        try:
                                            ft = _ber_read(f)
                                            fl = _ber_read(f)
                                            fd = f.read(fl)
                                            # 0x01=name, 0x05=pages contain commands
                                            if ft == 0x01 and fd:
                                                try:
                                                    text = fd.decode(encoding, errors='replace')
                                                    if _is_translatable(text):
                                                        strings[f"ev_{ev_id}_name"] = text
                                                except Exception:
                                                    pass
                                            # Event commands with text are in pages
                                            # Command strings have codes 10110 (show text)
                                            # For simplicity, extract all non-trivial strings
                                            elif ft in (0x02, 0x03, 0x04, 0x15, 0x16) and fd:
                                                try:
                                                    text = fd.decode(encoding, errors='replace')
                                                    if _is_translatable(text) and len(text) > 2:
                                                        strings[f"ev_{ev_id}_f{ft:02x}"] = text
                                                except Exception:
                                                    pass
                                        except EOFError:
                                            break
                            except Exception:
                                f.seek(chunk_end)
                        else:
                            f.seek(chunk_end)
                    except EOFError:
                        break
        except Exception as e:
            log(f"   ⚠️ Failed to parse {mf}: {e}")
            continue

        if strings:
            all_strings[mf] = strings
            log(f"   📄 {mf}: {len(strings)} strings")

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

    # If backups exist but progress is gone/empty, restore originals first.
    # This prevents re-extracting from half-patched files after a reset.
    backup_dir = os.path.join(game_dir, "GTool_Translation", "originals_backup")
    progress_path = get_progress_path(game_dir)
    progress_exists = os.path.exists(progress_path)
    progress_empty = True
    if progress_exists:
        try:
            with open(progress_path, 'r', encoding='utf-8') as f:
                progress_data = json.load(f)
                progress_empty = len(progress_data) == 0
        except Exception:
            progress_empty = True

    if os.path.isdir(backup_dir) and (not progress_exists or progress_empty):
        backed_files = [f for f in os.listdir(backup_dir)
                        if os.path.isfile(os.path.join(backup_dir, f))]
        if backed_files:
            log(f"🔄 Restoring {len(backed_files)} original files from backup...")
            for fn in backed_files:
                src = os.path.join(backup_dir, fn)
                dst = os.path.join(data_dir, fn)
                shutil.copy2(src, dst)
            log("✅ Originals restored — extracting from clean data")

    if version in (RPGMakerVersion.MV, RPGMakerVersion.MZ):
        db_strings = extract_db_strings(data_dir, log)
        map_strings = extract_map_strings(data_dir, log)
        all_strings = {**db_strings, **map_strings}
    elif version == RPGMakerVersion.VX_ACE:
        log("📦 Parsing Ruby Marshal (.rvdata2) files...")
        all_strings = extract_marshal_data(data_dir, '.rvdata2', log)
    elif version == RPGMakerVersion.VX:
        log("📦 Parsing Ruby Marshal (.rvdata) files...")
        all_strings = extract_marshal_data(data_dir, '.rvdata', log)
    elif version == RPGMakerVersion.XP:
        log("📦 Parsing Ruby Marshal (.rxdata) files...")
        all_strings = extract_marshal_data(data_dir, '.rxdata', log)
    elif version == RPGMakerVersion.RM2K3:
        log("📦 Parsing RPG Maker 2000/2003 binary files...")
        ldb_path = None
        for f in os.listdir(game_dir):
            if f.lower() == 'rpg_rt.ldb':
                ldb_path = os.path.join(game_dir, f)
                break
        all_strings = {}
        if ldb_path:
            all_strings.update(extract_rm2k3_ldb(ldb_path, log))
        all_strings.update(extract_rm2k3_maps(game_dir, log))
    else:
        all_strings = {}

    total = sum(len(v) for v in all_strings.values())
    log(f"📊 Total extractable strings: {total}")
    return version, data_dir, all_strings


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


def _clean_translation(text: str) -> str:
    """Clean up messy AI translation output.

    Handles common LLM quirks per-line:
    - 'original text -> translation' → keep only the translation
    - '"quoted translation"' → strip quotes
    Works on multi-line entries, preserving line count.
    """
    if not text or not text.strip():
        return text.strip()

    lines = text.split('\n')
    cleaned = []
    for line in lines:
        t = line.strip()

        # Strip 'original -> translation' pattern (keep right side)
        arrow_match = re.search(r'\s*->\s*', t)
        if arrow_match:
            right = t[arrow_match.end():].strip()
            if right:
                t = right

        # Strip surrounding quotes
        if len(t) >= 2 and t[0] == '"' and t[-1] == '"':
            t = t[1:-1].strip()

        cleaned.append(t)

    return '\n'.join(cleaned)


def parse_translated_chunk(response: str, chunk: Dict) -> Dict[str, str]:
    """Parse numbered translation response back into key->translation map.

    Handles multi-line entries where each [N] block can span multiple
    lines until the next [N+1] tag or end of response.
    Also cleans up common LLM output quirks (arrows, quotes, references).
    """
    translations = {}
    keys = chunk["keys"]

    # Split on [N] tag boundaries, capturing the number.
    # No ^ anchor — handles AI merging entries on one line (e.g. [1] text[2] text)
    # Safe because \d+ only matches digits, so [Common Equipment] etc. won't match.
    parts = re.split(r'\[(\d+)\]\s*', response.strip())

    if len(parts) > 2:
        # First pass: collect raw translations
        raw = {}
        idx = 1
        while idx < len(parts) - 1:
            try:
                num = int(parts[idx])
                text = parts[idx + 1].strip()
                key_idx = num - 1
                if 0 <= key_idx < len(keys):
                    raw[num] = (key_idx, text)
            except (ValueError, IndexError):
                pass
            idx += 2

        # Second pass: clean and resolve references
        for num, (key_idx, text) in raw.items():
            # Handle 'same as [N]' or 'same as [22]' references
            same_match = re.match(r'(?:same(?:\s+as)?)\s*\[?(\d+)\]?', text, re.IGNORECASE)
            if same_match:
                ref_num = int(same_match.group(1))
                if ref_num in raw:
                    _, ref_text = raw[ref_num]
                    text = _clean_translation(ref_text)
                else:
                    text = _clean_translation(text)
            else:
                text = _clean_translation(text)
            translations[keys[key_idx]] = text
    else:
        # Fallback: split by lines and match 1:1
        flines = [ln.strip() for ln in response.strip().split('\n') if ln.strip()]
        for j, line in enumerate(flines):
            if j < len(keys):
                clean = re.sub(r'^\d+[.)\]]\s*', '', line)
                translations[keys[j]] = _clean_translation(clean)

    return translations


def apply_translations(data_dir: str, trans_map_path: str,
                       log: Callable = print, version: str = None) -> bool:
    """Apply translations from the translation map back to game files.

    For MV/MZ (JSON): patches files directly with backups.
    For VX Ace/VX/XP/2K3 (binary): generates a patch JSON that can be
    used with a game plugin or applied via a separate tool.
    """
    try:
        with open(trans_map_path, 'r', encoding='utf-8') as f:
            trans_data = json.load(f)
    except Exception as e:
        log(f"❌ Failed to load translation map: {e}")
        return False

    backup_dir = os.path.join(os.path.dirname(trans_map_path), "originals_backup")
    os.makedirs(backup_dir, exist_ok=True)

    # For binary formats, we can't patch in-place easily — generate a patch file
    if version and version not in (RPGMakerVersion.MV, RPGMakerVersion.MZ):
        return _apply_binary_format_translations(data_dir, trans_data, trans_map_path, log, version)

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


def _apply_binary_format_translations(data_dir: str, trans_data: dict,
                                      trans_map_path: str, log: Callable,
                                      version: str) -> bool:
    """For binary formats (VX Ace/VX/XP/2K3), attempt in-place Marshal patching
    or generate a readable patch file for manual/plugin use."""
    backup_dir = os.path.join(os.path.dirname(trans_map_path), "originals_backup")
    os.makedirs(backup_dir, exist_ok=True)

    patched_total = 0

    for filename, entries in trans_data.items():
        file_path = os.path.join(data_dir, filename)
        if not os.path.exists(file_path):
            continue

        # Count how many translations we have for this file
        translated_entries = {k: v for k, v in entries.items()
                              if v.get("translated")}
        if not translated_entries:
            continue

        # For Marshal formats, try to load, patch, and re-serialize
        if version in (RPGMakerVersion.VX_ACE, RPGMakerVersion.VX, RPGMakerVersion.XP):
            try:
                # Backup original
                backup_path = os.path.join(backup_dir, filename)
                if not os.path.exists(backup_path):
                    shutil.copy2(file_path, backup_path)

                obj = _marshal_load(file_path)
                file_patched = _patch_marshal_object(obj, translated_entries)

                if file_patched > 0:
                    # Re-serialize using Ruby Marshal format
                    serialized = _marshal_dump(obj)
                    with open(file_path, 'wb') as f:
                        f.write(serialized)
                    patched_total += file_patched
                    log(f"   ✅ {filename}: {file_patched} strings patched (Marshal)")
            except Exception as e:
                log(f"   ⚠️ {filename}: Marshal patch failed ({e}), saving to patch file")
                patched_total += len(translated_entries)
        else:
            # 2K3 binary — too complex for in-place patching, count as patch-file output
            patched_total += len(translated_entries)

    # Always generate a human-readable patch file
    patch_path = os.path.join(os.path.dirname(trans_map_path), "translation_patch.json")
    patch_data = {}
    for filename, entries in trans_data.items():
        file_entries = {}
        for key, entry in entries.items():
            if entry.get("translated"):
                file_entries[key] = {
                    "original": entry["original"],
                    "translated": entry["translated"]
                }
        if file_entries:
            patch_data[filename] = file_entries

    with open(patch_path, 'w', encoding='utf-8') as f:
        json.dump(patch_data, f, ensure_ascii=False, indent=2)

    log(f"💾 Translation patch saved: {patch_path}")
    log(f"🎮 Total: {patched_total} strings processed")
    return True


def _patch_marshal_object(obj, translated_entries: dict, prefix: str = "") -> int:
    """Recursively patch translatable strings in a deserialized Marshal object."""
    if obj is None or not isinstance(obj, (dict, list)):
        return 0

    patched = 0

    if isinstance(obj, dict):
        if obj.get('__userdata__'):
            return 0
        # Direct field patches
        for field in _TRANSLATABLE_FIELDS + ['display_name']:
            for key, entry in translated_entries.items():
                if key.endswith(f"_{field}") and field in obj:
                    if isinstance(obj[field], str) and obj[field] == entry["original"]:
                        obj[field] = entry["translated"]
                        patched += 1
        # Recurse into pages, events, command lists
        for sub_key in ('pages', 'list', 'events'):
            sub = obj.get(sub_key)
            if isinstance(sub, (dict, list)):
                patched += _patch_marshal_object(sub, translated_entries, prefix)
    elif isinstance(obj, list):
        for item in obj:
            if isinstance(item, (dict, list)):
                patched += _patch_marshal_object(item, translated_entries, prefix)

    return patched


# ============================================================
# Ruby Marshal writer (minimal) — for writing patched VX Ace/VX/XP data
# ============================================================

def _marshal_dump(obj) -> bytes:
    """Serialize a Python object back to Ruby Marshal 4.8 format."""
    buf = io.BytesIO()
    writer = RubyMarshalWriter(buf)
    buf.write(b'\x04\x08')  # Marshal magic
    writer.write_value(obj)
    return buf.getvalue()


class RubyMarshalWriter:
    """Minimal Ruby Marshal writer for re-serializing patched RPG Maker data."""

    def __init__(self, buf: io.BytesIO):
        self._buf = buf
        self._symbols = {}
        self._objects = {}

    def _write_byte(self, b: int):
        self._buf.write(bytes([b & 0xFF]))

    def _write_fixnum(self, n: int):
        if n == 0:
            self._buf.write(b'\x00')
        elif 0 < n < 123:
            self._write_byte(n + 5)
        elif -128 < n < 0:
            self._write_byte(n - 5 & 0xFF)
        elif n > 0:
            # Positive multi-byte
            bs = []
            tmp = n
            while tmp > 0:
                bs.append(tmp & 0xFF)
                tmp >>= 8
            self._write_byte(len(bs))
            for b in bs:
                self._write_byte(b)
        else:
            # Negative multi-byte
            bs = []
            tmp = -n - 1
            for i in range(4):
                bs.append(~(tmp >> (8 * i)) & 0xFF)
                if tmp >> (8 * (i + 1)) == 0:
                    break
            self._write_byte(-len(bs) & 0xFF)
            for b in bs:
                self._write_byte(b)

    def _write_raw_string(self, s: bytes):
        self._write_fixnum(len(s))
        self._buf.write(s)

    def _write_symbol(self, name: str):
        if name in self._symbols:
            self._write_byte(ord(';'))
            self._write_fixnum(self._symbols[name])
        else:
            idx = len(self._symbols)
            self._symbols[name] = idx
            self._write_byte(ord(':'))
            self._write_raw_string(name.encode('utf-8'))

    def write_value(self, obj):
        if obj is None:
            self._write_byte(ord('0'))
        elif obj is True:
            self._write_byte(ord('T'))
        elif obj is False:
            self._write_byte(ord('F'))
        elif isinstance(obj, int):
            self._write_byte(ord('i'))
            self._write_fixnum(obj)
        elif isinstance(obj, float):
            self._write_byte(ord('f'))
            s = repr(obj).encode('ascii')
            self._write_raw_string(s)
        elif isinstance(obj, str):
            # Instance string with UTF-8 encoding
            self._write_byte(ord('I'))
            self._write_byte(ord('"'))
            encoded = obj.encode('utf-8')
            self._write_raw_string(encoded)
            # 1 ivar: :E => true (UTF-8)
            self._write_fixnum(1)
            self._write_symbol('E')
            self._write_byte(ord('T'))
        elif isinstance(obj, bytes):
            self._write_byte(ord('"'))
            self._write_raw_string(obj)
        elif isinstance(obj, list):
            self._write_byte(ord('['))
            self._write_fixnum(len(obj))
            for item in obj:
                self.write_value(item)
        elif isinstance(obj, dict):
            cls = obj.get('__class__')
            if cls and not obj.get('__userdata__'):
                # Object with class
                self._write_byte(ord('o'))
                self._write_symbol(cls)
                # Count non-meta keys
                real_keys = [k for k in obj if k not in ('__class__', '__userdata__')]
                self._write_fixnum(len(real_keys))
                for k in real_keys:
                    self._write_symbol(f'@{k}')
                    self.write_value(obj[k])
            elif obj.get('__userdata__'):
                # User-defined type — write as nil (data was not modified)
                self._write_byte(ord('0'))
            elif isinstance(obj, tuple) and len(obj) == 2 and obj[0] == '__sym__':
                self._write_symbol(obj[1])
            else:
                # Regular hash
                self._write_byte(ord('{'))
                self._write_fixnum(len(obj))
                for k, v in obj.items():
                    if isinstance(k, str):
                        self._write_symbol(k)
                    else:
                        self.write_value(k)
                    self.write_value(v)
        elif isinstance(obj, tuple) and len(obj) == 2 and obj[0] == '__sym__':
            self._write_symbol(obj[1])


def _wrap_to_lines(text: str, num_lines: int) -> list:
    """Distribute text evenly across num_lines, splitting at word boundaries.

    RPG Maker dialog boxes have a fixed number of lines per message command.
    If the AI returns a single-line translation for a multi-line original,
    this splits it into the required number of lines.
    """
    words = text.split()
    if not words or num_lines <= 1:
        return [text] if num_lines == 1 else [text] + [""] * (num_lines - 1)

    # Target roughly equal character count per line
    total_len = sum(len(w) for w in words) + len(words) - 1
    target_per_line = total_len / num_lines

    lines = []
    current_line = []
    current_len = 0

    for word in words:
        word_len = len(word) + (1 if current_line else 0)
        if current_line and current_len + word_len > target_per_line and len(lines) < num_lines - 1:
            lines.append(" ".join(current_line))
            current_line = [word]
            current_len = len(word)
        else:
            current_line.append(word)
            current_len += word_len

    if current_line:
        lines.append(" ".join(current_line))

    # Pad if we somehow got fewer lines
    while len(lines) < num_lines:
        lines.append("")

    return lines


def _patch_map_entry(data: dict, key: str, translated: str) -> int:
    """Patch a single map entry with translated text."""
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

    # Speaker name: event_E_page_P_speaker_I
    m = re.match(r'event_(\d+)_page_(\d+)_speaker_(\d+)', key)
    if m:
        ev_idx = int(m.group(1))
        pg_idx = int(m.group(2))
        cmd_idx = int(m.group(3))
        try:
            cmd = data["events"][ev_idx]["pages"][pg_idx]["list"][cmd_idx]
            if cmd.get("code", 0) == 101 and len(cmd.get("parameters", [])) > 4:
                cmd["parameters"][4] = translated
                return 1
        except (IndexError, KeyError, TypeError):
            pass

    # Grouped message block: event_E_page_P_msg_S_E
    m = re.match(r'event_(\d+)_page_(\d+)_msg_(\d+)_(\d+)', key)
    if m:
        ev_idx = int(m.group(1))
        pg_idx = int(m.group(2))
        cmd_start = int(m.group(3))
        cmd_end = int(m.group(4))
        try:
            cmd_list = data["events"][ev_idx]["pages"][pg_idx]["list"]
            num_lines = cmd_end - cmd_start + 1
            # Split translated text into the same number of lines
            trans_lines = translated.split("\n")
            if len(trans_lines) < num_lines and num_lines > 1:
                # AI returned fewer lines — word-wrap to fill original line count
                full_text = " ".join(ln.strip() for ln in trans_lines if ln.strip())
                trans_lines = _wrap_to_lines(full_text, num_lines)
            elif len(trans_lines) > num_lines:
                # Merge excess lines into the last slot
                trans_lines = trans_lines[:num_lines - 1] + [" ".join(trans_lines[num_lines - 1:])]
            for offset in range(num_lines):
                ci = cmd_start + offset
                if ci < len(cmd_list):
                    cmd = cmd_list[ci]
                    if isinstance(cmd, dict) and cmd.get("code", 0) in _DIALOG_CODES:
                        cmd["parameters"][0] = trans_lines[offset]
            return 1
        except (IndexError, KeyError, TypeError):
            pass

    # Choice: event_E_page_P_choice_I_C
    m = re.match(r'event_(\d+)_page_(\d+)_choice_(\d+)_(\d+)', key)
    if m:
        ev_idx = int(m.group(1))
        pg_idx = int(m.group(2))
        cmd_idx = int(m.group(3))
        choice_idx = int(m.group(4))
        try:
            cmd = data["events"][ev_idx]["pages"][pg_idx]["list"][cmd_idx]
            if cmd.get("code", 0) == _CHOICE_CODE:
                choices = cmd["parameters"][0]
                if isinstance(choices, list) and choice_idx < len(choices):
                    choices[choice_idx] = translated
                    return 1
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

    # Speaker name: ce_E_speaker_I
    m = re.match(r'ce_(\d+)_speaker_(\d+)', key)
    if m:
        ev_idx = int(m.group(1))
        cmd_idx = int(m.group(2))
        try:
            cmd = data[ev_idx]["list"][cmd_idx]
            if cmd.get("code", 0) == 101 and len(cmd.get("parameters", [])) > 4:
                cmd["parameters"][4] = translated
                return 1
        except (IndexError, KeyError, TypeError):
            pass

    # Grouped message block: ce_E_msg_S_E
    m = re.match(r'ce_(\d+)_msg_(\d+)_(\d+)', key)
    if m:
        ev_idx = int(m.group(1))
        cmd_start = int(m.group(2))
        cmd_end = int(m.group(3))
        try:
            cmd_list = data[ev_idx]["list"]
            num_lines = cmd_end - cmd_start + 1
            trans_lines = translated.split("\n")
            if len(trans_lines) < num_lines and num_lines > 1:
                full_text = " ".join(ln.strip() for ln in trans_lines if ln.strip())
                trans_lines = _wrap_to_lines(full_text, num_lines)
            elif len(trans_lines) > num_lines:
                trans_lines = trans_lines[:num_lines - 1] + [" ".join(trans_lines[num_lines - 1:])]
            for offset in range(num_lines):
                ci = cmd_start + offset
                if ci < len(cmd_list):
                    cmd = cmd_list[ci]
                    if isinstance(cmd, dict) and cmd.get("code", 0) in _DIALOG_CODES:
                        cmd["parameters"][0] = trans_lines[offset]
            return 1
        except (IndexError, KeyError, TypeError):
            pass

    # Choice: ce_E_choice_I_C
    m = re.match(r'ce_(\d+)_choice_(\d+)_(\d+)', key)
    if m:
        ev_idx = int(m.group(1))
        cmd_idx = int(m.group(2))
        choice_idx = int(m.group(3))
        try:
            cmd = data[ev_idx]["list"][cmd_idx]
            if cmd.get("code", 0) == _CHOICE_CODE:
                choices = cmd["parameters"][0]
                if isinstance(choices, list) and choice_idx < len(choices):
                    choices[choice_idx] = translated
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
