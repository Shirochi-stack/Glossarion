#!/usr/bin/env python3
"""
Glossarion Discord Bot
Translate files via Discord using your existing Glossarion installation

PDF Formatting Integration:
- When processing PDF files, the bot automatically uses the pdf_extractor module
- The pdf_extractor.generate_css_from_pdf() function detects and extracts:
  * base_font_size: The median font size from the PDF body text
  * font_family: The most common font family (mapped to web-safe fonts)
  * text_align: The predominant text alignment (left, center, right, justify)
  * line_height_ratio: The calculated line spacing ratio
- These variables are automatically applied during PDF -> HTML conversion
- No manual configuration needed - styling is preserved from the original PDF
"""

import discord
from discord import app_commands
from discord.ext import commands
import os
import sys
import asyncio
import tempfile
import shutil
import json
import logging
import time
import uuid
import functools
from urllib.parse import quote as urlquote
from typing import Optional

import requests
from aiohttp import web

# Add src directory to path
# In this repo layout, `discord_bot.py` typically lives inside the `src/` directory.
# Older deployments may have a nested `src/src` structure, so detect the correct one.
_base_dir = os.path.dirname(__file__)
_nested_src_dir = os.path.join(_base_dir, "src")

if os.path.isdir(_nested_src_dir) and os.path.exists(os.path.join(_nested_src_dir, "config.json")):
    src_dir = _nested_src_dir
else:
    src_dir = _base_dir

sys.path.insert(0, src_dir)

PROJECT_ROOT = os.path.dirname(src_dir)
HOSTED_FILES_DIR = os.path.join(PROJECT_ROOT, "hosted_files")
HOSTED_TTL_SECONDS = 24 * 60 * 60  # 24 hours
HOSTED_MAX_BYTES = 5 * 1024 * 1024 * 1024  # 5 GB
MAX_DISCORD_UPLOAD = 10 * 1024 * 1024  # 10 MB
FILE_HOST_PORT = int(os.getenv("FILE_HOST_PORT", "8080"))
FILE_HOST_BASE_URL = (os.getenv("FILE_HOST_BASE_URL") or f"http://localhost:{FILE_HOST_PORT}").rstrip("/")
GOFILE_TOKEN = os.getenv("GOFILE_TOKEN")
ORACLE_PAR_BASE = os.getenv("ORACLE_PAR_BASE", "")  # PAR base ending with /o/
USER_CONFIG_DIR = os.path.join(PROJECT_ROOT, "bot_user_configs")

_file_host_runner: Optional[web.AppRunner] = None
_file_host_site: Optional[web.TCPSite] = None
_file_host_started = False


def _safe_filename(name: str) -> str:
    name = os.path.basename(name or "file")
    keep = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_."
    cleaned = ''.join(ch if ch in keep else '_' for ch in name)
    return cleaned or "file"


def _cleanup_hosted_files() -> int:
    """Remove expired files and enforce the 5GB cap. Returns current total bytes."""
    os.makedirs(HOSTED_FILES_DIR, exist_ok=True)
    now = time.time()
    entries = []  # (mtime, size, path)
    total = 0
    for fname in list(os.listdir(HOSTED_FILES_DIR)):
        path = os.path.join(HOSTED_FILES_DIR, fname)
        try:
            st = os.stat(path)
        except FileNotFoundError:
            continue
        if not os.path.isfile(path):
            continue
        # Drop empty files proactively
        if st.st_size <= 0 or now - st.st_mtime > HOSTED_TTL_SECONDS:
            try:
                os.remove(path)
            except Exception:
                pass
            continue
        total += st.st_size
        entries.append((st.st_mtime, st.st_size, path))

    # Enforce quota by deleting oldest first
    if total > HOSTED_MAX_BYTES:
        entries.sort(key=lambda t: t[0])  # oldest first
        for mtime, size, path in entries:
            if total <= HOSTED_MAX_BYTES:
                break
            try:
                os.remove(path)
                total -= size
            except Exception:
                continue
    return total


def _hosted_download_url(stored_name: str) -> str:
    return f"{FILE_HOST_BASE_URL}/files/{urlquote(stored_name)}"


def _ensure_space(effective_bytes: int) -> None:
    total = _cleanup_hosted_files()
    if effective_bytes > HOSTED_MAX_BYTES:
        raise ValueError("File exceeds 5GB host capacity")
    if total + effective_bytes <= HOSTED_MAX_BYTES:
        return

    # Delete oldest files until there is room
    entries = []
    for fname in list(os.listdir(HOSTED_FILES_DIR)):
        path = os.path.join(HOSTED_FILES_DIR, fname)
        try:
            st = os.stat(path)
        except Exception:
            continue
        if not os.path.isfile(path):
            continue
        entries.append((st.st_mtime, st.st_size, path))
    entries.sort(key=lambda t: t[0])
    for mtime, size, path in entries:
        if total + effective_bytes <= HOSTED_MAX_BYTES:
            break
        try:
            os.remove(path)
            total -= size
        except Exception:
            continue
    if total + effective_bytes > HOSTED_MAX_BYTES:
        raise ValueError("Insufficient space in hosted storage (5GB cap)")


def _store_hosted_file(file_path: str, display_name: str, *, send_zip: bool) -> str:
    file_size = os.path.getsize(file_path)
    effective_size = file_size * (2 if send_zip else 1)
    _ensure_space(effective_size)

    safe_name = _safe_filename(display_name)
    stored_name = f"{int(time.time())}_{uuid.uuid4().hex[:8]}_{safe_name}"
    dest_path = os.path.join(HOSTED_FILES_DIR, stored_name)
    shutil.copy2(file_path, dest_path)
    return stored_name


async def _ensure_file_host_server():
    global _file_host_runner, _file_host_site, _file_host_started
    if _file_host_started:
        return

    async def serve_file(request: web.Request):
        fname = request.match_info.get("name", "")
        # Prevent path traversal
        safe = _safe_filename(fname)
        if safe != fname:
            raise web.HTTPNotFound()
        path = os.path.join(HOSTED_FILES_DIR, fname)
        if not os.path.isfile(path):
            raise web.HTTPNotFound()
        return web.FileResponse(path)

    os.makedirs(HOSTED_FILES_DIR, exist_ok=True)
    app = web.Application()
    app.add_routes([web.get('/files/{name}', serve_file)])

    _file_host_runner = web.AppRunner(app)
    await _file_host_runner.setup()
    _file_host_site = web.TCPSite(_file_host_runner, host='0.0.0.0', port=FILE_HOST_PORT)
    await _file_host_site.start()
    _file_host_started = True
    sys.stderr.write(f"[FILE HOST] Serving {HOSTED_FILES_DIR} at {FILE_HOST_BASE_URL}/files/<name>\n")
    sys.stderr.flush()


def _effective_discord_size(bytes_len: int, send_zip: bool) -> int:
    return int(bytes_len * (2 if send_zip else 1))

def _should_offload(bytes_len: int, send_zip: bool) -> bool:
    return _effective_discord_size(bytes_len, send_zip) > MAX_DISCORD_UPLOAD


def _offload_path_blocking(file_path: str, display_name: str, *, send_zip: bool) -> str:
    stored_name = _store_hosted_file(file_path, display_name, send_zip=send_zip)
    return _hosted_download_url(stored_name)


def _upload_to_oracle_par(path: str, display_name: str | None) -> str:
    base = ORACLE_PAR_BASE
    if not base:
        raise RuntimeError("ORACLE_PAR_BASE not set")
    base = base.rstrip('/') + '/'
    fname = display_name or os.path.basename(path)
    url = base + requests.utils.quote(fname)
    with open(path, "rb") as f:
        resp = requests.put(url, data=f, headers={"Content-Type": "application/octet-stream"}, timeout=180)
    if resp.status_code not in (200, 201, 204):
        raise RuntimeError(f"Oracle upload failed HTTP {resp.status_code}: {resp.text[:200]}")
    return url


def _upload_to_tmpfiles(path: str, display_name: str | None) -> str:
    fname = display_name or os.path.basename(path)
    with open(path, "rb") as f:
        files = {"file": (fname, f, "application/octet-stream")}
        resp = requests.post("https://tmpfiles.org/api/v1/upload", files=files, timeout=60)
    if resp.status_code != 200:
        raise RuntimeError(f"tmpfiles HTTP {resp.status_code}")
    try:
        data = resp.json()
    except Exception:
        data = {}
    link = data.get("data", {}).get("url") if isinstance(data.get("data"), dict) else None
    if not link:
        text = resp.text.strip()
        if text.startswith("http"):
            link = text
    if not link:
        raise RuntimeError(f"tmpfiles response invalid: {resp.text[:200]}")
    return link


def _upload_to_tempsh(path: str, display_name: str | None) -> str:
    fname = display_name or os.path.basename(path)
    with open(path, "rb") as f:
        files = {"file": (fname, f, "application/octet-stream")}
        resp = requests.post("https://temp.sh/upload", files=files, timeout=60)
    if resp.status_code != 200:
        raise RuntimeError(f"temp.sh HTTP {resp.status_code}")
    link = resp.text.strip()
    if not link.startswith("http"):
        raise RuntimeError(f"temp.sh response invalid: {link[:200]}")
    return link


def _upload_to_gofile(path: str, display_name: str | None) -> str:
    """Upload file to Gofile and return a direct link."""
    server_resp = requests.get("https://api.gofile.io/getServer", timeout=15)
    server_json = server_resp.json()
    server = server_json.get("data", {}).get("server")
    if not server:
        raise RuntimeError("Failed to get Gofile server")

    name = display_name or os.path.basename(path)
    with open(path, "rb") as f:
        files = {"file": (name, f, "application/octet-stream")}
        headers = {"Authorization": f"Bearer {GOFILE_TOKEN}"} if GOFILE_TOKEN else {}
        upload_resp = requests.post(f"https://{server}.gofile.io/uploadFile",
                                    files=files,
                                    headers=headers,
                                    timeout=300)
        data = upload_resp.json()
    if upload_resp.status_code != 200 or data.get("status") != "ok":
        raise RuntimeError(f"Gofile upload failed: {data}")
    link = data.get("data", {}).get("directLink") or data.get("data", {}).get("downloadPage")
    if not link:
        raise RuntimeError("Gofile response missing link")
    return link


def _pick_bot_tmpdir() -> str:
    """Pick a temp directory that exists and is writable.

    Some hosts run with TMPDIR/TEMP pointing at a directory that can disappear
    (e.g. cleaned up between runs). tempfile.mkdtemp() will then raise
    FileNotFoundError: [Errno 2] No such file or directory.

    We prefer a stable project-local directory to avoid that class of failure.
    """
    candidates = []

    # User override
    override = (os.getenv("GLOSSARION_BOT_TMPDIR") or "").strip()
    if override:
        candidates.append(override)

    # Project-local temp dir (most stable)
    try:
        project_root = os.path.dirname(src_dir)
        candidates.append(os.path.join(project_root, "bot_tmp"))
    except Exception:
        pass

    # System temp fallbacks
    for k in ("TMPDIR", "TEMP", "TMP"):
        v = (os.getenv(k) or "").strip()
        if v:
            candidates.append(v)

    # Common unix temp dirs
    candidates.extend(["/tmp", "/var/tmp"])

    # Last resort: current working directory
    try:
        candidates.append(os.getcwd())
    except Exception:
        pass

    for d in candidates:
        try:
            os.makedirs(d, exist_ok=True)
            if os.path.isdir(d) and os.access(d, os.W_OK):
                return d
        except Exception:
            continue

    # If everything fails, let tempfile decide (may still error, but nothing else we can do here)
    return ""


BOT_TMPDIR = _pick_bot_tmpdir()
if BOT_TMPDIR:
    sys.stderr.write(f"[CONFIG] Bot temp dir: {BOT_TMPDIR}\n")
    sys.stderr.flush()

    # Ensure all stdlib tempfile users (and many third-party libs) use the stable temp root.
    try:
        tempfile.tempdir = BOT_TMPDIR
    except Exception:
        pass

def _ensure_stable_cwd() -> None:
    """Ensure process CWD exists and tempfile has a stable base directory.

    On some hosts, TMPDIR can point at a directory that gets cleaned up, and the process CWD can
    also become invalid if something chdir()'d into a temp dir that later disappears.

    When that happens, unrelated code can start failing with:
      FileNotFoundError: [Errno 2] No such file or directory

    We recover by:
    - chdir() to the project root if CWD is missing
    - proactively avoiding running inside volatile temp directories
    - re-asserting TMPDIR + tempfile.tempdir to a stable directory
    """

    project_root = os.path.dirname(src_dir)

    def _is_within(child: str, parent: str) -> bool:
        try:
            child_abs = os.path.abspath(child)
            parent_abs = os.path.abspath(parent)
            return os.path.commonpath([child_abs, parent_abs]) == parent_abs
        except Exception:
            return False

    # Ensure CWD exists
    try:
        cwd = os.getcwd()
    except FileNotFoundError:
        try:
            os.chdir(project_root)
            cwd = os.getcwd()
        except Exception:
            cwd = ""

    # Even if CWD exists, don't run inside a temp root (process-wide chdir() is a footgun).
    try:
        if cwd and BOT_TMPDIR and _is_within(cwd, BOT_TMPDIR):
            os.chdir(project_root)
    except Exception:
        pass

    # Re-assert stable tempdir (helps libraries that call tempfile.gettempdir())
    try:
        if BOT_TMPDIR:
            os.makedirs(BOT_TMPDIR, exist_ok=True)
            os.environ['TMPDIR'] = BOT_TMPDIR
            tempfile.tempdir = BOT_TMPDIR
    except Exception:
        pass


# Ensure our process isn't running with a deleted working directory at startup.
_ensure_stable_cwd()

# Import Glossarion modules
try:
    # Core translation modules
    import TransateKRtoEN
    import extract_glossary_from_epub
    import extract_glossary_from_txt
    from model_options import get_model_options
    from api_key_encryption import decrypt_config
    
    # File processing modules
    import pdf_extractor
    import epub_converter
    import enhanced_text_extractor
    import txt_processor
    
    # Glossary management
    import GlossaryManager
    import glossary_compressor
    
    # Chapter and text processing
    import chapter_splitter
    import Chapter_Extractor
    import chapter_extraction_manager
    
    # API and client modules
    import unified_api_client
    try:
        import async_api_processor
    except ImportError:
        async_api_processor = None
    import multi_api_key_manager
    
    # Utility modules
    import history_manager
    try:
        import metadata_batch_translator
    except ImportError:
        metadata_batch_translator = None
    import google_free_translate
    
    # Duplicate detection
    import advanced_duplicate_detection
    import duplicate_detection_config
    
    # Image translation (may not be used in Discord but import for completeness)
    try:
        import image_translator
    except ImportError:
        image_translator = None
    try:
        import manga_translator
    except ImportError:
        manga_translator = None
    try:
        import manga_integration
    except ImportError:
        manga_integration = None
    
    # Don't import GUI modules - they require Qt/PySide6
    # (translator_gui, GlossaryManager_GUI, QA_Scanner_GUI, etc.)
    
    GLOSSARION_AVAILABLE = True
    glossary_main = extract_glossary_from_epub.main
except ImportError as e:
    GLOSSARION_AVAILABLE = False
    glossary_main = None
    print(f"⚠️ Glossarion modules not available: {e}")
    def decrypt_config(c):
        return c

# Config file
CONFIG_FILE = os.path.join(src_dir, "config.json")

# Bot setup
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="/", intents=intents)


class _SuppressExpectedAutocompleteErrors(logging.Filter):
    """Suppress noisy, expected autocomplete failures.

    Discord autocomplete interactions are short-lived and are frequently cancelled/expired
    while a user types. When that happens, responding with autocomplete choices can raise:
      - 10062 Unknown interaction
      - 40060 Interaction has already been acknowledged

    These are not actionable for us and just spam logs.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            if record.name != "discord.app_commands.tree":
                return True

            # Only filter the specific "Ignoring exception in autocomplete" log lines.
            msg = record.getMessage() or ""
            if "Ignoring exception in autocomplete" not in msg:
                return True

            exc = None
            if record.exc_info and len(record.exc_info) >= 2:
                exc = record.exc_info[1]

            if isinstance(exc, discord.NotFound) and getattr(exc, "code", None) == 10062:
                return False

            if isinstance(exc, discord.HTTPException) and getattr(exc, "code", None) == 40060:
                return False

            return True
        except Exception:
            # Never break logging.
            return True


# Install the filter early so it applies as soon as CommandTree logs anything.
logging.getLogger("discord.app_commands.tree").addFilter(_SuppressExpectedAutocompleteErrors())

# Global storage for translation state
translation_states = {}

# ---- Per-user config helpers ----
def _user_config_path(user_id: int) -> str:
    return os.path.join(USER_CONFIG_DIR, f"{user_id}.json")


def load_user_config(user_id: int) -> dict:
    try:
        path = _user_config_path(user_id)
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_user_config(user_id: int, payload: dict) -> None:
    try:
        os.makedirs(USER_CONFIG_DIR, exist_ok=True)
        cfg = load_user_config(user_id)
        cfg.update(payload)
        with open(_user_config_path(user_id), "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
    except Exception as e:
        sys.stderr.write(f"[CONFIG] Failed to save user config for {user_id}: {e}\n")
        sys.stderr.flush()


def _ephemeral(interaction: discord.Interaction) -> bool:
    """Use ephemeral responses in guilds; in DMs, send normal messages."""
    return interaction.guild is not None


async def _safe_defer(interaction: discord.Interaction, *, ephemeral: bool) -> bool:
    """Acknowledge a slash-command interaction ASAP.

    Discord requires an initial acknowledgement quickly (or the interaction expires).
    We defer immediately, then we can safely edit the original response.

    Returns True if we successfully acknowledged, False if the interaction is already gone.
    """
    try:
        if interaction.response.is_done():
            return True
        await interaction.response.defer(ephemeral=ephemeral)
        return True
    except discord.NotFound as e:
        if getattr(e, "code", None) == 10062:
            return False
        raise
    except discord.HTTPException as e:
        if getattr(e, "code", None) in (10062, 40060):
            return False
        raise


async def _safe_edit_original_response(
    interaction: discord.Interaction,
    *,
    content: Optional[str] = None,
    embed: Optional[discord.Embed] = None,
    view: Optional[discord.ui.View] = None,
):
    try:
        return await interaction.edit_original_response(content=content, embed=embed, view=view)
    except discord.NotFound as e:
        if getattr(e, "code", None) == 10062:
            return None
        raise
    except discord.HTTPException as e:
        if getattr(e, "code", None) in (10062, 40060):
            return None
        raise


async def _safe_send_message(
    interaction: discord.Interaction,
    content: Optional[str] = None,
    *,
    embed: Optional[discord.Embed] = None,
    ephemeral: bool = False,
    **kwargs,
):
    """Send a message for an interaction without crashing on common interaction races.

    NOTE: discord.NotFound (10062) is a subclass of discord.HTTPException, so we must
    handle it *before* the generic HTTPException handler.

    Returns a discord.Message when possible, otherwise None.
    """
    try:
        if interaction.response.is_done():
            return await interaction.followup.send(
                content=content,
                embed=embed,
                ephemeral=ephemeral,
                wait=True,
                **kwargs,
            )

        await interaction.response.send_message(
            content=content,
            embed=embed,
            ephemeral=ephemeral,
            **kwargs,
        )
        try:
            return await interaction.original_response()
        except Exception:
            return None

    except discord.NotFound as e:
        # Interaction expired/cancelled (common under load).
        if getattr(e, "code", None) == 10062:
            return None
        raise

    except discord.HTTPException as e:
        # If the interaction was already acknowledged, fall back to followup.
        if getattr(e, "code", None) == 40060:
            try:
                return await interaction.followup.send(
                    content=content,
                    embed=embed,
                    ephemeral=ephemeral,
                    wait=True,
                    **kwargs,
                )
            except Exception:
                return None

        # Some platforms return 10062 as HTTPException; treat it as non-fatal.
        if getattr(e, "code", None) == 10062:
            return None

        raise


class LogView(discord.ui.View):
    """View with buttons to toggle log display and stop translation"""
    def __init__(self, user_id: int):
        super().__init__(timeout=None)  # No timeout for persistent view
        self.user_id = user_id
    
    @discord.ui.button(label="Show More Logs", style=discord.ButtonStyle.secondary, emoji="🔽", custom_id="toggle_logs")
    async def toggle_logs(self, interaction: discord.Interaction, button: discord.ui.Button):
        """Toggle between compact and full log view"""
        state = translation_states.get(self.user_id)
        if not state:
            await _safe_send_message(interaction, "❌ Translation session expired", ephemeral=_ephemeral(interaction))
            return
        
        try:
            # Toggle the state
            state['show_full'] = not state.get('show_full', False)
            
            # Update button label
            if state['show_full']:
                button.label = "Show Less"
                button.emoji = "🔼"
            else:
                button.label = "Show More Logs"
                button.emoji = "🔽"
            
            # Get log text based on current state
            logs = state.get('logs', [])
            if state['show_full']:
                log_text = '\n'.join(logs)
                if len(log_text) > 3900:
                    log_text = "..." + log_text[-3900:]
            else:
                log_text = '\n'.join(logs[-10:])
                if len(log_text) > 800:
                    log_text = log_text[-800:]
            
            if not log_text:
                log_text = "No logs yet..."
            
            embed = discord.Embed(
                title="📚 Translation in Progress",
                description=f"**Status:** Processing... ({len(logs)} logs)\n\n```{log_text}```",
                color=discord.Color.blue()
            )
            
            await interaction.response.edit_message(embed=embed, view=self)
        except Exception as e:
            sys.stderr.write(f"[BUTTON ERROR] {e}\n")
            try:
                await _safe_send_message(interaction, f"❌ Error: {e}", ephemeral=_ephemeral(interaction))
            except:
                pass
    
    @discord.ui.button(label="Stop Translation", style=discord.ButtonStyle.danger, emoji="⏹️", custom_id="stop_translation")
    async def stop_translation(self, interaction: discord.Interaction, button: discord.ui.Button):
        """Stop the translation process"""
        state = translation_states.get(self.user_id)
        if not state:
            await _safe_send_message(interaction, "❌ Translation session expired", ephemeral=_ephemeral(interaction))
            return
        
        try:
            state['stop_requested'] = True
            button.disabled = True
            button.label = "Stopped"
            button.style = discord.ButtonStyle.secondary
            
            await interaction.response.edit_message(view=self)
            await interaction.followup.send("⏹️ Translation stop requested...", ephemeral=_ephemeral(interaction))
        except Exception as e:
            sys.stderr.write(f"[BUTTON ERROR] {e}\n")
            try:
                await _safe_send_message(interaction, f"❌ Error: {e}", ephemeral=_ephemeral(interaction))
            except:
                pass


def load_config():
    """Load Glossarion config (decrypted)"""
    try:
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            config = json.load(f)
            return decrypt_config(config)
    except:
        return {}


@bot.event
async def on_ready():
    print(f"✅ {bot.user} is online!")
    try:
        synced = await bot.tree.sync()
        print(f"✅ Synced {len(synced)} command(s)")
    except Exception as e:
        print(f"❌ Failed to sync commands: {e}")


async def model_autocomplete(interaction: discord.Interaction, current: str):
    """Autocomplete for model selection - shows popular models from model_options.py"""
    if GLOSSARION_AVAILABLE:
        try:
            all_models = get_model_options()
            # Filter models that match current input
            if current:
                matches = [m for m in all_models if current.lower() in m.lower()]
            else:
                # Show popular models first when no input
                popular = ['gpt-4', 'gpt-4-turbo', 'gpt-4o', 'claude-3-5-sonnet', 'claude-3-opus', 
                          'gemini-2.0-flash-exp', 'gemini-1.5-pro', 'deepseek-chat']
                matches = [m for m in popular if m in all_models] + all_models[:15]
            
            # Return up to 25 choices (Discord limit)
            return [app_commands.Choice(name=m, value=m) for m in matches[:25]]
        except:
            pass
    
    # Fallback choices if model_options unavailable
    return [
        app_commands.Choice(name="gpt-4", value="gpt-4"),
        app_commands.Choice(name="gpt-4-turbo", value="gpt-4-turbo"),
        app_commands.Choice(name="claude-3-5-sonnet", value="claude-3-5-sonnet"),
        app_commands.Choice(name="gemini-2.0-flash-exp", value="gemini-2.0-flash-exp"),
    ]


async def profile_autocomplete(interaction: discord.Interaction, current: str):
    """Autocomplete prompt profiles defined in translator_gui (config prompt_profiles)."""
    try:
        cfg = load_config()
        profiles = list((cfg.get("prompt_profiles") or {}).keys())
        if "Universal" not in profiles:
            profiles.insert(0, "Universal")
        if current:
            profiles = [p for p in profiles if current.lower() in p.lower()]
        return [app_commands.Choice(name=p, value=p) for p in profiles[:25]]
    except Exception:
        return [app_commands.Choice(name="Universal", value="Universal")]


@bot.tree.command(name="translate", description="Translate EPUB, TXT, or PDF file")
@app_commands.describe(
    api_key="Your API key",
    model="AI model to use (optional; defaults to config.json if omitted)",
    file="EPUB, TXT, or PDF file to translate (optional if using url)",
    url="Google Drive or Dropbox link to file (optional if using file attachment)",
    custom_endpoint_url="Custom OpenAI-compatible base URL (auto-enables when set; omit to disable)",
    google_credentials_file="Google Cloud credentials JSON file upload (for Vertex AI models)",
    extraction_mode="Text extraction method (default: Enhanced/html2text)",
    temperature="Translation temperature 0.0-1.0 (default: 0.3)",
    batch_size="Paragraphs per batch (default: 10)",
    max_output_tokens="Max output tokens (default: 65536)",
    disable_smart_filter="Disable smart glossary filter (default: False)",
    duplicate_algorithm="Duplicate handling: auto/strict/balanced/aggressive/basic (default: balanced)",
    manual_glossary="Manual glossary file (.csv or .json) to upload and use instead of auto-generated",
    enable_auto_glossary="Enable automatic glossary generation (default: True)",
    request_merge_count="Chapters per request (set >=2 to enable request merging; <=1 disables; omit to disable)",
    split_the_merge="Split merged translation output back into separate files (default: True)",
    send_zip="Return output as a ZIP archive instead of individual file (default: False)",
    compression_factor="Compression factor (1.0-3.0; overrides auto-compression if set)",
    thinking="Enable/disable AI thinking capabilities (GPT/Gemini/DeepSeek) - Default: True",
    gemini_thinking_level="Gemini 3 thinking level (low/high) - Default: high",
    gemini_thinking_budget="Gemini thinking budget (-1=auto, 0=disabled) - Default: -1",
    or_thinking_tokens="OpenRouter thinking tokens - Default: 2000",
    gpt_effort="GPT-5/OpenAI thinking effort (none/low/medium/high/xhigh) - Default: medium",
    target_language="Target language",
    profile="Prompt profile from translator_gui (default: Universal)"
)
@app_commands.choices(extraction_mode=[
    app_commands.Choice(name="Enhanced (html2text)", value="enhanced"),
    app_commands.Choice(name="Standard (BeautifulSoup)", value="standard"),
])
@app_commands.choices(gemini_thinking_level=[
    app_commands.Choice(name="High", value="high"),
    app_commands.Choice(name="Low", value="low"),
])
@app_commands.choices(gpt_effort=[
    app_commands.Choice(name="None", value="none"),
    app_commands.Choice(name="Low", value="low"),
    app_commands.Choice(name="Medium", value="medium"),
    app_commands.Choice(name="High", value="high"),
    app_commands.Choice(name="XHigh", value="xhigh"),
])
@app_commands.autocomplete(model=model_autocomplete)
@app_commands.autocomplete(profile=profile_autocomplete)
async def translate(
    interaction: discord.Interaction,
    api_key: str,
    model: Optional[str] = None,
    file: discord.Attachment = None,
    url: str = None,
    custom_endpoint_url: Optional[str] = None,
    google_credentials_file: discord.Attachment = None,
    extraction_mode: str = "enhanced",
    temperature: float = 0.3,
    batch_size: int = 10,
    max_output_tokens: int = 65536,
    disable_smart_filter: bool = False,
    duplicate_algorithm: str = "balanced",
    manual_glossary: discord.Attachment = None,
    enable_auto_glossary: bool = True,
    request_merge_count: Optional[int] = None,
    split_the_merge: bool = True,
    send_zip: bool = False,
    compression_factor: float = None,
    thinking: bool = True,
    gemini_thinking_level: str = "high",
    gemini_thinking_budget: int = -1,
    or_thinking_tokens: int = 2000,
    gpt_effort: str = "medium",
    target_language: str = "English",
    profile: str = "Universal"
):
    """Translate file using Glossarion"""
    
    # Acknowledge ASAP to avoid 10062 Unknown interaction under load.
    if not await _safe_defer(interaction, ephemeral=_ephemeral(interaction)):
        return

    _ensure_stable_cwd()

    if not GLOSSARION_AVAILABLE:
        await _safe_edit_original_response(interaction, content="❌ Glossarion not available")
        return
    
    # Validate input - must have either file or URL
    if not file and not url:
        await _safe_edit_original_response(interaction, content="❌ Please provide either a file attachment or a URL")
        return
    
    # Get filename and validate extension
    if file:
        filename = file.filename
    elif url:
        # Extract filename from URL or use default
        if 'drive.google.com' in url:
            filename = 'google_drive_file.epub'  # Will be updated after download
        elif 'dropbox.com' in url:
            filename = 'dropbox_file.epub'  # Will be updated after download
        else:
            # Try to get filename from URL path
            from urllib.parse import urlparse, unquote
            parsed = urlparse(url)
            filename = unquote(os.path.basename(parsed.path)) or 'downloaded_file.epub'

    # Never trust user/remote-provided names to be a safe path.
    # Keep /translate isolated from path traversal and accidental subdirectories.
    filename = os.path.basename(filename)

    # Validate file extension
    if not (filename.endswith('.epub') or filename.endswith('.txt') or filename.endswith('.pdf')):
        await _safe_edit_original_response(interaction, content="❌ File must be EPUB, TXT, or PDF format")
        return

    # Validate request merge count early (if explicitly provided)
    if request_merge_count is not None and request_merge_count < 0:
        await _safe_edit_original_response(interaction, content="❌ request_merge_count must be >= 0")
        return

    # Validate custom endpoint URL early (if provided)
    if custom_endpoint_url is not None:
        custom_endpoint_url = custom_endpoint_url.strip()
        if custom_endpoint_url and not (custom_endpoint_url.startswith('http://') or custom_endpoint_url.startswith('https://')):
            await _safe_edit_original_response(interaction, content="❌ custom_endpoint_url must start with http:// or https://")
            return

    user_id = interaction.user.id
    user_config = load_user_config(user_id)

    # Load config early so we can default the model without relying on autocomplete.
    config = load_config()

    # Default model: prefer explicit user input, otherwise config.json, then env var, then a safe fallback.
    model = (model or '').strip() or (config.get('model') or '').strip() or (os.getenv('MODEL') or '').strip() or 'gpt-4o'

    # Initial response (we already deferred; now edit the original response)
    profile_note = (
        f"Profile: {profile or 'Universal'}\n"
        "Note: target_lang replacement is skipped for profile prompts; choose the profile language you need.\n"
        "Profiles ending with _txt use html2text (default). Others use BeautifulSoup. 'ocr' profiles are for images."
    )
    embed = discord.Embed(
        title="📚 Translation Started",
        description=(
            f"**File:** {filename}\n"
            f"**Model:** {model}\n"
            f"**Target:** {target_language}\n\n"
            f"{profile_note}"
        ),
        color=discord.Color.blue()
    )
    msg_obj = await _safe_edit_original_response(interaction, embed=embed)
    if msg_obj is None:
        return
    try:
        message = await interaction.original_response()
    except Exception:
        message = None
    if message is None:
        return
    # Persist user preferences for next runs
    try:
        save_user_config(user_id, {
            "translate": {
                "model": model,
                "custom_endpoint_url": custom_endpoint_url,
                "extraction_mode": extraction_mode,
                "temperature": temperature,
                "batch_size": batch_size,
                "max_output_tokens": max_output_tokens,
                "disable_smart_filter": disable_smart_filter,
                "duplicate_algorithm": duplicate_algorithm,
                "enable_auto_glossary": enable_auto_glossary,
                "request_merge_count": request_merge_count,
                "split_the_merge": split_the_merge,
                "send_zip": send_zip,
                "compression_factor": compression_factor,
                "thinking": thinking,
                "gemini_thinking_level": gemini_thinking_level,
                "gemini_thinking_budget": gemini_thinking_budget,
                "or_thinking_tokens": or_thinking_tokens,
                "gpt_effort": gpt_effort,
                "target_language": target_language,
                "profile": profile,
            }
        })
    except Exception as e:
        sys.stderr.write(f"[CONFIG] Failed to persist /translate prefs: {e}\n")
        sys.stderr.flush()
    
    # Create temp directory
    temp_dir = tempfile.mkdtemp(
        prefix=f"discord_translate_{interaction.user.id}_",
        dir=BOT_TMPDIR or None,
    )
    input_path = os.path.join(temp_dir, filename)
    
    try:
        # Download file from attachment or URL
        if file:
            await file.save(input_path)
        elif url:
            import aiohttp
            
            # Convert Google Drive/Dropbox share links to direct download links
            download_url = url
            if 'drive.google.com' in url:
                # Extract file ID from various Google Drive URL formats
                if '/file/d/' in url:
                    file_id = url.split('/file/d/')[1].split('/')[0]
                elif 'id=' in url:
                    file_id = url.split('id=')[1].split('&')[0]
                else:
                    await interaction.edit_original_response(embed=discord.Embed(
                        title="❌ Invalid URL",
                        description="Could not parse Google Drive file ID from URL",
                        color=discord.Color.red()
                    ))
                    return
                download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
            elif 'dropbox.com' in url:
                # Convert Dropbox share link to direct download
                download_url = url.replace('www.dropbox.com', 'dl.dropboxusercontent.com').replace('?dl=0', '').replace('?dl=1', '')
                if '?dl=' not in download_url:
                    download_url += '?dl=1'
            
            # Download the file
            async with aiohttp.ClientSession() as session:
                async with session.get(download_url) as response:
                    if response.status == 200:
                        with open(input_path, 'wb') as f:
                            f.write(await response.read())
                        
                        # Try to get actual filename from response headers
                        if 'content-disposition' in response.headers:
                            import re
                            content_disp = response.headers['content-disposition']
                            fname_match = re.findall('filename="(.+)"', content_disp)
                            if fname_match:
                                actual_filename = fname_match[0]
                                # Update filename if we got a better one
                                new_input_path = os.path.join(temp_dir, actual_filename)
                                os.rename(input_path, new_input_path)
                                input_path = new_input_path
                                filename = actual_filename
                    else:
                        await interaction.edit_original_response(embed=discord.Embed(
                            title="❌ Download Failed",
                            description=f"Failed to download file from URL (HTTP {response.status})",
                            color=discord.Color.red()
                        ))
                        return

        # Ensure the input file actually exists before starting the executor thread.
        # If this fails, /translate can end up in a bad state (e.g. cwd inside a deleted temp dir).
        try:
            if not os.path.isfile(input_path):
                raise FileNotFoundError(f"Downloaded/attached file not found on disk: {input_path}")
            if os.path.getsize(input_path) <= 0:
                raise FileNotFoundError(f"Downloaded/attached file is empty: {input_path}")
        except Exception as e:
            await interaction.edit_original_response(embed=discord.Embed(
                title="❌ Input File Error",
                description=str(e),
                color=discord.Color.red()
            ))
            return

        # Get system prompt from config
        prompt_profiles = config.get('prompt_profiles', {})
        chosen_profile = (profile or "Universal").strip()
        system_prompt = None
        if prompt_profiles:
            # When a profile is chosen, use it as-is; do not substitute target_lang (per user note).
            system_prompt = prompt_profiles.get(chosen_profile)
            if system_prompt is None and 'Universal' in prompt_profiles:
                system_prompt = prompt_profiles['Universal']
        if not system_prompt:
            # Fallback to basic prompt
            system_prompt = f"Translate to {target_language}. Preserve all formatting."
        
        # Custom OpenAI Endpoint (single source of truth: custom_endpoint_url)
        # If omitted, keep disabled.
        if custom_endpoint_url:
            os.environ['USE_CUSTOM_OPENAI_ENDPOINT'] = '1'
            os.environ['OPENAI_CUSTOM_BASE_URL'] = custom_endpoint_url
            sys.stderr.write(f"[CONFIG] Custom OpenAI Endpoint enabled: {custom_endpoint_url}\n")
        else:
            os.environ['USE_CUSTOM_OPENAI_ENDPOINT'] = '0'
            os.environ['OPENAI_CUSTOM_BASE_URL'] = ''
        
        # Set model and API key
        os.environ['MODEL'] = model
        os.environ['SYSTEM_PROMPT'] = system_prompt
        os.environ['PROFILE_NAME'] = chosen_profile.lower() if chosen_profile else "universal"
        os.environ['OUTPUT_DIRECTORY'] = temp_dir
        
        # Set translation parameters
        os.environ['BATCH_TRANSLATION'] = '1'
        os.environ['BATCH_SIZE'] = str(batch_size)
        os.environ['MAX_OUTPUT_TOKENS'] = str(max_output_tokens)
        os.environ['TRANSLATION_TEMPERATURE'] = str(temperature)
        # Preserve markdown/structure and force BeautifulSoup for traditional APIs (from Other Settings)
        os.environ['ENHANCED_PRESERVE_STRUCTURE'] = '1' if config.get('enhanced_preserve_structure', True) else '0'
        os.environ['FORCE_BS_FOR_TRADITIONAL'] = '1' if config.get('force_bs_for_traditional', False) else '0'
        
        # Handle compression factor
        # TransateKRtoEN ultimately uses COMPRESSION_FACTOR; "AUTO_COMPRESSION_FACTOR" is treated
        # as a UI/bot toggle only.
        if compression_factor is not None:
            # Explicit override from the slash command.
            os.environ['COMPRESSION_FACTOR'] = str(compression_factor)
            os.environ['AUTO_COMPRESSION_FACTOR'] = '0'
            sys.stderr.write(f"[CONFIG] Manual compression factor: {compression_factor} (Auto-compression disabled)\n")
        else:
            # Respect config setting for auto-compression.
            # (config.json uses auto_compress_enabled; older variants may use auto_compression_factor)
            auto_comp = bool(config.get('auto_compress_enabled', config.get('auto_compression_factor', True)))
            os.environ['AUTO_COMPRESSION_FACTOR'] = '1' if auto_comp else '0'

            # Always set a deterministic COMPRESSION_FACTOR so downstream code doesn't fall back
            # to a different default unexpectedly.
            os.environ['COMPRESSION_FACTOR'] = str(config.get('compression_factor', 3.0))
        
        # Disable contextual translation by default (each batch is independent)
        os.environ['CONTEXTUAL'] = '0'
        # Disable emergency paragraph restoration
        os.environ['EMERGENCY_PARAGRAPH_RESTORE'] = '0'
        # Enable AI artifact removal
        os.environ['REMOVE_AI_ARTIFACTS'] = '1'
        # Retain original source filenames (no 'response_' prefix)
        os.environ['RETAIN_SOURCE_EXTENSION'] = '1'
        
        # Disable input token limit by default (no chapter size restrictions)
        os.environ['TOKEN_LIMIT_DISABLED'] = '1'
        os.environ['DISABLE_INPUT_TOKEN_LIMIT'] = '1'
        os.environ['MAX_INPUT_TOKENS'] = ''  # Empty string = unlimited (matches GUI behavior)
        
        # Disable image translation for Discord bot (images don't work well via Discord)
        os.environ['ENABLE_IMAGE_TRANSLATION'] = '0'
        
        # Set extraction mode
        os.environ['TEXT_EXTRACTION_METHOD'] = extraction_mode
        if extraction_mode == 'enhanced':
            os.environ['EXTRACTION_MODE'] = 'enhanced'
            os.environ['ENHANCED_FILTERING'] = 'smart'
            os.environ['ENHANCED_PRESERVE_STRUCTURE'] = '1'
        else:
            os.environ['EXTRACTION_MODE'] = 'smart'
            os.environ['FILE_FILTERING_LEVEL'] = 'smart'
        
        # Set PDF-specific styling extraction variables from pdf_extractor
        # These ensure PDF font sizes, alignments, and styles are preserved
        if filename.endswith('.pdf'):
            sys.stderr.write(f"[CONFIG] Enabling PDF formatting extraction (font size, alignment, etc.)\n")
            sys.stderr.flush()
            # Force XHTML render mode for better PDF extraction quality
            os.environ['PDF_RENDER_MODE'] = 'xhtml'
            sys.stderr.write(f"[CONFIG] Using XHTML render mode for PDF\n")
            sys.stderr.flush()
            # The pdf_extractor.generate_css_from_pdf() function will automatically
            # detect and apply: base_font_size, font_family, text_align, line_height_ratio
            # from the actual PDF file during processing
        
        # Enable automatic glossary generation (user configurable)
        os.environ['ENABLE_AUTO_GLOSSARY'] = '1' if enable_auto_glossary else '0'
        # Set glossary parameters (use config if available, otherwise use defaults)
        os.environ['GLOSSARY_MIN_FREQUENCY'] = str(config.get('glossary_min_frequency', 2))
        os.environ['GLOSSARY_MAX_NAMES'] = str(config.get('glossary_max_names', 50))
        os.environ['GLOSSARY_MAX_TITLES'] = str(config.get('glossary_max_titles', 30))
        os.environ['APPEND_GLOSSARY'] = '1'

        # IMPORTANT: Treat empty-string config values as missing.
        # TransateKRtoEN.build_system_prompt() hard-fails if APPEND_GLOSSARY_PROMPT is blank.
        append_prompt = (config.get('append_glossary_prompt') or '').strip()
        if not append_prompt:
            append_prompt = '- Follow this reference glossary for consistent translation (Do not output any raw entries):'
        if not append_prompt.endswith('\n'):
            append_prompt += '\n'
        os.environ['APPEND_GLOSSARY_PROMPT'] = append_prompt

        # CRITICAL: Auto glossary uses AUTO_GLOSSARY_PROMPT (unified prompt used by the GUI).
        # If this is missing, GlossaryManager falls back to the legacy honorific/title regex scanner.
        os.environ['AUTO_GLOSSARY_PROMPT'] = (
            config.get('unified_auto_glossary_prompt', '')
            or config.get('auto_glossary_prompt', '')
            or ''
        )
        # Ensure glossary translations target the same language as the main translation
        os.environ['GLOSSARY_TARGET_LANGUAGE'] = target_language
        os.environ['OUTPUT_LANGUAGE'] = target_language
        # Align throttling/timeouts with config defaults (matches GUI behavior)
        os.environ['SEND_INTERVAL_SECONDS'] = str(config.get('delay', 2.0))
        os.environ['THREAD_SUBMISSION_DELAY_SECONDS'] = str(config.get('thread_submission_delay', 0.5))
        os.environ['RETRY_TRUNCATED'] = '1'  # Enable auto-retry truncated responses by default
        os.environ['MAX_RETRY_TOKENS'] = str(max_output_tokens)  # Align retry token limit to output token limit
        os.environ['RETRY_TIMEOUT'] = '1' if config.get('retry_timeout', False) else '0'
        os.environ['CHUNK_TIMEOUT'] = str(config.get('chunk_timeout', 900))
        os.environ['ENABLE_HTTP_TUNING'] = '1' if config.get('enable_http_tuning', False) else '0'
        os.environ['CONNECT_TIMEOUT'] = str(config.get('connect_timeout', 10))
        # Don't set READ_TIMEOUT for the bot; chunk timeout is the single source of truth.
        os.environ.pop('READ_TIMEOUT', None)
        os.environ['HTTP_POOL_CONNECTIONS'] = str(config.get('http_pool_connections', 20))
        os.environ['HTTP_POOL_MAXSIZE'] = str(config.get('http_pool_maxsize', 50))
        os.environ['IGNORE_RETRY_AFTER'] = '1' if config.get('ignore_retry_after', False) else '0'
        # Cap retries for the Discord bot to keep runs predictable.
        os.environ['MAX_RETRIES'] = '3'
        # Set all glossary variables from GUI
        os.environ['GLOSSARY_COMPRESSION_FACTOR'] = str(config.get('glossary_compression_factor', 1.2))
        # Enable glossary prompt compression (filtering unused entries) by default
        os.environ['COMPRESS_GLOSSARY_PROMPT'] = '1' if config.get('compress_glossary_prompt', True) else '0'
        os.environ['GLOSSARY_FILTER_MODE'] = config.get('glossary_filter_mode', 'all')
        os.environ['GLOSSARY_STRIP_HONORIFICS'] = '1' if config.get('glossary_strip_honorifics', True) else '0'
        os.environ['GLOSSARY_FUZZY_THRESHOLD'] = str(config.get('glossary_fuzzy_threshold', 0.90))
        os.environ['GLOSSARY_MAX_TEXT_SIZE'] = str(config.get('glossary_max_text_size', 50000))
        # Cap glossary max sentences for the Discord bot to keep prompts small/predictable.
        # (GlossaryManager reads this via GLOSSARY_MAX_SENTENCES)
        os.environ['GLOSSARY_MAX_SENTENCES'] = '200'
        os.environ['GLOSSARY_CHAPTER_SPLIT_THRESHOLD'] = str(config.get('glossary_chapter_split_threshold', 50000))
        os.environ['GLOSSARY_SKIP_FREQUENCY_CHECK'] = '0'  # Enable frequency checking
        os.environ['CONTEXT_WINDOW_SIZE'] = str(config.get('glossary_context_window', 2))
        os.environ['GLOSSARY_USE_LEGACY_CSV'] = '0'  # Use modern JSON format
        os.environ['GLOSSARY_DUPLICATE_KEY_MODE'] = config.get('glossary_duplicate_key_mode', 'auto')
        os.environ['GLOSSARY_DUPLICATE_CUSTOM_FIELD'] = config.get('glossary_duplicate_custom_field', '')
        os.environ['GLOSSARY_DUPLICATE_ALGORITHM'] = duplicate_algorithm
        # Gender context and description for automatic glossary (enabled by default)
        os.environ['GLOSSARY_INCLUDE_GENDER_CONTEXT'] = '1' if config.get('include_gender_context', True) else '0'
        os.environ['GLOSSARY_INCLUDE_DESCRIPTION'] = '1' if config.get('include_description', True) else '0'
        # Custom glossary fields (additional columns) - default to ['description']
        custom_fields = config.get('custom_glossary_fields', [])
        if not custom_fields and not config.get('custom_field_description_removed', False):
            custom_fields = ['description']
        os.environ['GLOSSARY_CUSTOM_FIELDS'] = json.dumps(custom_fields)
        # Glossary-specific overrides for API settings
        os.environ['GLOSSARY_MAX_OUTPUT_TOKENS'] = str(config.get('glossary_max_output_tokens', max_output_tokens))
        os.environ['GLOSSARY_TEMPERATURE'] = str(config.get('manual_glossary_temperature', 0.1))
        os.environ['GLOSSARY_REQUEST_MERGING_ENABLED'] = '0'  # Disable by default
        os.environ['GLOSSARY_REQUEST_MERGE_COUNT'] = str(config.get('glossary_request_merge_count', 10))
        
        # Set duplicate detection mode to balanced
        os.environ['DUPLICATE_DETECTION_MODE'] = 'balanced'
        
        # Disable batch translate headers (metadata translation)
        os.environ['BATCH_TRANSLATE_HEADERS'] = '0'
        
        # Set manual glossary path if provided (download attachment first)
        if manual_glossary:
            # Validate glossary file extension
            if manual_glossary.filename.endswith('.csv') or manual_glossary.filename.endswith('.json'):
                glossary_path = os.path.join(temp_dir, manual_glossary.filename)
                await manual_glossary.save(glossary_path)
                os.environ['MANUAL_GLOSSARY'] = glossary_path
                sys.stderr.write(f"[CONFIG] Using manual glossary: {manual_glossary.filename}\n")
                sys.stderr.flush()
            else:
                sys.stderr.write(f"[WARNING] Manual glossary must be .csv or .json: {manual_glossary.filename}\n")
                sys.stderr.flush()
        
        # Request merging settings (combine multiple chapters into single API request)
        # Single source of truth: request_merge_count
        if request_merge_count is None:
            # If omitted, keep merging disabled
            request_merging_enabled = False
            request_merge_count_raw = 1
        else:
            request_merge_count_raw = int(request_merge_count)
            request_merging_enabled = request_merge_count_raw >= 2

        # Keep the count safe for downstream code (even when disabled)
        request_merge_count_effective = max(1, request_merge_count_raw)

        os.environ['REQUEST_MERGING_ENABLED'] = '1' if request_merging_enabled else '0'
        os.environ['REQUEST_MERGE_COUNT'] = str(request_merge_count_effective)
        os.environ['SPLIT_THE_MERGE'] = '1' if split_the_merge else '0'
        os.environ['DISABLE_MERGE_FALLBACK'] = '1'  # Mark as qa_failed if split fails
        os.environ['SYNTHETIC_MERGE_HEADERS'] = '1'  # Use synthetic headers for better splitting
        
        # Disable Gemini safety filter by default (enabled for Discord bot)
        os.environ['DISABLE_GEMINI_SAFETY'] = 'true'
        
        # Handle Thinking Toggle
        # If thinking is True (default), we don't need to do anything as we respect the config/env
        # If thinking is False, we explicitly disable all thinking features
        if not thinking:
            os.environ['ENABLE_GPT_THINKING'] = '0'
            os.environ['ENABLE_GEMINI_THINKING'] = '0'
            os.environ['ENABLE_DEEPSEEK_THINKING'] = '0'
            sys.stderr.write(f"[CONFIG] Thinking capabilities disabled via command\n")
        
        # Handle Vertex AI / Google Cloud credentials
        if '@' in model or model.startswith('vertex/'):
            # Prefer an uploaded credentials file; otherwise fall back to config.json.
            google_creds = None
            if google_credentials_file is not None:
                # Save uploaded creds into the job temp dir.
                # Discord provides the filename; ensure it's safe.
                creds_name = os.path.basename(getattr(google_credentials_file, 'filename', '') or 'google_credentials.json')
                if not creds_name.lower().endswith('.json'):
                    creds_name = 'google_credentials.json'
                google_creds = os.path.join(temp_dir, creds_name)
                try:
                    await google_credentials_file.save(google_creds)
                except Exception as e:
                    sys.stderr.write(f"[CONFIG] Failed to save uploaded Google credentials: {e}\n")
                    google_creds = None

            if google_creds is None:
                google_creds = config.get('google_cloud_credentials')

            if google_creds and os.path.exists(google_creds):
                os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = google_creds
                sys.stderr.write(f"[CONFIG] Using Google Cloud credentials: {os.path.basename(google_creds)}\n")
                sys.stderr.flush()

                # Extract project ID from credentials
                try:
                    with open(google_creds, 'r', encoding='utf-8') as f:
                        creds_data = json.load(f)
                        project_id = creds_data.get('project_id', 'vertex-ai-project')
                        os.environ['GOOGLE_CLOUD_PROJECT'] = project_id
                        if not api_key:
                            api_key = project_id
                except Exception:
                    pass
        
        # Set API key - TransateKRtoEN checks multiple env vars
        os.environ['API_KEY'] = api_key
        os.environ['OPENAI_API_KEY'] = api_key
        os.environ['OPENAI_OR_Gemini_API_KEY'] = api_key
        
        # Set provider-specific keys
        if 'claude' in model.lower():
            os.environ['ANTHROPIC_API_KEY'] = api_key
        elif 'gemini' in model.lower():
            os.environ['GOOGLE_API_KEY'] = api_key
            os.environ['GEMINI_API_KEY'] = api_key
        
        # Initialize translation state in global storage
        user_id = interaction.user.id
        translation_states[user_id] = {
            'logs': [],
            'show_full': False,
            'stop_requested': False,
            'last_update': 0,
            'pending_update': False
        }
        state = translation_states[user_id]
        
        def log_callback(msg):
            if msg and msg.strip():
                state['logs'].append(msg.strip())
                # Use stderr to avoid recursion (stdout is redirected to callback)
                sys.stderr.write(f"[LOG] {msg.strip()}\n")
                sys.stderr.flush()
                
                # Rate limit: update at most once per second to avoid Discord rate limits
                import time
                current_time = time.time()
                if current_time - state['last_update'] >= 1.0:
                    state['last_update'] = current_time
                    state['pending_update'] = False
                    asyncio.run_coroutine_threadsafe(update_progress(), bot.loop)
                else:
                    # Mark that we have a pending update
                    state['pending_update'] = True
        
        def stop_callback():
            """Check if stop was requested"""
            return state['stop_requested']
        
        async def periodic_update_check():
            """Check for pending updates every second and flush them"""
            import time
            while user_id in translation_states and not state['stop_requested']:
                await asyncio.sleep(1)
                if state.get('pending_update', False):
                    state['pending_update'] = False
                    state['last_update'] = time.time()
                    await update_progress()
        
        async def update_progress():
            try:
                logs = state['logs']
                # Respect the user's choice of log view (show_full)
                if state['show_full']:
                    # Show all logs, truncated to Discord's 4096 char limit
                    log_text = '\n'.join(logs)
                    if len(log_text) > 3900:
                        log_text = "..." + log_text[-3900:]
                else:
                    # Show last 10 logs (increased from 5 for better visibility)
                    log_text = '\n'.join(logs[-10:])
                    if len(log_text) > 800:
                        log_text = log_text[-800:]
                
                if not log_text:
                    log_text = "Starting..."
                
                embed = discord.Embed(
                    title="📚 Translation in Progress",
                    description=f"**Status:** Processing... ({len(logs)} logs)\n\n```{log_text}```",
                    color=discord.Color.blue()
                )
                
                # Add buttons to toggle log view and stop translation
                view = LogView(user_id)
                await message.edit(embed=embed, view=view)
            except Exception as e:
                sys.stderr.write(f"[ERROR] Failed to update progress: {e}\n")
                sys.stderr.flush()
        
        # Run translation
        await update_progress()
        
        # Start periodic update checker
        update_task = asyncio.create_task(periodic_update_check())
        
        def run_translation():
            # Ensure we aren't running inside a deleted directory.
            _ensure_stable_cwd()

            sys.stderr.write(f"[TRANSLATE] Starting translation for: {input_path}\\n")
            sys.stderr.write(f"[TRANSLATE] Temp directory: {temp_dir}\\n")
            sys.stderr.flush()

            # IMPORTANT: Do NOT chdir() in a multi-command Discord bot.
            # chdir() is process-wide and can break other concurrent interactions.
            # TransateKRtoEN is patched to respect OUTPUT_DIRECTORY for absolute output paths.
            original_argv = sys.argv[:]
            try:
                sys.argv = ['discord_bot.py', input_path]
                result = TransateKRtoEN.main(log_callback=log_callback, stop_callback=stop_callback)

                sys.stderr.write(f"[TRANSLATE] Translation completed\\n")
                sys.stderr.flush()
                return result
            finally:
                # Prevent leaking argv changes across commands.
                sys.argv = original_argv
        
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, run_translation)
        
        # Cancel the periodic update task
        update_task.cancel()
        try:
            await update_task
        except asyncio.CancelledError:
            pass
        
        # Determine output format and file
        output_file_path = None
        output_display_name = None
        is_zip_output = False

        # Prefer deterministic output selection from the translation output folder.
        # TransateKRtoEN writes into: OUTPUT_DIRECTORY/<file_base>/...
        output_base = os.path.splitext(os.path.basename(filename))[0]
        safe_base = output_base.replace('/', '_').replace('\\', '_').replace(':', '_').replace('*', '_').replace('?', '_').replace('"', '_').replace('<', '_').replace('>', '_').replace('|', '_')

        # The translator historically used the raw base name; keep both candidates for compatibility.
        output_dir_candidates = [
            os.path.join(temp_dir, output_base),
            os.path.join(temp_dir, safe_base),
        ]
        output_subdir = None
        for d in output_dir_candidates:
            if os.path.exists(d) and os.path.isdir(d):
                output_subdir = d
                break
        if output_subdir is None:
            output_subdir = os.path.join(temp_dir, safe_base)

        def _pick_newest(d: str, exts: list[str], must_contain: Optional[str] = None) -> Optional[str]:
            try:
                if not os.path.exists(d) or not os.path.isdir(d):
                    return None
                candidates = []
                for name in os.listdir(d):
                    p = os.path.join(d, name)
                    if p == input_path:
                        continue
                    if name.lower().endswith('.zip'):
                        continue
                    nlow = name.lower()
                    if exts and not any(nlow.endswith(e) for e in exts):
                        continue
                    if must_contain and must_contain.lower() not in nlow:
                        continue
                    if os.path.isfile(p):
                        try:
                            mtime = os.path.getmtime(p)
                        except Exception:
                            mtime = 0
                        candidates.append((mtime, p))
                if not candidates:
                    return None
                candidates.sort(key=lambda t: t[0], reverse=True)
                return candidates[0][1]
            except Exception:
                return None

        input_ext = os.path.splitext(filename)[1].lower()

        # If user didn't request zip, try to pick the most likely final artifact.
        if not send_zip:
            sys.stderr.write(f"[OUTPUT] Selecting output from: {output_subdir}\\n")
            sys.stderr.flush()

            # Prefer translated-named artifacts for text/PDF flows.
            if input_ext == '.epub':
                output_file_path = _pick_newest(output_subdir, exts=['.epub'])
            elif input_ext == '.pdf':
                output_file_path = (
                    _pick_newest(output_subdir, exts=['.pdf'], must_contain='_translated')
                    or _pick_newest(output_subdir, exts=['.html'], must_contain='_translated')
                    or _pick_newest(output_subdir, exts=['.txt'], must_contain='_translated')
                    or _pick_newest(output_subdir, exts=['.pdf', '.html', '.txt'])
                )
            else:
                # .txt and other text-like inputs
                output_file_path = (
                    _pick_newest(output_subdir, exts=['.txt'], must_contain='_translated')
                    or _pick_newest(output_subdir, exts=['.pdf'], must_contain='_translated')
                    or _pick_newest(output_subdir, exts=['.html'], must_contain='_translated')
                    or _pick_newest(output_subdir, exts=[input_ext])
                    or _pick_newest(output_subdir, exts=['.pdf', '.html', '.txt'])
                )

            if output_file_path:
                output_display_name = os.path.basename(output_file_path)
                sys.stderr.write(f"[OUTPUT] Selected output file: {output_file_path}\\n")
                sys.stderr.flush()

        # If we selected a file but it doesn't exist, treat it as a miss and fall back to ZIP.
        try:
            if output_file_path and not os.path.exists(output_file_path):
                sys.stderr.write(f"[OUTPUT] Selected file missing on disk, falling back to ZIP: {output_file_path}\\n")
                sys.stderr.flush()
                output_file_path = None
                output_display_name = None
        except Exception:
            output_file_path = None
            output_display_name = None

        # If zip requested OR file not found, proceed with zipping
        if send_zip or not output_file_path:
            is_zip_output = True

            sys.stderr.write(f"[ZIP] Creating zip archive of output...\\n")
            sys.stderr.write(f"[ZIP] Temp dir: {temp_dir}\\n")
            sys.stderr.write(f"[ZIP] Output dir: {output_subdir}\\n")

            # Zip the translation output folder if it exists; otherwise zip the whole temp dir.
            if os.path.exists(output_subdir) and os.path.isdir(output_subdir):
                zip_source_dir = output_subdir
                sys.stderr.write(f"[ZIP] Using output directory as source\\n")
            else:
                zip_source_dir = temp_dir
                sys.stderr.write(f"[ZIP] Using temp dir as source (no output dir found)\\n")
            sys.stderr.flush()
            
            zip_filename = f"{safe_base}_translated.zip"
            zip_path = os.path.join(temp_dir, zip_filename)
            
            # Update status to show zipping
            embed = discord.Embed(
                title="📦 Creating Archive",
                description="Compressing output files...",
                color=discord.Color.blue()
            )
            try:
                await message.edit(embed=embed, view=None)
            except discord.errors.HTTPException:
                pass
            
            try:
                # Create zip archive in background thread
                def create_zip():
                    sys.stderr.write(f"[ZIP] Starting compression...\\n")
                    # ... (zip logic) ...
                    import zipfile
                    files_added = 0
                    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                        for root, dirs, files in os.walk(zip_source_dir):
                            for file_item in files:
                                file_path = os.path.join(root, file_item)
                                # Skip zip and input
                                if file_item.endswith('.zip'): continue
                                if file_path == input_path: continue
                                
                                arcname = os.path.relpath(file_path, zip_source_dir)
                                zipf.write(file_path, arcname)
                                files_added += 1
                    return zip_path
                
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, create_zip)
                
                output_file_path = zip_path
                output_display_name = zip_filename
                
            except Exception as e:
                sys.stderr.write(f"[ERROR] Failed to create zip: {e}\\n")
                raise e

        async def _dm_or_channel(content: str = None, embed: Optional[discord.Embed] = None, file: Optional[discord.File] = None):
            # Always prefer DM; fall back to channel if DMs blocked.
            try:
                await interaction.user.send(content=content, embed=embed, file=file)
                return True
            except discord.Forbidden:
                try:
                    await interaction.channel.send(content=content, embed=embed, file=file)
                    return True
                except Exception:
                    return False
            except Exception:
                return False

        # Send the result (either zip or direct file)
        if os.path.exists(output_file_path):
            file_size = os.path.getsize(output_file_path)
            sys.stderr.write(f"[SUCCESS] Ready to send: {output_file_path} ({file_size / 1024 / 1024:.2f}MB)\\n")

            if _should_offload(file_size, send_zip):
                loop = asyncio.get_event_loop()
                download_url = None
                # 1) tmpfiles
                for uploader, name in (( _upload_to_tmpfiles, "tmpfiles"), (_upload_to_tempsh, "temp.sh"), (_upload_to_oracle_par, "oracle"), (_upload_to_gofile, "gofile")):
                    try:
                        download_url = await loop.run_in_executor(
                            None,
                            functools.partial(uploader, output_file_path, output_display_name),
                        )
                        break
                    except Exception as e2:
                        sys.stderr.write(f"[HOST ERROR] {name}: {e2}\n")
                        continue
                if download_url:
                    embed = discord.Embed(
                        title="✅ Translation Complete!",
                        description=(
                            f"**File:** {output_display_name}\\n"
                            f"**Size:** {file_size / 1024 / 1024:.2f}MB\\n"
                            f"**Download:** <{download_url}>"
                        ),
                        color=discord.Color.green()
                    )
                    await _dm_or_channel(embed=embed)
                else:
                    await _dm_or_channel(
                        content="Could not upload the output file right now. Please try again later."
                    )
            else:
                if state['stop_requested']:
                    title = "⏹️ Translation Stopped - Partial Results"
                    desc_text = "Contains partial translation output."
                    color = discord.Color.orange()
                else:
                    title = "✅ Translation Complete!"
                    desc_text = "Translation finished successfully."
                    color = discord.Color.green()
                
                embed = discord.Embed(
                    title=title,
                    description=f"**File:** {output_display_name}\\n**Size:** {file_size / 1024 / 1024:.2f}MB\\n\\n{desc_text}",
                    color=color
                )
                sent = await _dm_or_channel(embed=embed)
                
                try:
                    msg_content = f"Here's your {('zipped ' if is_zip_output else '')}translation output!"
                    await _dm_or_channel(
                        content=msg_content,
                        file=discord.File(output_file_path, filename=output_display_name),
                    )
                except Exception:
                    # Fallback: try uploading to Gofile if direct send fails
                    try:
                        loop = asyncio.get_event_loop()
                        download_url = await loop.run_in_executor(
                            None,
                            functools.partial(_upload_to_gofile, output_file_path, output_display_name),
                        )
                        await _dm_or_channel(
                            content=f"Download: {download_url}",
                        )
                    except Exception:
                        await _dm_or_channel(
                            content="Upload failed and Gofile is unavailable right now."
                        )
        else:
            raise FileNotFoundError(f"Output file not found: {output_file_path}")
    
    except Exception as e:
        import traceback
        error = f"```\n{traceback.format_exc()[-1000:]}\n```"
        embed = discord.Embed(
            title="❌ Error",
            description=f"{str(e)}\n{error}",
            color=discord.Color.red()
        )
        try:
            await message.edit(embed=embed, view=None)
        except discord.errors.HTTPException:
            await interaction.followup.send(embed=embed)
    
    finally:
        # Cleanup translation state
        if user_id in translation_states:
            del translation_states[user_id]

        # Ensure we are not sitting inside the temp dir (or a deleted cwd) before deleting it.
        try:
            cwd = os.getcwd()
        except FileNotFoundError:
            try:
                os.chdir(src_dir)
            except Exception:
                pass
        else:
            try:
                if os.path.commonpath([os.path.abspath(cwd), os.path.abspath(temp_dir)]) == os.path.abspath(temp_dir):
                    os.chdir(src_dir)
            except Exception:
                pass

        # Cleanup temp directory
        try:
            shutil.rmtree(temp_dir)
        except Exception:
            pass

@bot.tree.command(name="settings", description="Show your saved Glossarion bot preferences")
async def settings(interaction: discord.Interaction):
    """Display the caller's saved /translate and /extract defaults."""

    if not await _safe_defer(interaction, ephemeral=True):
        return

    user_id = interaction.user.id
    cfg = load_user_config(user_id) or {}
    translate_cfg = cfg.get("translate") or {}
    extract_cfg = cfg.get("extract") or {}

    if not translate_cfg and not extract_cfg:
        await _safe_edit_original_response(
            interaction,
            content="ℹ️ No saved settings yet. Run /translate or /extract once to store them."
        )
        return

    def _fmt(section: dict) -> str:
        if not section:
            return "_none_"
        import json
        text = json.dumps(section, indent=2, ensure_ascii=False)
        if len(text) > 1500:
            text = text[:1500] + "... (truncated)"
        return f"```json\n{text}\n```"

    embed = discord.Embed(
        title="Your Saved Settings",
        color=discord.Color.blue()
    )
    embed.add_field(name="Translate", value=_fmt(translate_cfg), inline=False)
    embed.add_field(name="Extract", value=_fmt(extract_cfg), inline=False)

    await _safe_edit_original_response(interaction, embed=embed)

@bot.tree.command(name="extract", description="Extract glossary from EPUB, TXT, or PDF file")
@app_commands.describe(
    api_key="Your API key",
    model="AI model to use (optional; defaults to config.json if omitted)",
    file="EPUB, TXT, or PDF file to extract glossary from (optional if using url)",
    url="Google Drive or Dropbox link to file (optional if using file attachment)",
    custom_endpoint_url="Custom OpenAI-compatible base URL (auto-enables when set; omit to disable)",
    google_credentials_file="Google Cloud credentials JSON file upload (for Vertex AI models)",
    extraction_mode="Text extraction method (default: Enhanced/html2text)",
    temperature="Glossary extraction temperature 0.0-1.0 (default: 0.1)",
    batch_size="Paragraphs per batch (default: 10)",
    max_output_tokens="Max output tokens (default: 65536)",
    glossary_compression_factor="Glossary compression factor (default: 1.2)",
    merge_count="Chapters per request (set >=2 to enable request merging; <=1 disables; omit to disable)",
    duplicate_algorithm="Duplicate handling: auto/strict/balanced/aggressive/basic (default: balanced)",
    send_zip="Return output as a ZIP archive instead of individual file (default: False)",
    thinking="Enable/disable AI thinking capabilities (GPT/Gemini/DeepSeek) - Default: True",
    gemini_thinking_level="Gemini 3 thinking level (low/high) - Default: high",
    gemini_thinking_budget="Gemini thinking budget (-1=auto, 0=disabled) - Default: -1",
    or_thinking_tokens="OpenRouter thinking tokens - Default: 2000",
    gpt_effort="GPT-5/OpenAI thinking effort (none/low/medium/high/xhigh) - Default: medium",
    target_language="Target language for translations"
)
@app_commands.choices(extraction_mode=[
    app_commands.Choice(name="Enhanced (html2text)", value="enhanced"),
    app_commands.Choice(name="Standard (BeautifulSoup)", value="standard"),
])
@app_commands.choices(gemini_thinking_level=[
    app_commands.Choice(name="High", value="high"),
    app_commands.Choice(name="Low", value="low"),
])
@app_commands.choices(gpt_effort=[
    app_commands.Choice(name="None", value="none"),
    app_commands.Choice(name="Low", value="low"),
    app_commands.Choice(name="Medium", value="medium"),
    app_commands.Choice(name="High", value="high"),
    app_commands.Choice(name="XHigh", value="xhigh"),
])
@app_commands.autocomplete(model=model_autocomplete)
async def extract(
    interaction: discord.Interaction,
    api_key: str,
    model: Optional[str] = None,
    file: discord.Attachment = None,
    url: str = None,
    custom_endpoint_url: Optional[str] = None,
    google_credentials_file: discord.Attachment = None,
    extraction_mode: str = "enhanced",
    temperature: float = 0.1,
    batch_size: int = 10,
    max_output_tokens: int = 65536,
    glossary_compression_factor: float = 1.2,
    merge_count: Optional[int] = None,
    duplicate_algorithm: str = "balanced",
    send_zip: bool = False,
    thinking: bool = True,
    gemini_thinking_level: str = "high",
    gemini_thinking_budget: int = -1,
    or_thinking_tokens: int = 2000,
    gpt_effort: str = "medium",
    target_language: str = "English"
):
    """Extract glossary from file using Glossarion"""
    
    # Acknowledge ASAP to avoid 10062 Unknown interaction under load.
    if not await _safe_defer(interaction, ephemeral=_ephemeral(interaction)):
        return

    _ensure_stable_cwd()

    if not GLOSSARION_AVAILABLE or not glossary_main:
        await _safe_edit_original_response(interaction, content="❌ Glossarion glossary extraction not available")
        return
    
    # Validate input - must have either file or URL
    if not file and not url:
        await _safe_edit_original_response(interaction, content="❌ Please provide either a file attachment or a URL")
        return
    
    # Get filename and validate extension
    if file:
        filename = file.filename
    elif url:
        # Extract filename from URL or use default
        if 'drive.google.com' in url:
            filename = 'google_drive_file.epub'
        elif 'dropbox.com' in url:
            filename = 'dropbox_file.epub'
        else:
            from urllib.parse import urlparse, unquote
            parsed = urlparse(url)
            filename = unquote(os.path.basename(parsed.path)) or 'downloaded_file.epub'

    # Never trust user/remote-provided names to be a safe path.
    # Keep /extract isolated from path traversal and accidental subdirectories.
    filename = os.path.basename(filename)

    # Validate file extension
    if not (filename.endswith('.epub') or filename.endswith('.txt') or filename.endswith('.pdf')):
        await _safe_edit_original_response(interaction, content="❌ File must be EPUB, TXT, or PDF format")
        return

    # Validate request merge count early (if explicitly provided)
    if merge_count is not None and merge_count < 0:
        await _safe_edit_original_response(interaction, content="❌ merge_count must be >= 0")
        return

    # Validate custom endpoint URL early (if provided)
    if custom_endpoint_url is not None:
        custom_endpoint_url = custom_endpoint_url.strip()
        if custom_endpoint_url and not (custom_endpoint_url.startswith('http://') or custom_endpoint_url.startswith('https://')):
            await _safe_edit_original_response(interaction, content="❌ custom_endpoint_url must start with http:// or https://")
            return
    user_id = interaction.user.id
    user_config = load_user_config(user_id)

    # Load config early so we can default the model without relying on autocomplete.
    config = load_config()

    # Default model: prefer explicit user input, otherwise config.json, then env var, then a safe fallback.
    model = (model or '').strip() or (config.get('model') or '').strip() or (os.getenv('MODEL') or '').strip() or 'gpt-4o'

    # Initial response (we already deferred; now edit the original response)
    embed = discord.Embed(
        title="📚 Glossary Extraction Started",
        description=f"**File:** {filename}\n**Model:** {model}\n**Target:** {target_language}",
        color=discord.Color.blue()
    )
    msg_obj = await _safe_edit_original_response(interaction, embed=embed)
    if msg_obj is None:
        return
    try:
        message = await interaction.original_response()
    except Exception:
        message = None
    if message is None:
        return
    # Persist user preferences for next runs
    try:
        save_user_config(user_id, {
            "extract": {
                "model": model,
                "custom_endpoint_url": custom_endpoint_url,
                "extraction_mode": extraction_mode,
                "temperature": temperature,
                "batch_size": batch_size,
                "max_output_tokens": max_output_tokens,
                "glossary_compression_factor": glossary_compression_factor,
                "merge_count": merge_count,
                "duplicate_algorithm": duplicate_algorithm,
                "send_zip": send_zip,
                "thinking": thinking,
                "gemini_thinking_level": gemini_thinking_level,
                "gemini_thinking_budget": gemini_thinking_budget,
                "or_thinking_tokens": or_thinking_tokens,
                "gpt_effort": gpt_effort,
                "target_language": target_language,
            }
        })
    except Exception as e:
        sys.stderr.write(f"[CONFIG] Failed to persist /extract prefs: {e}\n")
        sys.stderr.flush()
    
    # Create temp directory
    temp_dir = tempfile.mkdtemp(
        prefix=f"discord_extract_{interaction.user.id}_",
        dir=BOT_TMPDIR or None,
    )
    input_path = os.path.join(temp_dir, filename)
    
    try:
        # Download file from attachment or URL
        if file:
            await file.save(input_path)
        elif url:
            import aiohttp
            
            # Convert Google Drive/Dropbox share links to direct download links
            download_url = url
            if 'drive.google.com' in url:
                if '/file/d/' in url:
                    file_id = url.split('/file/d/')[1].split('/')[0]
                elif 'id=' in url:
                    file_id = url.split('id=')[1].split('&')[0]
                else:
                    await interaction.edit_original_response(embed=discord.Embed(
                        title="❌ Invalid URL",
                        description="Could not parse Google Drive file ID from URL",
                        color=discord.Color.red()
                    ))
                    return
                download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
            elif 'dropbox.com' in url:
                download_url = url.replace('www.dropbox.com', 'dl.dropboxusercontent.com').replace('?dl=0', '').replace('?dl=1', '')
                if '?dl=' not in download_url:
                    download_url += '?dl=1'
            
            # Download the file
            async with aiohttp.ClientSession() as session:
                async with session.get(download_url) as response:
                    if response.status == 200:
                        with open(input_path, 'wb') as f:
                            f.write(await response.read())
                        
                        # Try to get actual filename from response headers
                        if 'content-disposition' in response.headers:
                            import re
                            content_disp = response.headers['content-disposition']
                            fname_match = re.findall('filename="(.+)"', content_disp)
                            if fname_match:
                                actual_filename = os.path.basename(fname_match[0])
                                new_input_path = os.path.join(temp_dir, actual_filename)
                                os.rename(input_path, new_input_path)
                                input_path = new_input_path
                                filename = actual_filename
                    else:
                        await interaction.edit_original_response(embed=discord.Embed(
                            title="❌ Download Failed",
                            description=f"Failed to download file from URL (HTTP {response.status})",
                            color=discord.Color.red()
                        ))
                        return

        # Ensure the input file actually exists before starting the executor thread.
        try:
            if not os.path.isfile(input_path):
                raise FileNotFoundError(f"Downloaded/attached file not found on disk: {input_path}")
            if os.path.getsize(input_path) <= 0:
                raise FileNotFoundError(f"Downloaded/attached file is empty: {input_path}")
        except Exception as e:
            await interaction.edit_original_response(embed=discord.Embed(
                title="❌ Input File Error",
                description=str(e),
                color=discord.Color.red()
            ))
            return

        # Get glossary prompts from config
        glossary_prompt = config.get('manual_glossary_prompt', '')
        
        # Custom OpenAI Endpoint (single source of truth: custom_endpoint_url)
        # If omitted, keep disabled.
        if custom_endpoint_url:
            os.environ['USE_CUSTOM_OPENAI_ENDPOINT'] = '1'
            os.environ['OPENAI_CUSTOM_BASE_URL'] = custom_endpoint_url
            sys.stderr.write(f"[CONFIG] Custom OpenAI Endpoint enabled: {custom_endpoint_url}\n")
        else:
            os.environ['USE_CUSTOM_OPENAI_ENDPOINT'] = '0'
            os.environ['OPENAI_CUSTOM_BASE_URL'] = ''
        
        # Set model and API key
        os.environ['MODEL'] = model
        os.environ['GLOSSARY_SYSTEM_PROMPT'] = glossary_prompt
        
        # Set translation parameters (same as /translate)
        os.environ['BATCH_TRANSLATION'] = '1'
        os.environ['BATCH_SIZE'] = str(batch_size)
        os.environ['MAX_OUTPUT_TOKENS'] = str(max_output_tokens)
        os.environ['GLOSSARY_TEMPERATURE'] = str(temperature)
        os.environ['TRANSLATION_TEMPERATURE'] = str(temperature)
        os.environ['GLOSSARY_MAX_OUTPUT_TOKENS'] = str(max_output_tokens)
        # Preserve markdown/structure and force BeautifulSoup for traditional APIs (from Other Settings)
        os.environ['ENHANCED_PRESERVE_STRUCTURE'] = '1' if config.get('enhanced_preserve_structure', True) else '0'
        os.environ['FORCE_BS_FOR_TRADITIONAL'] = '1' if config.get('force_bs_for_traditional', False) else '0'
        
        # Set extraction mode
        os.environ['TEXT_EXTRACTION_METHOD'] = extraction_mode
        if extraction_mode == 'enhanced':
            os.environ['EXTRACTION_MODE'] = 'enhanced'
            os.environ['ENHANCED_FILTERING'] = 'smart'
            os.environ['ENHANCED_PRESERVE_STRUCTURE'] = '1'
        else:
            os.environ['EXTRACTION_MODE'] = 'smart'
            os.environ['FILE_FILTERING_LEVEL'] = 'smart'
        
        # Set PDF-specific styling extraction variables from pdf_extractor
        # These ensure PDF font sizes, alignments, and styles are preserved
        if filename.endswith('.pdf'):
            sys.stderr.write(f"[CONFIG] Enabling PDF formatting extraction (font size, alignment, etc.)\n")
            sys.stderr.flush()
            # Force XHTML render mode for better PDF extraction quality
            os.environ['PDF_RENDER_MODE'] = 'xhtml'
            sys.stderr.write(f"[CONFIG] Using XHTML render mode for PDF\n")
            sys.stderr.flush()
            # The pdf_extractor.generate_css_from_pdf() function will automatically
            # detect and apply: base_font_size, font_family, text_align, line_height_ratio
            # from the actual PDF file during processing
        
        # Set all glossary variables from config (same as /translate)
        os.environ['ENABLE_AUTO_GLOSSARY'] = '1'
        os.environ['GLOSSARY_MIN_FREQUENCY'] = str(config.get('glossary_min_frequency', 2))
        os.environ['GLOSSARY_MAX_NAMES'] = str(config.get('glossary_max_names', 50))
        os.environ['GLOSSARY_MAX_TITLES'] = str(config.get('glossary_max_titles', 30))
        os.environ['GLOSSARY_COMPRESSION_FACTOR'] = str(glossary_compression_factor)
        os.environ['GLOSSARY_FILTER_MODE'] = config.get('glossary_filter_mode', 'all')
        os.environ['GLOSSARY_STRIP_HONORIFICS'] = '1' if config.get('glossary_strip_honorifics', True) else '0'
        os.environ['GLOSSARY_FUZZY_THRESHOLD'] = str(config.get('glossary_fuzzy_threshold', 0.90))
        os.environ['GLOSSARY_MAX_TEXT_SIZE'] = str(config.get('glossary_max_text_size', 50000))
        # Cap glossary max sentences for the Discord bot to keep prompts small/predictable.
        # (GlossaryManager reads this via GLOSSARY_MAX_SENTENCES)
        os.environ['GLOSSARY_MAX_SENTENCES'] = '200'
        os.environ['GLOSSARY_CHAPTER_SPLIT_THRESHOLD'] = str(config.get('glossary_chapter_split_threshold', 50000))
        os.environ['GLOSSARY_SKIP_FREQUENCY_CHECK'] = '0'
        os.environ['CONTEXT_WINDOW_SIZE'] = str(config.get('glossary_context_window', 2))
        os.environ['GLOSSARY_CONTEXT_LIMIT'] = str(config.get('manual_context_limit', 2))
        os.environ['GLOSSARY_USE_LEGACY_CSV'] = '0'
        os.environ['GLOSSARY_DUPLICATE_KEY_MODE'] = 'skip'
        os.environ['GLOSSARY_DISABLE_HONORIFICS_FILTER'] = '1' if config.get('glossary_disable_honorifics_filter', False) else '0'
        # Ensure glossary output language matches the command's target_language
        os.environ['GLOSSARY_TARGET_LANGUAGE'] = target_language
        os.environ['OUTPUT_LANGUAGE'] = target_language
        # Align throttling/timeouts with config defaults (matches GUI behavior)
        os.environ['SEND_INTERVAL_SECONDS'] = str(config.get('delay', 2.0))
        os.environ['THREAD_SUBMISSION_DELAY_SECONDS'] = str(config.get('thread_submission_delay', 0.5))
        os.environ['RETRY_TIMEOUT'] = '1' if config.get('retry_timeout', False) else '0'
        os.environ['CHUNK_TIMEOUT'] = str(config.get('chunk_timeout', 900))
        os.environ['ENABLE_HTTP_TUNING'] = '1' if config.get('enable_http_tuning', False) else '0'
        os.environ['CONNECT_TIMEOUT'] = str(config.get('connect_timeout', 10))
        # Don't set READ_TIMEOUT for the bot; chunk timeout is the single source of truth.
        os.environ.pop('READ_TIMEOUT', None)
        os.environ['HTTP_POOL_CONNECTIONS'] = str(config.get('http_pool_connections', 20))
        os.environ['HTTP_POOL_MAXSIZE'] = str(config.get('http_pool_maxsize', 50))
        os.environ['IGNORE_RETRY_AFTER'] = '1' if config.get('ignore_retry_after', False) else '0'
        # Cap retries for the Discord bot to keep runs predictable.
        os.environ['MAX_RETRIES'] = '3'

        # Glossary request merging settings
        # Single source of truth: merge_count
        if merge_count is None:
            # If omitted, keep merging disabled
            glossary_request_merging_enabled = False
            glossary_request_merge_count_raw = 1
        else:
            glossary_request_merge_count_raw = int(merge_count)
            glossary_request_merging_enabled = glossary_request_merge_count_raw >= 2

        # Keep the count safe for downstream code (even when disabled)
        glossary_request_merge_count_effective = max(1, glossary_request_merge_count_raw)

        os.environ['GLOSSARY_REQUEST_MERGING_ENABLED'] = '1' if glossary_request_merging_enabled else '0'
        os.environ['GLOSSARY_REQUEST_MERGE_COUNT'] = str(glossary_request_merge_count_effective)
        os.environ['GLOSSARY_DUPLICATE_ALGORITHM'] = duplicate_algorithm
        # Use config defaults for gender context and description (manual glossary extraction)
        os.environ['GLOSSARY_INCLUDE_GENDER_CONTEXT'] = '1' if config.get('include_gender_context', True) else '0'
        os.environ['GLOSSARY_INCLUDE_DESCRIPTION'] = '1' if config.get('include_description', True) else '0'
        # Custom glossary fields (additional columns) - default to ['description']
        custom_fields = config.get('custom_glossary_fields', [])
        if not custom_fields and not config.get('custom_field_description_removed', False):
            custom_fields = ['description']
        os.environ['GLOSSARY_CUSTOM_FIELDS'] = json.dumps(custom_fields)
        os.environ['DISABLE_GEMINI_SAFETY'] = 'true'
        
        # Handle Thinking Toggle
        if not thinking:
            os.environ['ENABLE_GPT_THINKING'] = '0'
            os.environ['ENABLE_GEMINI_THINKING'] = '0'
            os.environ['ENABLE_DEEPSEEK_THINKING'] = '0'
            sys.stderr.write(f"[CONFIG] Thinking capabilities disabled via command\n")
        else:
            # Set specific thinking variables if thinking is enabled
            os.environ['GEMINI_THINKING_LEVEL'] = gemini_thinking_level
            os.environ['THINKING_BUDGET'] = str(gemini_thinking_budget)
            os.environ['GPT_REASONING_TOKENS'] = str(or_thinking_tokens)
            os.environ['GPT_EFFORT'] = gpt_effort
        
        # Handle Vertex AI / Google Cloud credentials
        if '@' in model or model.startswith('vertex/'):
            # Prefer an uploaded credentials file; otherwise fall back to config.json.
            google_creds = None
            if google_credentials_file is not None:
                creds_name = os.path.basename(getattr(google_credentials_file, 'filename', '') or 'google_credentials.json')
                if not creds_name.lower().endswith('.json'):
                    creds_name = 'google_credentials.json'
                google_creds = os.path.join(temp_dir, creds_name)
                try:
                    await google_credentials_file.save(google_creds)
                except Exception as e:
                    sys.stderr.write(f"[CONFIG] Failed to save uploaded Google credentials: {e}\n")
                    google_creds = None

            if google_creds is None:
                google_creds = config.get('google_cloud_credentials')

            if google_creds and os.path.exists(google_creds):
                os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = google_creds
                sys.stderr.write(f"[CONFIG] Using Google Cloud credentials: {os.path.basename(google_creds)}\n")
                sys.stderr.flush()

                try:
                    with open(google_creds, 'r', encoding='utf-8') as f:
                        creds_data = json.load(f)
                        project_id = creds_data.get('project_id', 'vertex-ai-project')
                        os.environ['GOOGLE_CLOUD_PROJECT'] = project_id
                        if not api_key:
                            api_key = project_id
                except Exception:
                    pass
        
        # Set API key
        os.environ['API_KEY'] = api_key
        os.environ['OPENAI_API_KEY'] = api_key
        os.environ['OPENAI_OR_Gemini_API_KEY'] = api_key
        
        if 'claude' in model.lower():
            os.environ['ANTHROPIC_API_KEY'] = api_key
        elif 'gemini' in model.lower():
            os.environ['GOOGLE_API_KEY'] = api_key
            os.environ['GEMINI_API_KEY'] = api_key
        
        # Initialize extraction state
        user_id = interaction.user.id
        translation_states[user_id] = {
            'logs': [],
            'show_full': False,
            'stop_requested': False,
            'last_update': 0,
            'pending_update': False
        }
        state = translation_states[user_id]
        
        def log_callback(msg):
            if msg and msg.strip():
                state['logs'].append(msg.strip())
                sys.stderr.write(f"[LOG] {msg.strip()}\n")
                sys.stderr.flush()
                
                import time
                current_time = time.time()
                if current_time - state['last_update'] >= 1.0:
                    state['last_update'] = current_time
                    state['pending_update'] = False
                    asyncio.run_coroutine_threadsafe(update_progress(), bot.loop)
                else:
                    state['pending_update'] = True
        
        def stop_callback():
            return state['stop_requested']
        
        async def periodic_update_check():
            import time
            while user_id in translation_states and not state['stop_requested']:
                await asyncio.sleep(1)
                if state.get('pending_update', False):
                    state['pending_update'] = False
                    state['last_update'] = time.time()
                    await update_progress()
        
        async def update_progress():
            try:
                logs = state['logs']
                if state['show_full']:
                    log_text = '\n'.join(logs)
                    if len(log_text) > 3900:
                        log_text = "..." + log_text[-3900:]
                else:
                    log_text = '\n'.join(logs[-10:])
                    if len(log_text) > 800:
                        log_text = log_text[-800:]
                
                if not log_text:
                    log_text = "Starting..."
                
                embed = discord.Embed(
                    title="📚 Glossary Extraction in Progress",
                    description=f"**Status:** Processing... ({len(logs)} logs)\n\n```{log_text}```",
                    color=discord.Color.blue()
                )
                
                view = LogView(user_id)
                await message.edit(embed=embed, view=view)
            except Exception as e:
                sys.stderr.write(f"[ERROR] Failed to update progress: {e}\n")
                sys.stderr.flush()
        
        await update_progress()
        update_task = asyncio.create_task(periodic_update_check())
        
        def run_extraction():
            _ensure_stable_cwd()

            sys.stderr.write(f"[EXTRACT] Starting glossary extraction for: {input_path}\\n")
            sys.stderr.write(f"[EXTRACT] Temp directory: {temp_dir}\\n")
            sys.stderr.flush()

            # Defensive checks: if these fail, raise an explicit error instead of a generic Errno 2.
            if not os.path.isdir(temp_dir):
                raise FileNotFoundError(f"Temp directory does not exist: {temp_dir}")
            if not os.path.isfile(input_path):
                raise FileNotFoundError(f"Input file does not exist: {input_path}")

            # Avoid chdir() to eliminate CWD-dependent bugs and races.
            # Use absolute output/config paths so the extractor is deterministic.
            output_base = os.path.splitext(os.path.basename(filename))[0] or "glossary"
            output_path = os.path.join(temp_dir, f"{output_base}_glossary.json")

            # Bot-only deployments often don't ship a config.json (it's gitignored).
            # The extractor currently expects a file path, so if one isn't present,
            # create a minimal config in the temp dir and rely on env vars (API_KEY, MODEL, etc.).
            config_path = CONFIG_FILE
            if not os.path.exists(config_path):
                config_path = os.path.join(temp_dir, "config.json")
                try:
                    if not os.path.exists(config_path):
                        with open(config_path, "w", encoding="utf-8") as f:
                            json.dump({}, f)
                except Exception as e:
                    # If we can't write a temp config for any reason, fall back to the original path
                    # so the error message is explicit.
                    sys.stderr.write(f"[EXTRACT] Failed to create temp config.json: {e}\n")
                    sys.stderr.flush()
                    config_path = CONFIG_FILE

            original_argv = sys.argv[:]
            try:
                sys.argv = [
                    'extract_glossary_from_epub.py',
                    '--epub', input_path,
                    '--output', output_path,
                    '--config', config_path
                ]

                glossary_main(log_callback=log_callback, stop_callback=stop_callback)

                sys.stderr.write(f"[EXTRACT] Glossary extraction completed\n")
                sys.stderr.flush()
                return output_path
            finally:
                # Prevent leaking argv changes across commands.
                sys.argv = original_argv
        
        loop = asyncio.get_event_loop()
        extraction_future = loop.run_in_executor(None, run_extraction)
        output_filename = await extraction_future

        update_task.cancel()
        try:
            await update_task
        except asyncio.CancelledError:
            pass
        
        if state['stop_requested']:
            embed = discord.Embed(
                title="⏹️ Extraction Stopped",
                description="Glossary extraction was stopped by user.",
                color=discord.Color.orange()
            )
            await message.edit(embed=embed, view=None)
            return
        
        # Prefer the explicit output path returned by the extractor thread.
        # This avoids brittle "find the right file" scanning.
        output_file_path = None
        output_display_name = None
        is_zip_output = False

        try:
            if output_filename and os.path.isfile(output_filename):
                output_file_path = output_filename
                output_display_name = os.path.basename(output_filename)
        except Exception:
            pass

        glossary_dir = os.path.join(temp_dir, 'Glossary')

        # If a ZIP was requested, prefer zipping the Glossary directory if present;
        # otherwise, zip whatever single output file we produced.
        if send_zip:
            is_zip_output = True
            output_base = os.path.splitext(os.path.basename(filename))[0] or "glossary"
            zip_filename = f"{output_base}_glossary.zip"
            zip_path = os.path.join(temp_dir, zip_filename)

            import zipfile
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                if os.path.exists(glossary_dir) and os.path.isdir(glossary_dir):
                    for root, dirs, files in os.walk(glossary_dir):
                        for file_item in files:
                            file_path = os.path.join(root, file_item)
                            arcname = os.path.relpath(file_path, glossary_dir)
                            zipf.write(file_path, arcname)
                elif output_file_path and os.path.exists(output_file_path):
                    zipf.write(output_file_path, os.path.basename(output_file_path))

            output_file_path = zip_path
            output_display_name = zip_filename

        # Legacy fallback: if we didn't get an explicit output file, try the historical Glossary/ folder.
        if not output_file_path:
            if os.path.exists(glossary_dir) and os.path.isdir(glossary_dir):
                for f in os.listdir(glossary_dir):
                    if f.endswith('.csv'):
                        output_file_path = os.path.join(glossary_dir, f)
                        output_display_name = f
                        break

        async def _dm_or_channel(content: str = None, embed: Optional[discord.Embed] = None, file: Optional[discord.File] = None):
            try:
                await interaction.user.send(content=content, embed=embed, file=file)
                return True
            except discord.Forbidden:
                try:
                    await interaction.channel.send(content=content, embed=embed, file=file)
                    return True
                except Exception:
                    return False
            except Exception:
                return False

        # Send output
        if output_file_path and os.path.exists(output_file_path):
            file_size = os.path.getsize(output_file_path)

            if _should_offload(file_size, send_zip):
                loop = asyncio.get_event_loop()
                download_url = None
                for uploader, name in (( _upload_to_tmpfiles, "tmpfiles"), (_upload_to_tempsh, "temp.sh"), (_upload_to_oracle_par, "oracle"), (_upload_to_gofile, "gofile")):
                    try:
                        download_url = await loop.run_in_executor(
                            None,
                            functools.partial(uploader, output_file_path, output_display_name),
                        )
                        break
                    except Exception as e2:
                        sys.stderr.write(f"[HOST ERROR] {name}: {e2}\n")
                        continue
                if download_url:
                    embed = discord.Embed(
                        title="✅ Glossary Extraction Complete!",
                        description=(
                            f"**File:** {output_display_name}\\n"
                            f"**Size:** {file_size / 1024 / 1024:.2f}MB\\n"
                            f"**Download:** <{download_url}>"
                        ),
                        color=discord.Color.green()
                    )
                    await _dm_or_channel(embed=embed)
                else:
                    await _dm_or_channel(
                        content="Could not upload the glossary output right now. Please try again later."
                    )
            else:
                embed = discord.Embed(
                    title="✅ Glossary Extraction Complete!",
                    description=f"**File:** {output_display_name}\n**Size:** {file_size / 1024:.2f}KB",
                    color=discord.Color.green()
                )
                await _dm_or_channel(embed=embed)

                try:
                    await _dm_or_channel(
                        content=f"Here's your extracted glossary{(' (zipped)' if is_zip_output else '')}!",
                        file=discord.File(output_file_path, filename=output_display_name),
                    )
                except Exception:
                    try:
                        loop = asyncio.get_event_loop()
                        download_url = await loop.run_in_executor(
                            None,
                            functools.partial(_upload_to_gofile, output_file_path, output_display_name),
                        )
                        await _dm_or_channel(
                            content=f"Download: {download_url}",
                        )
                    except Exception:
                        await _dm_or_channel(
                            content="Upload failed and Gofile is unavailable right now."
                        )
        else:
            embed = discord.Embed(
                title="❌ Extraction Failed",
                description="Could not find or prepare glossary output file",
                color=discord.Color.red()
            )
            await _dm_or_channel(embed=embed)
    
    except Exception as e:
        import traceback
        error = f"```\n{traceback.format_exc()[-1000:]}\n```"
        embed = discord.Embed(
            title="❌ Error",
            description=f"{str(e)}\n{error}",
            color=discord.Color.red()
        )
        await message.edit(embed=embed, view=None)
    
    finally:
        # Ensure the background extraction thread has finished before removing temp_dir.
        # If we delete early, the executor thread can crash with FileNotFoundError.
        try:
            if 'state' in locals():
                state['stop_requested'] = True
        except Exception:
            pass

        try:
            if 'update_task' in locals() and update_task:
                update_task.cancel()
        except Exception:
            pass

        try:
            if 'extraction_future' in locals() and extraction_future and not extraction_future.done():
                try:
                    await asyncio.wait_for(extraction_future, timeout=10)
                except Exception:
                    # If it doesn't finish quickly, don't delete the directory out from under it.
                    pass
        except Exception:
            pass

        if user_id in translation_states:
            del translation_states[user_id]

        try:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
        except Exception:
            pass


@bot.tree.command(name="models", description="List available AI models")
async def models(interaction: discord.Interaction):
    """List available models"""
    if GLOSSARION_AVAILABLE:
        model_list = get_model_options()
        
        # Group by provider
        providers = {}
        for model in model_list:
            provider = model.split('-')[0] if '-' in model else model
            if provider not in providers:
                providers[provider] = []
            providers[provider].append(model)
        
        embed = discord.Embed(
            title="🤖 Available Models",
            description="Use with `/translate`",
            color=discord.Color.blue()
        )
        
        for provider, mods in list(providers.items())[:10]:
            text = '\n'.join([f"• `{m}`" for m in mods[:5]])
            if len(mods) > 5:
                text += f"\n• ... +{len(mods) - 5} more"
            embed.add_field(name=provider.upper(), value=text, inline=True)
        
        await _safe_send_message(interaction, embed=embed, ephemeral=_ephemeral(interaction))
    else:
        await _safe_send_message(interaction, "❌ Not available", ephemeral=_ephemeral(interaction))


@bot.tree.command(name="help", description="Show help")
async def help_command(interaction: discord.Interaction):
    """Show help"""
    embed = discord.Embed(
        title="📚 Glossarion Discord Bot",
        description="Translate EPUB/TXT files using AI",
        color=discord.Color.blue()
    )
    
    embed.add_field(
        name="Commands",
        value="`/translate` - Translate file\\n`/extract` - Extract glossary\\n`/models` - List models\\n`/help` - This message\\n\\nUse `send_zip: True` to force ZIP output.",
        inline=False
    )
    
    embed.add_field(
        name="Example",
        value="```\n/translate\n  file: novel.epub\n  api_key: sk-...\n  model: gpt-4\n  target_language: English\n```",
        inline=False
    )
    
    embed.add_field(
        name="Notes",
        value=(
            "• Files over 25MB auto-hosted for 24h (shared 5GB cap)\n"
            "• Uses your Glossarion config\n"
            "• API key not stored"
        ),
        inline=False
    )
    
    await _safe_send_message(interaction, embed=embed, ephemeral=_ephemeral(interaction))


def main():
    """Start bot"""
    token = os.getenv('DISCORD_BOT_TOKEN')
    
    if not token:
        print("❌ DISCORD_BOT_TOKEN not set!")
        print("\nSetup:")
        print("1. Create bot at https://discord.com/developers/applications")
        print("2. Get token from Bot section")
        print("3. Set environment variable:")
        print("   Windows: set DISCORD_BOT_TOKEN=your_token")
        print("   Linux/Mac: export DISCORD_BOT_TOKEN=your_token")
        print("4. Invite bot with 'bot' + 'applications.commands' scopes")
        return
    
    if not GLOSSARION_AVAILABLE:
        print("⚠️ Glossarion not available - translations will fail")
    
    print("🚀 Starting Glossarion Discord Bot...")
    bot.run(token)


if __name__ == "__main__":
    main()
