#!/usr/bin/env python3
"""
Glossarion Discord Bot
Translate files via Discord using your existing Glossarion installation
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

# Add src directory to path
src_dir = os.path.join(os.path.dirname(__file__), "src")
sys.path.insert(0, src_dir)

# Import Glossarion modules
try:
    import TransateKRtoEN
    import extract_glossary_from_epub
    from model_options import get_model_options
    from api_key_encryption import decrypt_config
    # Don't import TranslatorGUI - it requires Qt/GUI. Just use TransateKRtoEN directly
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

# Global storage for translation state
translation_states = {}


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
            await interaction.response.send_message("❌ Translation session expired", ephemeral=True)
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
                await interaction.response.send_message(f"❌ Error: {e}", ephemeral=True)
            except:
                pass
    
    @discord.ui.button(label="Stop Translation", style=discord.ButtonStyle.danger, emoji="⏹️", custom_id="stop_translation")
    async def stop_translation(self, interaction: discord.Interaction, button: discord.ui.Button):
        """Stop the translation process"""
        state = translation_states.get(self.user_id)
        if not state:
            await interaction.response.send_message("❌ Translation session expired", ephemeral=True)
            return
        
        try:
            state['stop_requested'] = True
            button.disabled = True
            button.label = "Stopped"
            button.style = discord.ButtonStyle.secondary
            
            await interaction.response.edit_message(view=self)
            await interaction.followup.send("⏹️ Translation stop requested...", ephemeral=True)
        except Exception as e:
            sys.stderr.write(f"[BUTTON ERROR] {e}\n")
            try:
                await interaction.response.send_message(f"❌ Error: {e}", ephemeral=True)
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


@bot.tree.command(name="translate", description="Translate EPUB or TXT file")
@app_commands.describe(
    api_key="Your API key",
    model="AI model to use (or type custom model name)",
    file="EPUB or TXT file to translate (optional if using url)",
    url="Google Drive or Dropbox link to file (optional if using file attachment)",
    google_credentials_path="Path to Google Cloud credentials JSON (for Vertex AI models)",
    extraction_mode="Text extraction method",
    temperature="Translation temperature 0.0-1.0 (default: 0.3)",
    batch_size="Paragraphs per batch (default: 10)",
    max_output_tokens="Max output tokens (default: 65536)",
    disable_smart_filter="Disable smart glossary filter (default: False)",
    target_language="Target language"
)
@app_commands.choices(extraction_mode=[
    app_commands.Choice(name="Enhanced (html2text)", value="enhanced"),
    app_commands.Choice(name="Standard (BeautifulSoup)", value="standard"),
])
@app_commands.autocomplete(model=model_autocomplete)
async def translate(
    interaction: discord.Interaction,
    api_key: str,
    model: str,
    file: discord.Attachment = None,
    url: str = None,
    google_credentials_path: str = None,
    extraction_mode: str = "enhanced",
    temperature: float = 0.3,
    batch_size: int = 10,
    max_output_tokens: int = 65536,
    disable_smart_filter: bool = False,
    target_language: str = "English"
):
    """Translate file using Glossarion"""
    
    if not GLOSSARION_AVAILABLE:
        await interaction.response.send_message(
            "❌ Glossarion not available", 
            ephemeral=True
        )
        return
    
    # Validate input - must have either file or URL
    if not file and not url:
        await interaction.response.send_message(
            "❌ Please provide either a file attachment or a URL", 
            ephemeral=True
        )
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
    
    # Validate file extension
    if not (filename.endswith('.epub') or filename.endswith('.txt')):
        await interaction.response.send_message(
            "❌ File must be EPUB or TXT format", 
            ephemeral=True
        )
        return
    
    # Initial response (ephemeral - only visible to user)
    embed = discord.Embed(
        title="📚 Translation Started",
        description=f"**File:** {filename}\n**Model:** {model}\n**Target:** {target_language}",
        color=discord.Color.blue()
    )
    await interaction.response.send_message(embed=embed, ephemeral=True)
    message = await interaction.original_response()
    
    # Create temp directory
    temp_dir = tempfile.mkdtemp(prefix=f"discord_translate_{interaction.user.id}_")
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
        
        # Load config
        config = load_config()
        
        # Get system prompt from config
        prompt_profiles = config.get('prompt_profiles', {})
        if 'Universal' in prompt_profiles:
            system_prompt = prompt_profiles['Universal'].replace('{target_lang}', target_language)
        else:
            # Fallback to first available profile or basic prompt
            system_prompt = f"Translate to {target_language}. Preserve all formatting."
        
        # Set model and API key
        os.environ['MODEL'] = model
        os.environ['SYSTEM_PROMPT'] = system_prompt
        os.environ['OUTPUT_DIRECTORY'] = temp_dir
        
        # Set translation parameters
        os.environ['BATCH_TRANSLATION'] = '1'
        os.environ['BATCH_SIZE'] = str(batch_size)
        os.environ['MAX_OUTPUT_TOKENS'] = str(max_output_tokens)
        os.environ['TRANSLATION_TEMPERATURE'] = str(temperature)
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
        
        # Enable automatic glossary generation (set to auto/on by default)
        os.environ['ENABLE_AUTO_GLOSSARY'] = '1'
        # Set glossary parameters (use config if available, otherwise use defaults)
        os.environ['GLOSSARY_MIN_FREQUENCY'] = str(config.get('glossary_min_frequency', 2))
        os.environ['GLOSSARY_MAX_NAMES'] = str(config.get('glossary_max_names', 50))
        os.environ['GLOSSARY_MAX_TITLES'] = str(config.get('glossary_max_titles', 30))
        os.environ['APPEND_GLOSSARY'] = '1'
        os.environ['APPEND_GLOSSARY_PROMPT'] = config.get('append_glossary_prompt', '- Follow this reference glossary for consistent translation (Do not output any raw entries):\n')
        # Set all glossary variables from GUI
        os.environ['GLOSSARY_COMPRESSION_FACTOR'] = str(config.get('glossary_compression_factor', 0.88))
        os.environ['GLOSSARY_FILTER_MODE'] = config.get('glossary_filter_mode', 'all')
        os.environ['GLOSSARY_STRIP_HONORIFICS'] = '1' if config.get('glossary_strip_honorifics', True) else '0'
        os.environ['GLOSSARY_FUZZY_THRESHOLD'] = str(config.get('glossary_fuzzy_threshold', 0.90))
        os.environ['GLOSSARY_MAX_TEXT_SIZE'] = str(config.get('glossary_max_text_size', 50000))
        os.environ['GLOSSARY_MAX_SENTENCES'] = str(config.get('glossary_max_sentences', 200))
        os.environ['GLOSSARY_CHAPTER_SPLIT_THRESHOLD'] = str(config.get('glossary_chapter_split_threshold', 50000))
        os.environ['GLOSSARY_SKIP_FREQUENCY_CHECK'] = '0'  # Enable frequency checking
        os.environ['CONTEXT_WINDOW_SIZE'] = str(config.get('glossary_context_window', 2))
        os.environ['GLOSSARY_USE_LEGACY_CSV'] = '0'  # Use modern JSON format
        os.environ['GLOSSARY_DUPLICATE_KEY_MODE'] = config.get('glossary_duplicate_key_mode', 'auto')
        os.environ['GLOSSARY_DUPLICATE_CUSTOM_FIELD'] = config.get('glossary_duplicate_custom_field', '')
        # Glossary-specific overrides for API settings
        os.environ['GLOSSARY_MAX_OUTPUT_TOKENS'] = str(config.get('glossary_max_output_tokens', max_output_tokens))
        os.environ['GLOSSARY_TEMPERATURE'] = str(config.get('manual_glossary_temperature', 0.1))
        os.environ['GLOSSARY_REQUEST_MERGING_ENABLED'] = '0'  # Disable by default
        os.environ['GLOSSARY_REQUEST_MERGE_COUNT'] = str(config.get('glossary_request_merge_count', 10))
        
        # Set duplicate detection mode to balanced
        os.environ['DUPLICATE_DETECTION_MODE'] = 'balanced'
        
        # Disable Gemini safety filter by default (enabled for Discord bot)
        os.environ['DISABLE_GEMINI_SAFETY'] = 'true'
        
        # Handle Vertex AI / Google Cloud credentials
        if '@' in model or model.startswith('vertex/'):
            # Use provided credentials path, fallback to config if not provided
            google_creds = google_credentials_path if google_credentials_path else config.get('google_cloud_credentials')
            if google_creds and os.path.exists(google_creds):
                os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = google_creds
                sys.stderr.write(f"[CONFIG] Using Google Cloud credentials: {os.path.basename(google_creds)}\n")
                sys.stderr.flush()
                
                # Extract project ID from credentials
                try:
                    with open(google_creds, 'r') as f:
                        creds_data = json.load(f)
                        project_id = creds_data.get('project_id', 'vertex-ai-project')
                        os.environ['GOOGLE_CLOUD_PROJECT'] = project_id
                        if not api_key:
                            api_key = project_id
                except:
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
            sys.stderr.write(f"[TRANSLATE] Starting translation for: {input_path}\n")
            sys.stderr.write(f"[TRANSLATE] Temp directory: {temp_dir}\n")
            sys.stderr.flush()
            
            # CRITICAL: Change to temp directory so TransateKRtoEN creates output there
            original_cwd = os.getcwd()
            try:
                os.chdir(temp_dir)
                sys.stderr.write(f"[TRANSLATE] Changed working directory to: {os.getcwd()}\n")
                sys.stderr.flush()
                
                sys.argv = ['discord_bot.py', input_path]
                result = TransateKRtoEN.main(log_callback=log_callback, stop_callback=stop_callback)
                
                sys.stderr.write(f"[TRANSLATE] Translation completed\n")
                sys.stderr.flush()
                return result
            finally:
                # Restore original working directory
                os.chdir(original_cwd)
                sys.stderr.write(f"[TRANSLATE] Restored working directory to: {os.getcwd()}\n")
                sys.stderr.flush()
        
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, run_translation)
        
        # Cancel the periodic update task
        update_task.cancel()
        try:
            await update_task
        except asyncio.CancelledError:
            pass
        
        # Create a zip file of the entire output directory (even if stopped)
        # This allows users to get partial results
        # TransateKRtoEN creates a subdirectory with the file's basename
        output_base = os.path.splitext(filename)[0]  # Use filename variable, not file.filename
        # Sanitize only problematic characters for filesystem, keep Korean/unicode
        safe_base = output_base.replace('/', '_').replace('\\', '_').replace(':', '_').replace('*', '_').replace('?', '_').replace('"', '_').replace('<', '_').replace('>', '_').replace('|', '_')
        
        # Check for the actual output directory created by TransateKRtoEN
        # It creates a folder named after the input file (without extension)
        output_subdir = os.path.join(temp_dir, safe_base)
        
        sys.stderr.write(f"[ZIP] Creating zip archive of output...\n")
        sys.stderr.write(f"[ZIP] Temp dir: {temp_dir}\n")
        sys.stderr.write(f"[ZIP] Expected output subdir: {output_subdir}\n")
        sys.stderr.write(f"[ZIP] Output subdir exists: {os.path.exists(output_subdir)}\n")
        sys.stderr.flush()
        
        # If output subdir exists, zip from there, otherwise zip from temp_dir root
        if os.path.exists(output_subdir) and os.path.isdir(output_subdir):
            zip_source_dir = output_subdir
            sys.stderr.write(f"[ZIP] Using output subdirectory as source\n")
        else:
            zip_source_dir = temp_dir
            sys.stderr.write(f"[ZIP] Using temp dir as source (no subdirectory found)\n")
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
            # Interaction expired, we'll send as new message later
            pass
        
        try:
            # Create zip archive in background thread to avoid blocking Discord
            def create_zip():
                sys.stderr.write(f"[ZIP] Starting compression...\n")
                sys.stderr.write(f"[ZIP] Zip source directory: {zip_source_dir}\n")
                sys.stderr.flush()
                
                # First, list all files in zip_source_dir
                all_files = []
                for root, dirs, files in os.walk(zip_source_dir):
                    for file_item in files:
                        file_path = os.path.join(root, file_item)
                        all_files.append(file_path)
                        sys.stderr.write(f"[ZIP DEBUG] Found file: {file_path} (size: {os.path.getsize(file_path)} bytes)\n")
                        sys.stderr.flush()
                
                sys.stderr.write(f"[ZIP] Total files found: {len(all_files)}\n")
                sys.stderr.flush()
                
                import zipfile
                files_added = 0
                with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for root, dirs, files in os.walk(zip_source_dir):
                        for file_item in files:
                            file_path = os.path.join(root, file_item)
                            # Skip the zip file itself and the input file
                            if file_item.endswith('.zip'):
                                sys.stderr.write(f"[ZIP] Skipping zip file: {file_item}\n")
                                sys.stderr.flush()
                                continue
                            if file_path == input_path:
                                sys.stderr.write(f"[ZIP] Skipping input file: {file_item}\n")
                                sys.stderr.flush()
                                continue
                            # Use relative path from zip_source_dir, not temp_dir
                            arcname = os.path.relpath(file_path, zip_source_dir)
                            sys.stderr.write(f"[ZIP] Adding: {arcname}\n")
                            sys.stderr.flush()
                            zipf.write(file_path, arcname)
                            files_added += 1
                
                sys.stderr.write(f"[ZIP] Compression complete! Added {files_added} files\n")
                sys.stderr.flush()
                return zip_path
            
            loop = asyncio.get_event_loop()
            zip_result = await loop.run_in_executor(None, create_zip)
            sys.stderr.write(f"[ZIP] Executor returned, checking file...\n")
            sys.stderr.flush()
            
            if os.path.exists(zip_path):
                file_size = os.path.getsize(zip_path)
                max_size = 25 * 1024 * 1024  # 25MB Discord limit
                
                sys.stderr.write(f"[SUCCESS] Created zip: {zip_path}\n")
                sys.stderr.write(f"[SUCCESS] Zip size: {file_size / 1024 / 1024:.2f}MB\n")
                sys.stderr.flush()
                
                if file_size > max_size:
                    title = "⏹️ Translation Stopped - File Too Large" if state['stop_requested'] else "⚠️ File Too Large"
                    description = f"Translation output ({file_size / 1024 / 1024:.2f}MB) exceeds Discord's 25MB limit"
                    embed = discord.Embed(
                        title=title,
                        description=description,
                        color=discord.Color.orange()
                    )
                    try:
                        await message.edit(embed=embed, view=None)
                    except discord.errors.HTTPException:
                        # Interaction expired - send as new followup
                        await interaction.followup.send(embed=embed)
                else:
                    if state['stop_requested']:
                        title = "⏹️ Translation Stopped - Partial Results"
                        description = f"**File:** {zip_filename}\n**Size:** {file_size / 1024 / 1024:.2f}MB\n\nContains partial translation output."
                        color = discord.Color.orange()
                    else:
                        title = "✅ Translation Complete!"
                        description = f"**File:** {zip_filename}\n**Size:** {file_size / 1024 / 1024:.2f}MB\n\nContains all translation outputs and glossaries."
                        color = discord.Color.green()
                    
                    embed = discord.Embed(
                        title=title,
                        description=description,
                        color=color
                    )
                    try:
                        await message.edit(embed=embed, view=None)
                    except discord.errors.HTTPException:
                        # Interaction expired - send as new followup
                        await interaction.followup.send(embed=embed)
                    
                    try:
                        message_text = "Here's your partial translation output (zipped)!" if state['stop_requested'] else "Here's your translation output (zipped)!"
                        await interaction.followup.send(
                            message_text,
                            file=discord.File(zip_path),
                            ephemeral=True
                        )
                    except discord.errors.HTTPException as e:
                        # File too large even though we checked - Discord rejected it
                        await interaction.followup.send(
                            f"Translation complete but zip is too large to upload ({file_size / 1024 / 1024:.2f}MB).\n"
                            f"Please retrieve it from the server.",
                            ephemeral=True
                        )
            else:
                raise FileNotFoundError(f"Zip file not created: {zip_path}")
                
        except Exception as e:
            sys.stderr.write(f"[ERROR] Failed to create zip: {e}\n")
            sys.stderr.flush()
            # List all files in temp_dir for debugging
            for root, dirs, files in os.walk(temp_dir):
                sys.stderr.write(f"[ERROR] {root}: {files}\n")
            sys.stderr.flush()
            
            embed = discord.Embed(
                title="❌ Translation Failed",
                description=f"Could not create output archive: {e}",
                color=discord.Color.red()
            )
            try:
                await message.edit(embed=embed, view=None)
            except discord.errors.HTTPException:
                await interaction.followup.send(embed=embed)
    
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
        
        # Cleanup temp directory
        try:
            shutil.rmtree(temp_dir)
        except:
            pass


@bot.tree.command(name="extract", description="Extract glossary from EPUB or TXT file")
@app_commands.describe(
    api_key="Your API key",
    model="AI model to use (or type custom model name)",
    file="EPUB or TXT file to extract glossary from (optional if using url)",
    url="Google Drive or Dropbox link to file (optional if using file attachment)",
    google_credentials_path="Path to Google Cloud credentials JSON (for Vertex AI models)",
    extraction_mode="Text extraction method",
    temperature="Glossary extraction temperature 0.0-1.0 (default: 0.1)",
    max_output_tokens="Max output tokens (default: 65536)",
    request_merging="Enable request merging to batch API calls (default: False)",
    merge_count="Number of requests to merge when request merging is enabled (default: 10)",
    target_language="Target language for translations"
)
@app_commands.choices(extraction_mode=[
    app_commands.Choice(name="Enhanced (html2text)", value="enhanced"),
    app_commands.Choice(name="Standard (BeautifulSoup)", value="standard"),
])
@app_commands.autocomplete(model=model_autocomplete)
async def extract(
    interaction: discord.Interaction,
    api_key: str,
    model: str,
    file: discord.Attachment = None,
    url: str = None,
    google_credentials_path: str = None,
    extraction_mode: str = "enhanced",
    temperature: float = 0.1,
    max_output_tokens: int = 65536,
    request_merging: bool = False,
    merge_count: int = 10,
    target_language: str = "English"
):
    """Extract glossary from file using Glossarion"""
    
    if not GLOSSARION_AVAILABLE or not glossary_main:
        await interaction.response.send_message(
            "❌ Glossarion glossary extraction not available", 
            ephemeral=True
        )
        return
    
    # Validate input - must have either file or URL
    if not file and not url:
        await interaction.response.send_message(
            "❌ Please provide either a file attachment or a URL", 
            ephemeral=True
        )
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
    
    # Validate file extension
    if not (filename.endswith('.epub') or filename.endswith('.txt')):
        await interaction.response.send_message(
            "❌ File must be EPUB or TXT format", 
            ephemeral=True
        )
        return
    
    # Initial response
    embed = discord.Embed(
        title="📚 Glossary Extraction Started",
        description=f"**File:** {filename}\n**Model:** {model}\n**Target:** {target_language}",
        color=discord.Color.blue()
    )
    await interaction.response.send_message(embed=embed, ephemeral=True)
    message = await interaction.original_response()
    
    # Create temp directory
    temp_dir = tempfile.mkdtemp(prefix=f"discord_extract_{interaction.user.id}_")
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
                                actual_filename = fname_match[0]
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
        
        # Load config
        config = load_config()
        
        # Get glossary prompts from config
        glossary_prompt = config.get('manual_glossary_prompt', '')
        
        # Set model and API key
        os.environ['MODEL'] = model
        os.environ['GLOSSARY_SYSTEM_PROMPT'] = glossary_prompt
        
        # Set translation parameters (same as /translate)
        os.environ['BATCH_TRANSLATION'] = '1'
        os.environ['BATCH_SIZE'] = '10'
        os.environ['MAX_OUTPUT_TOKENS'] = str(max_output_tokens)
        os.environ['GLOSSARY_TEMPERATURE'] = str(temperature)
        os.environ['TRANSLATION_TEMPERATURE'] = str(temperature)
        os.environ['GLOSSARY_MAX_OUTPUT_TOKENS'] = str(max_output_tokens)
        
        # Set extraction mode
        os.environ['TEXT_EXTRACTION_METHOD'] = extraction_mode
        if extraction_mode == 'enhanced':
            os.environ['EXTRACTION_MODE'] = 'enhanced'
            os.environ['ENHANCED_FILTERING'] = 'smart'
            os.environ['ENHANCED_PRESERVE_STRUCTURE'] = '1'
        else:
            os.environ['EXTRACTION_MODE'] = 'smart'
            os.environ['FILE_FILTERING_LEVEL'] = 'smart'
        
        # Set all glossary variables from config (same as /translate)
        os.environ['ENABLE_AUTO_GLOSSARY'] = '1'
        os.environ['GLOSSARY_MIN_FREQUENCY'] = str(config.get('glossary_min_frequency', 2))
        os.environ['GLOSSARY_MAX_NAMES'] = str(config.get('glossary_max_names', 50))
        os.environ['GLOSSARY_MAX_TITLES'] = str(config.get('glossary_max_titles', 30))
        os.environ['GLOSSARY_COMPRESSION_FACTOR'] = str(config.get('glossary_compression_factor', 0.88))
        os.environ['GLOSSARY_FILTER_MODE'] = config.get('glossary_filter_mode', 'all')
        os.environ['GLOSSARY_STRIP_HONORIFICS'] = '1' if config.get('glossary_strip_honorifics', True) else '0'
        os.environ['GLOSSARY_FUZZY_THRESHOLD'] = str(config.get('glossary_fuzzy_threshold', 0.90))
        os.environ['GLOSSARY_MAX_TEXT_SIZE'] = str(config.get('glossary_max_text_size', 50000))
        os.environ['GLOSSARY_MAX_SENTENCES'] = str(config.get('glossary_max_sentences', 200))
        os.environ['GLOSSARY_CHAPTER_SPLIT_THRESHOLD'] = str(config.get('glossary_chapter_split_threshold', 50000))
        os.environ['GLOSSARY_SKIP_FREQUENCY_CHECK'] = '0'
        os.environ['CONTEXT_WINDOW_SIZE'] = str(config.get('glossary_context_window', 2))
        os.environ['GLOSSARY_CONTEXT_LIMIT'] = str(config.get('manual_context_limit', 2))
        os.environ['GLOSSARY_USE_LEGACY_CSV'] = '0'
        os.environ['GLOSSARY_DUPLICATE_KEY_MODE'] = 'skip'
        os.environ['GLOSSARY_DISABLE_HONORIFICS_FILTER'] = '1' if config.get('glossary_disable_honorifics_filter', False) else '0'
        os.environ['GLOSSARY_REQUEST_MERGING_ENABLED'] = '1' if request_merging else '0'
        os.environ['GLOSSARY_REQUEST_MERGE_COUNT'] = str(merge_count)
        os.environ['DISABLE_GEMINI_SAFETY'] = 'true'
        
        # Handle Vertex AI / Google Cloud credentials
        if '@' in model or model.startswith('vertex/'):
            # Use provided credentials path, fallback to config if not provided
            google_creds = google_credentials_path if google_credentials_path else config.get('google_cloud_credentials')
            if google_creds and os.path.exists(google_creds):
                os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = google_creds
                sys.stderr.write(f"[CONFIG] Using Google Cloud credentials: {os.path.basename(google_creds)}\n")
                sys.stderr.flush()
                
                try:
                    with open(google_creds, 'r') as f:
                        creds_data = json.load(f)
                        project_id = creds_data.get('project_id', 'vertex-ai-project')
                        os.environ['GOOGLE_CLOUD_PROJECT'] = project_id
                        if not api_key:
                            api_key = project_id
                except:
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
            sys.stderr.write(f"[EXTRACT] Starting glossary extraction for: {input_path}\n")
            sys.stderr.write(f"[EXTRACT] Temp directory: {temp_dir}\n")
            sys.stderr.flush()
            
            original_cwd = os.getcwd()
            try:
                os.chdir(temp_dir)
                sys.stderr.write(f"[EXTRACT] Changed working directory to: {os.getcwd()}\n")
                sys.stderr.flush()
                
                # Set up sys.argv for glossary extraction
                output_base = os.path.splitext(filename)[0]
                output_path = f"{output_base}_glossary.json"
                
                sys.argv = [
                    'extract_glossary_from_epub.py',
                    '--epub', input_path,
                    '--output', output_path,
                    '--config', CONFIG_FILE
                ]
                
                result = glossary_main(log_callback=log_callback, stop_callback=stop_callback)
                
                sys.stderr.write(f"[EXTRACT] Glossary extraction completed\n")
                sys.stderr.flush()
                return output_path
            finally:
                os.chdir(original_cwd)
                sys.stderr.write(f"[EXTRACT] Restored working directory to: {os.getcwd()}\n")
                sys.stderr.flush()
        
        loop = asyncio.get_event_loop()
        output_filename = await loop.run_in_executor(None, run_extraction)
        
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
        
        # Find the glossary file
        glossary_path = None
        possible_paths = [
            os.path.join(temp_dir, output_filename),
            os.path.join(temp_dir, 'Glossary', output_filename),
            os.path.join(temp_dir, f"{os.path.splitext(filename)[0]}_glossary.json")
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                glossary_path = path
                break
        
        if glossary_path and os.path.exists(glossary_path):
            file_size = os.path.getsize(glossary_path)
            
            embed = discord.Embed(
                title="✅ Glossary Extraction Complete!",
                description=f"**File:** {os.path.basename(glossary_path)}\n**Size:** {file_size / 1024:.2f}KB",
                color=discord.Color.green()
            )
            await message.edit(embed=embed, view=None)
            
            try:
                await interaction.followup.send(
                    f"Here's your extracted glossary!",
                    file=discord.File(glossary_path),
                    ephemeral=True
                )
            except discord.errors.HTTPException as e:
                await interaction.followup.send(
                    f"Glossary complete but file is too large to upload.\n"
                    f"Please retrieve it from the server.",
                    ephemeral=True
                )
        else:
            embed = discord.Embed(
                title="❌ Extraction Failed",
                description="Could not find glossary output file",
                color=discord.Color.red()
            )
            await message.edit(embed=embed, view=None)
    
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
        if user_id in translation_states:
            del translation_states[user_id]
        
        try:
            shutil.rmtree(temp_dir)
        except:
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
        
        await interaction.response.send_message(embed=embed, ephemeral=True)
    else:
        await interaction.response.send_message("❌ Not available", ephemeral=True)


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
        value="`/translate` - Translate file\n`/extract` - Extract glossary\n`/models` - List models\n`/help` - This message",
        inline=False
    )
    
    embed.add_field(
        name="Example",
        value="```\n/translate\n  file: novel.epub\n  api_key: sk-...\n  model: gpt-4\n  target_language: English\n```",
        inline=False
    )
    
    embed.add_field(
        name="Notes",
        value="• Max file size: 25MB\n• Uses your Glossarion config\n• API key not stored",
        inline=False
    )
    
    await interaction.response.send_message(embed=embed, ephemeral=True)


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
