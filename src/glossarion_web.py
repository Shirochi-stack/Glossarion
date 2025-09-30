#!/usr/bin/env python3
"""
Glossarion Web - Gradio Web Interface
AI-powered translation in your browser
"""

import gradio as gr
import os
import sys
import json
import tempfile
import base64
from pathlib import Path

# Import API key encryption/decryption
try:
    from api_key_encryption import decrypt_config
    API_KEY_ENCRYPTION_AVAILABLE = True
except ImportError:
    API_KEY_ENCRYPTION_AVAILABLE = False
    def decrypt_config(config):
        return config  # Fallback: return config as-is

# Import your existing translation modules
try:
    import TransateKRtoEN
    from model_options import get_model_options
    TRANSLATION_AVAILABLE = True
except ImportError:
    TRANSLATION_AVAILABLE = False
    print("⚠️ Translation modules not found")

# Import manga translation modules
try:
    from manga_translator import MangaTranslator
    from unified_api_client import UnifiedClient
    MANGA_TRANSLATION_AVAILABLE = True
except ImportError as e:
    MANGA_TRANSLATION_AVAILABLE = False
    print(f"⚠️ Manga translation modules not found: {e}")


class GlossarionWeb:
    """Web interface for Glossarion translator"""
    
    def __init__(self):
        self.config_file = "config.json"
        self.config = self.load_config()
        # Decrypt API keys for use
        if API_KEY_ENCRYPTION_AVAILABLE:
            self.config = decrypt_config(self.config)
        self.models = get_model_options() if TRANSLATION_AVAILABLE else ["gpt-4", "claude-3-5-sonnet"]
        
        # Default prompts from the GUI (same as translator_gui.py)
        self.default_prompts = {
            "korean": (
                "You are a professional Korean to English novel translator, you must strictly output only English text and HTML tags while following these rules:\n"
                "- Use a natural, comedy-friendly English translation style that captures both humor and readability without losing any original meaning.\n"
                "- Include 100% of the source text - every word, phrase, and sentence must be fully translated without exception.\n"
                "- Retain Korean honorifics and respectful speech markers in romanized form, including but not limited to: -nim, -ssi, -yang, -gun, -isiyeo, -hasoseo. For archaic/classical Korean honorific forms (like 이시여/isiyeo, 하소서/hasoseo), preserve them as-is rather than converting to modern equivalents.\n"
                "- Always localize Korean terminology to proper English equivalents instead of literal translations (examples: 마왕 = Demon King; 마술 = magic).\n"
                "- When translating Korean's pronoun-dropping style, insert pronouns in English only where needed for clarity: prioritize original pronouns as implied or according to the glossary, and only use they/them as a last resort, use I/me for first-person narration, and maintain natural English flow without overusing pronouns just because they're omitted in Korean.\n"
                "- All Korean profanity must be translated to English profanity.\n"
                "- Preserve original intent, and speech tone.\n"
                "- Retain onomatopoeia in Romaji.\n"
                "- Keep original Korean quotation marks (" ", ' ', 「」, 『』) as-is without converting to English quotes.\n"
                "- Every Korean/Chinese/Japanese character must be converted to its English meaning. Examples: The character 생 means 'life/living', 활 means 'active', 관 means 'hall/building' - together 생활관 means Dormitory.\n"
                "- Preserve ALL HTML tags exactly as they appear in the source, including <head>, <title>, <h1>, <h2>, <p>, <br>, <div>, etc.\n"
            ),
            "japanese": (
                "You are a professional Japanese to English novel translator, you must strictly output only English text and HTML tags while following these rules:\n"
                "- Use a natural, comedy-friendly English translation style that captures both humor and readability without losing any original meaning.\n"
                "- Include 100% of the source text - every word, phrase, and sentence must be fully translated without exception.\n"
                "- Retain Japanese honorifics and respectful speech markers in romanized form, including but not limited to: -san, -sama, -chan, -kun, -dono, -sensei, -senpai, -kouhai. For archaic/classical Japanese honorific forms, preserve them as-is rather than converting to modern equivalents.\n"
                "- Always localize Japanese terminology to proper English equivalents instead of literal translations (examples: 魔王 = Demon King; 魔術 = magic).\n"
                "- When translating Japanese's pronoun-dropping style, insert pronouns in English only where needed for clarity: prioritize original pronouns as implied or according to the glossary, and only use they/them as a last resort, use I/me for first-person narration while reflecting the Japanese pronoun's nuance (私/僕/俺/etc.) through speech patterns rather than the pronoun itself, and maintain natural English flow without overusing pronouns just because they're omitted in Japanese.\n"
                "- All Japanese profanity must be translated to English profanity.\n"
                "- Preserve original intent, and speech tone.\n"
                "- Retain onomatopoeia in Romaji.\n"
                "- Keep original Japanese quotation marks (「」, 『』) as-is without converting to English quotes.\n"
                "- Every Korean/Chinese/Japanese character must be converted to its English meaning. Examples: The character 生 means 'life/living', 活 means 'active', 館 means 'hall/building' - together 生活館 means Dormitory.\n"
                "- Preserve ALL HTML tags exactly as they appear in the source, including <head>, <title>, <h1>, <h2>, <p>, <br>, <div>, etc.\n"
            ),
            "chinese": (
                "You are a professional Chinese to English novel translator, you must strictly output only English text and HTML tags while following these rules:\n"
                "- Use a natural, comedy-friendly English translation style that captures both humor and readability without losing any original meaning.\n"
                "- Include 100% of the source text - every word, phrase, and sentence must be fully translated without exception.\n"
                "- Always localize Chinese terminology to proper English equivalents instead of literal translations (examples: 魔王 = Demon King; 魔法 = magic).\n"
                "- When translating Chinese's pronoun-dropping style, insert pronouns in English only where needed for clarity while maintaining natural English flow.\n"
                "- All Chinese profanity must be translated to English profanity.\n"
                "- Preserve original intent, and speech tone.\n"
                "- Retain onomatopoeia in Pinyin.\n"
                "- Keep original Chinese quotation marks (「」, 『』) as-is without converting to English quotes.\n"
                "- Every Korean/Chinese/Japanese character must be converted to its English meaning. Examples: The character 生 means 'life/living', 活 means 'active', 館 means 'hall/building' - together 生活館 means Dormitory.\n"
                "- Preserve ALL HTML tags exactly as they appear in the source, including <head>, <title>, <h1>, <h2>, <p>, <br>, <div>, etc.\n"
            ),
            "Manga_JP": (
                "You are a professional Japanese to English Manga translator.\n"
                "You have both the image of the Manga panel and the extracted text to work with.\n"
                "Output only English text while following these rules: \n\n"

                "VISUAL CONTEXT:\n"
                "- Analyze the character's facial expressions and body language in the image.\n"
                "- Consider the scene's mood and atmosphere.\n"
                "- Note any action or movement depicted.\n"
                "- Use visual cues to determine the appropriate tone and emotion.\n"
                "- USE THE IMAGE to inform your translation choices. The image is not decorative - it contains essential context for accurate translation.\n\n"

                "DIALOGUE REQUIREMENTS:\n"
                "- Match the translation tone to the character's expression.\n"
                "- If a character looks angry, use appropriately intense language.\n"
                "- If a character looks shy or embarrassed, reflect that in the translation.\n"
                "- Keep speech patterns consistent with the character's appearance and demeanor.\n"
                "- Retain honorifics and onomatopoeia in Romaji.\n\n"

                "IMPORTANT: Use both the visual context and text to create the most accurate and natural-sounding translation.\n"
            ), 
            "Manga_KR": (
                "You are a professional Korean to English Manhwa translator.\n"
                "You have both the image of the Manhwa panel and the extracted text to work with.\n"
                "Output only English text while following these rules: \n\n"

                "VISUAL CONTEXT:\n"
                "- Analyze the character's facial expressions and body language in the image.\n"
                "- Consider the scene's mood and atmosphere.\n"
                "- Note any action or movement depicted.\n"
                "- Use visual cues to determine the appropriate tone and emotion.\n"
                "- USE THE IMAGE to inform your translation choices. The image is not decorative - it contains essential context for accurate translation.\n\n"

                "DIALOGUE REQUIREMENTS:\n"
                "- Match the translation tone to the character's expression.\n"
                "- If a character looks angry, use appropriately intense language.\n"
                "- If a character looks shy or embarrassed, reflect that in the translation.\n"
                "- Keep speech patterns consistent with the character's appearance and demeanor.\n"
                "- Retain honorifics and onomatopoeia in Romaji.\n\n"

                "IMPORTANT: Use both the visual context and text to create the most accurate and natural-sounding translation.\n"
            ), 
            "Manga_CN": (
                "You are a professional Chinese to English Manga translator.\n"
                "You have both the image of the Manga panel and the extracted text to work with.\n"
                "Output only English text while following these rules: \n\n"

                "VISUAL CONTEXT:\n"
                "- Analyze the character's facial expressions and body language in the image.\n"
                "- Consider the scene's mood and atmosphere.\n"
                "- Note any action or movement depicted.\n"
                "- Use visual cues to determine the appropriate tone and emotion.\n"
                "- USE THE IMAGE to inform your translation choices. The image is not decorative - it contains essential context for accurate translation.\n"

                "DIALOGUE REQUIREMENTS:\n"
                "- Match the translation tone to the character's expression.\n"
                "- If a character looks angry, use appropriately intense language.\n"
                "- If a character looks shy or embarrassed, reflect that in the translation.\n"
                "- Keep speech patterns consistent with the character's appearance and demeanor.\n"
                "- Retain honorifics and onomatopoeia in Romaji.\n\n"

                "IMPORTANT: Use both the visual context and text to create the most accurate and natural-sounding translation.\n"
            ),
            "Original": "Return everything exactly as seen on the source."
        }
        
        # Load profiles from config, fallback to defaults
        self.profiles = self.config.get('prompt_profiles', self.default_prompts.copy())
        if not self.profiles:
            self.profiles = self.default_prompts.copy()
    
    def load_config(self):
        """Load configuration"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load config: {e}")
        return {}
    
    def save_config(self, config):
        """Save configuration"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
            # Reload config to ensure consistency
            self.config = self.load_config()
        except Exception as e:
            return f"❌ Failed to save config: {e}"
        return "✅ Configuration saved"
    
    def translate_epub(
        self,
        epub_file,
        model,
        api_key,
        profile_name,
        system_prompt,
        temperature,
        max_tokens,
        glossary_file=None,
        progress=gr.Progress()
    ):
        """Translate EPUB file"""
        
        if not TRANSLATION_AVAILABLE:
            return None, "❌ Translation modules not loaded"
        
        if not epub_file:
            return None, "❌ Please upload an EPUB file"
        
        if not api_key:
            return None, "❌ Please provide an API key"
        
        if not profile_name:
            return None, "❌ Please select a translation profile"
        
        try:
            # Progress tracking
            progress(0, desc="Starting translation...")
            
            # Save uploaded file to temp location if needed
            input_path = epub_file.name if hasattr(epub_file, 'name') else epub_file
            epub_base = os.path.splitext(os.path.basename(input_path))[0]
            
            # Use the provided system prompt (user may have edited it)
            translation_prompt = system_prompt if system_prompt else self.profiles.get(profile_name, "")
            
            # Set the input path as a command line argument simulation
            # TransateKRtoEN.main() reads from sys.argv if config doesn't have it
            import sys
            original_argv = sys.argv.copy()
            sys.argv = ['glossarion_web.py', input_path]
            
            # Set environment variables for TransateKRtoEN.main()
            os.environ['INPUT_PATH'] = input_path
            os.environ['MODEL'] = model
            os.environ['TRANSLATION_TEMPERATURE'] = str(temperature)
            os.environ['MAX_OUTPUT_TOKENS'] = str(max_tokens)
            
            # Set API key environment variable
            if 'gpt' in model.lower() or 'openai' in model.lower():
                os.environ['OPENAI_API_KEY'] = api_key
                os.environ['API_KEY'] = api_key
            elif 'claude' in model.lower():
                os.environ['ANTHROPIC_API_KEY'] = api_key
                os.environ['API_KEY'] = api_key
            elif 'gemini' in model.lower():
                os.environ['GOOGLE_API_KEY'] = api_key
                os.environ['API_KEY'] = api_key
            else:
                os.environ['API_KEY'] = api_key
            
            # Set the system prompt
            if translation_prompt:
                # Save to temp profile
                temp_config = self.config.copy()
                temp_config['prompt_profiles'] = temp_config.get('prompt_profiles', {})
                temp_config['prompt_profiles'][profile_name] = translation_prompt
                temp_config['active_profile'] = profile_name
                
                # Save temporarily
                with open(self.config_file, 'w', encoding='utf-8') as f:
                    json.dump(temp_config, f, ensure_ascii=False, indent=2)
            
            progress(0.1, desc="Initializing translation...")
            
            # Create a thread-safe queue for capturing logs
            import queue
            import threading
            log_queue = queue.Queue()
            last_log = ""
            
            def log_callback(msg):
                """Capture log messages without recursion"""
                nonlocal last_log
                if msg and msg.strip():
                    last_log = msg.strip()
                    log_queue.put(msg.strip())
            
            # Monitor logs in a separate thread
            def update_progress():
                while True:
                    try:
                        msg = log_queue.get(timeout=0.5)
                        # Extract progress if available
                        if '✅' in msg or '✓' in msg:
                            progress(0.5, desc=msg[:100])  # Limit message length
                        elif '🔄' in msg or 'Translating' in msg:
                            progress(0.3, desc=msg[:100])
                        else:
                            progress(0.2, desc=msg[:100])
                    except queue.Empty:
                        if last_log:
                            progress(0.2, desc=last_log[:100])
                        continue
                    except:
                        break
            
            progress_thread = threading.Thread(target=update_progress, daemon=True)
            progress_thread.start()
            
            # Call translation function (it reads from environment and config)
            try:
                result = TransateKRtoEN.main(
                    log_callback=log_callback,
                    stop_callback=None
                )
            finally:
                # Restore original sys.argv
                sys.argv = original_argv
                # Stop progress thread
                log_queue.put(None)
            
            progress(1.0, desc="Translation complete!")
            
            # Check for output EPUB in the output directory
            output_dir = epub_base
            if os.path.exists(output_dir):
                # Look for compiled EPUB
                compiled_epub = os.path.join(output_dir, f"{epub_base}_translated.epub")
                if os.path.exists(compiled_epub):
                    return compiled_epub, f"✅ Translation successful!\n\nTranslated: {os.path.basename(compiled_epub)}"
            
            return None, "❌ Translation failed - output file not created"
                
        except Exception as e:
            import traceback
            error_msg = f"❌ Error during translation:\n{str(e)}\n\n{traceback.format_exc()}"
            return None, error_msg
    
    def extract_glossary(
        self,
        epub_file,
        model,
        api_key,
        min_frequency,
        max_names,
        progress=gr.Progress()
    ):
        """Extract glossary from EPUB"""
        
        if not epub_file:
            return None, "❌ Please upload an EPUB file"
        
        try:
            import extract_glossary_from_epub
            
            progress(0, desc="Starting glossary extraction...")
            
            input_path = epub_file.name
            output_path = input_path.replace('.epub', '_glossary.csv')
            
            # Set API key
            if 'gpt' in model.lower():
                os.environ['OPENAI_API_KEY'] = api_key
            elif 'claude' in model.lower():
                os.environ['ANTHROPIC_API_KEY'] = api_key
            
            progress(0.2, desc="Extracting text...")
            
            # Set environment variables for glossary extraction
            os.environ['MODEL'] = model
            os.environ['GLOSSARY_MIN_FREQUENCY'] = str(min_frequency)
            os.environ['GLOSSARY_MAX_NAMES'] = str(max_names)
            
            # Call with proper arguments (check the actual signature)
            result = extract_glossary_from_epub.main(
                log_callback=None,
                stop_callback=None
            )
            
            progress(1.0, desc="Glossary extraction complete!")
            
            if os.path.exists(output_path):
                return output_path, f"✅ Glossary extracted!\n\nSaved to: {os.path.basename(output_path)}"
            else:
                return None, "❌ Glossary extraction failed"
                
        except Exception as e:
            return None, f"❌ Error: {str(e)}"
    
    def translate_manga(
        self,
        image_files,
        model,
        api_key,
        profile_name,
        system_prompt,
        ocr_provider,
        google_creds_path,
        azure_key,
        azure_endpoint,
        enable_bubble_detection,
        enable_inpainting,
        font_size_mode,
        font_size,
        font_multiplier,
        min_font_size,
        max_font_size,
        text_color,
        shadow_enabled,
        shadow_color,
        shadow_offset_x,
        shadow_offset_y,
        shadow_blur,
        bg_opacity,
        bg_style
    ):
        """Translate manga images - GENERATOR that yields (logs, image, cbz_file, status) updates"""
        
        if not MANGA_TRANSLATION_AVAILABLE:
            yield "❌ Manga translation modules not loaded", None, None, gr.update(value="❌ Error", visible=True)
            return
        
        if not image_files:
            yield "❌ Please upload at least one image", None, None, gr.update(value="❌ Error", visible=True)
            return
        
        if not api_key:
            yield "❌ Please provide an API key", None, None, gr.update(value="❌ Error", visible=True)
            return
        
        if ocr_provider == "google" and not google_creds_path:
            yield "❌ Please provide Google Cloud credentials JSON file", None, None, gr.update(value="❌ Error", visible=True)
            return
        
        if ocr_provider == "azure" and (not azure_key or not azure_endpoint):
            yield "❌ Please provide Azure API key and endpoint", None, None, gr.update(value="❌ Error", visible=True)
            return
        
        try:
            
            # Set API key environment variable
            if 'gpt' in model.lower() or 'openai' in model.lower():
                os.environ['OPENAI_API_KEY'] = api_key
            elif 'claude' in model.lower():
                os.environ['ANTHROPIC_API_KEY'] = api_key
            elif 'gemini' in model.lower():
                os.environ['GOOGLE_API_KEY'] = api_key
            
            # Set Google Cloud credentials if provided
            if ocr_provider == "google" and google_creds_path:
                os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = google_creds_path.name
            
            # Set Azure credentials if provided
            if ocr_provider == "azure":
                os.environ['AZURE_VISION_KEY'] = azure_key
                os.environ['AZURE_VISION_ENDPOINT'] = azure_endpoint
            
            # Apply text visibility settings to config
            # Convert hex color to RGB tuple
            def hex_to_rgb(hex_color):
                hex_color = hex_color.lstrip('#')
                return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
            
            text_rgb = hex_to_rgb(text_color)
            shadow_rgb = hex_to_rgb(shadow_color)
            
            self.config['manga_font_size_mode'] = font_size_mode
            self.config['manga_font_size'] = int(font_size)
            self.config['manga_font_size_multiplier'] = float(font_multiplier)
            self.config['manga_max_font_size'] = int(max_font_size)
            self.config['manga_text_color'] = list(text_rgb)
            self.config['manga_shadow_enabled'] = bool(shadow_enabled)
            self.config['manga_shadow_color'] = list(shadow_rgb)
            self.config['manga_shadow_offset_x'] = int(shadow_offset_x)
            self.config['manga_shadow_offset_y'] = int(shadow_offset_y)
            self.config['manga_shadow_blur'] = int(shadow_blur)
            self.config['manga_bg_opacity'] = int(bg_opacity)
            self.config['manga_bg_style'] = bg_style
            
            # Also update nested manga_settings structure
            if 'manga_settings' not in self.config:
                self.config['manga_settings'] = {}
            if 'rendering' not in self.config['manga_settings']:
                self.config['manga_settings']['rendering'] = {}
            if 'font_sizing' not in self.config['manga_settings']:
                self.config['manga_settings']['font_sizing'] = {}
            
            self.config['manga_settings']['rendering']['auto_min_size'] = int(min_font_size)
            self.config['manga_settings']['font_sizing']['min_size'] = int(min_font_size)
            self.config['manga_settings']['rendering']['auto_max_size'] = int(max_font_size)
            self.config['manga_settings']['font_sizing']['max_size'] = int(max_font_size)
            
            # Prepare output directory
            output_dir = tempfile.mkdtemp(prefix="manga_translated_")
            translated_files = []
            cbz_mode = False
            cbz_output_path = None
            
            # Check if any file is a CBZ/ZIP archive
            import zipfile
            files_to_process = image_files if isinstance(image_files, list) else [image_files]
            extracted_images = []
            
            for file in files_to_process:
                file_path = file.name if hasattr(file, 'name') else file
                if file_path.lower().endswith(('.cbz', '.zip')):
                    # Extract CBZ
                    cbz_mode = True
                    translation_logs.append(f"📚 Extracting CBZ: {os.path.basename(file_path)}")
                    extract_dir = tempfile.mkdtemp(prefix="cbz_extract_")
                    
                    try:
                        with zipfile.ZipFile(file_path, 'r') as zip_ref:
                            zip_ref.extractall(extract_dir)
                        
                        # Find all image files in extracted directory
                        import glob
                        for ext in ['*.png', '*.jpg', '*.jpeg', '*.webp', '*.bmp', '*.gif']:
                            extracted_images.extend(glob.glob(os.path.join(extract_dir, '**', ext), recursive=True))
                        
                        # Sort naturally (by filename)
                        extracted_images.sort()
                        translation_logs.append(f"✅ Extracted {len(extracted_images)} images from CBZ")
                        
                        # Prepare CBZ output path
                        cbz_output_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(file_path))[0]}_translated.cbz")
                    except Exception as e:
                        translation_logs.append(f"❌ Error extracting CBZ: {str(e)}")
                else:
                    # Regular image file
                    extracted_images.append(file_path)
            
            # Use extracted images if CBZ was processed, otherwise use original files
            if extracted_images:
                # Create mock file objects for extracted images
                class MockFile:
                    def __init__(self, path):
                        self.name = path
                
                files_to_process = [MockFile(img) for img in extracted_images]
            
            total_images = len(files_to_process)
            
            # Merge web app config with SimpleConfig for MangaTranslator
            # This includes all the text visibility settings we just set
            merged_config = self.config.copy()
            
            # Override with web-specific settings
            merged_config['model'] = model
            merged_config['active_profile'] = profile_name
            
            # Update manga_settings
            if 'manga_settings' not in merged_config:
                merged_config['manga_settings'] = {}
            if 'ocr' not in merged_config['manga_settings']:
                merged_config['manga_settings']['ocr'] = {}
            if 'inpainting' not in merged_config['manga_settings']:
                merged_config['manga_settings']['inpainting'] = {}
            
            merged_config['manga_settings']['ocr']['provider'] = ocr_provider
            merged_config['manga_settings']['ocr']['bubble_detection_enabled'] = enable_bubble_detection
            merged_config['manga_settings']['inpainting']['method'] = 'local' if enable_inpainting else 'none'
            
            # Create a simple config object for MangaTranslator
            class SimpleConfig:
                def __init__(self, cfg):
                    self.config = cfg
                
                def get(self, key, default=None):
                    return self.config.get(key, default)
            
            # Create mock GUI object with necessary attributes
            class MockGUI:
                def __init__(self, config, profile_name, system_prompt, max_output_tokens, api_key, model):
                    self.config = config
                    # Add profile_var mock for MangaTranslator compatibility
                    class ProfileVar:
                        def __init__(self, profile):
                            self.profile = profile
                        def get(self):
                            return self.profile
                    self.profile_var = ProfileVar(profile_name)
                    # Add prompt_profiles BOTH to config AND as attribute (manga_translator checks both)
                    if 'prompt_profiles' not in self.config:
                        self.config['prompt_profiles'] = {}
                    self.config['prompt_profiles'][profile_name] = system_prompt
                    # Also set as direct attribute for line 4653 check
                    self.prompt_profiles = self.config['prompt_profiles']
                    # Add max_output_tokens as direct attribute (line 299 check)
                    self.max_output_tokens = max_output_tokens
                    # Add mock GUI attributes that MangaTranslator expects
                    class MockVar:
                        def __init__(self, val):
                            self.val = val
                        def get(self):
                            return self.val
                    self.delay_entry = MockVar(config.get('delay', 2.0))
                    self.trans_temp = MockVar(config.get('translation_temperature', 0.3))
                    self.contextual_var = MockVar(config.get('contextual', False))
                    self.trans_history = MockVar(config.get('translation_history_limit', 2))
                    self.translation_history_rolling_var = MockVar(config.get('translation_history_rolling', False))
                    self.token_limit_disabled = config.get('token_limit_disabled', False)
                    self.token_limit_entry = MockVar(config.get('token_limit', 200000))
                    # Add API key and model for custom-api OCR provider
                    self.api_key_entry = MockVar(api_key)
                    self.model_var = MockVar(model)
            
            simple_config = SimpleConfig(merged_config)
            # Get max_output_tokens from config or use from web app config
            web_max_tokens = merged_config.get('max_output_tokens', 16000)
            mock_gui = MockGUI(simple_config.config, profile_name, system_prompt, web_max_tokens, api_key, model)
            
            # Setup OCR configuration
            ocr_config = {
                'provider': ocr_provider
            }
            
            if ocr_provider == 'google':
                ocr_config['google_credentials_path'] = google_creds_path.name if google_creds_path else None
            elif ocr_provider == 'azure':
                ocr_config['azure_key'] = azure_key
                ocr_config['azure_endpoint'] = azure_endpoint
            
            # Create UnifiedClient for translation API calls
            try:
                unified_client = UnifiedClient(
                    api_key=api_key,
                    model=model,
                    output_dir=output_dir
                )
            except Exception as e:
                error_log = f"❌ Failed to initialize API client: {str(e)}"
                yield error_log, None, None, gr.update(value=error_log, visible=True)
                return
            
            # Log storage - will be yielded as live updates
            translation_logs = []
            last_yield_log_count = [0]  # Track when we last yielded
            last_yield_time = [0]  # Track last yield time
            
            # Track current image being processed
            current_image_idx = [0]
            
            import time
            
            def should_yield_logs():
                """Check if we should yield log updates (every 2 logs or 1 second)"""
                current_time = time.time()
                log_count_diff = len(translation_logs) - last_yield_log_count[0]
                time_diff = current_time - last_yield_time[0]
                
                # Yield if 2+ new logs OR 1+ seconds passed
                return log_count_diff >= 2 or time_diff >= 1.0
            
            def capture_log(msg, level="info"):
                """Capture logs - caller will yield periodically"""
                if msg and msg.strip():
                    log_msg = msg.strip()
                    translation_logs.append(log_msg)
            
            # Initialize timing
            last_yield_time[0] = time.time()
            
            # Create MangaTranslator instance
            try:
                translator = MangaTranslator(
                    ocr_config=ocr_config,
                    unified_client=unified_client,
                    main_gui=mock_gui,
                    log_callback=capture_log
                )
            except Exception as e:
                error_log = f"❌ Failed to initialize manga translator: {str(e)}"
                yield error_log, None, None, gr.update(value=error_log, visible=True)
                return
            
            # Process each image with real progress tracking
            for idx, img_file in enumerate(files_to_process, 1):
                try:
                    # Update current image index for log capture
                    current_image_idx[0] = idx
                    
                    # Calculate progress range for this image
                    start_progress = (idx - 1) / total_images
                    end_progress = idx / total_images
                    
                    input_path = img_file.name if hasattr(img_file, 'name') else img_file
                    output_path = os.path.join(output_dir, f"translated_{os.path.basename(input_path)}")
                    filename = os.path.basename(input_path)
                    
                    # Log start of processing and YIELD update
                    start_msg = f"🎨 [{idx}/{total_images}] Starting: {filename}"
                    translation_logs.append(start_msg)
                    translation_logs.append(f"Image path: {input_path}")
                    translation_logs.append(f"Processing with OCR: {ocr_provider}, Model: {model}")
                    translation_logs.append("-" * 60)
                    
                    # Yield initial log update
                    last_yield_log_count[0] = len(translation_logs)
                    last_yield_time[0] = time.time()
                    yield "\n".join(translation_logs), None, None, gr.update(visible=False)
                    
                    # Start processing in a thread so we can yield logs periodically
                    import threading
                    processing_complete = [False]
                    result_container = [None]
                    
                    def process_wrapper():
                        result_container[0] = translator.process_image(
                            image_path=input_path,
                            output_path=output_path,
                            batch_index=idx,
                            batch_total=total_images
                        )
                        processing_complete[0] = True
                    
                    # Start processing in background
                    process_thread = threading.Thread(target=process_wrapper, daemon=True)
                    process_thread.start()
                    
                    # Poll for log updates while processing
                    while not processing_complete[0]:
                        time.sleep(0.5)  # Check every 0.5 seconds
                        if should_yield_logs():
                            last_yield_log_count[0] = len(translation_logs)
                            last_yield_time[0] = time.time()
                            yield "\n".join(translation_logs), None, None, gr.update(visible=False)
                    
                    # Wait for thread to complete
                    process_thread.join(timeout=1)
                    result = result_container[0]
                    
                    if result.get('success'):
                        # Use the output path from the result
                        final_output = result.get('output_path', output_path)
                        if os.path.exists(final_output):
                            translated_files.append(final_output)
                            translation_logs.append(f"✅ Image {idx}/{total_images} COMPLETE: {filename} | Total: {len(translated_files)}/{total_images} done")
                            translation_logs.append("")
                            # Yield progress update with completed image
                            yield "\n".join(translation_logs), gr.update(value=final_output, visible=True), None, gr.update(visible=False)
                        else:
                            translation_logs.append(f"⚠️ Image {idx}/{total_images}: Output file missing for {filename}")
                            translation_logs.append(f"⚠️ Warning: Output file not found for image {idx}")
                            translation_logs.append("")
                            # Yield progress update
                            yield "\n".join(translation_logs), None, None, gr.update(visible=False)
                    else:
                        errors = result.get('errors', [])
                        error_msg = errors[0] if errors else 'Unknown error'
                        translation_logs.append(f"❌ Image {idx}/{total_images} FAILED: {error_msg[:50]}")
                        translation_logs.append(f"⚠️ Error on image {idx}: {error_msg}")
                        translation_logs.append("")
                        # Yield progress update
                        yield "\n".join(translation_logs), None, None, gr.update(visible=False)
                        
                        # If translation failed, save original with error overlay
                        from PIL import Image as PILImage, ImageDraw, ImageFont
                        img = PILImage.open(input_path)
                        draw = ImageDraw.Draw(img)
                        # Add error message
                        draw.text((10, 10), f"Translation Error: {error_msg[:50]}", fill="red")
                        img.save(output_path)
                        translated_files.append(output_path)
                    
                except Exception as e:
                    import traceback
                    error_trace = traceback.format_exc()
                    translation_logs.append(f"❌ Image {idx}/{total_images} ERROR: {str(e)[:60]}")
                    translation_logs.append(f"❌ Exception on image {idx}: {str(e)}")
                    print(f"Manga translation error for {input_path}:\n{error_trace}")
                    
                    # Save original on error
                    try:
                        from PIL import Image as PILImage
                        img = PILImage.open(input_path)
                        img.save(output_path)
                        translated_files.append(output_path)
                    except:
                        pass
                    continue
            
            # Add completion message
            translation_logs.append("\n" + "="*60)
            translation_logs.append(f"✅ ALL COMPLETE! Successfully translated {len(translated_files)}/{total_images} images")
            translation_logs.append("="*60)
            
            # If CBZ mode, compile translated images into CBZ archive
            final_output_for_display = None
            if cbz_mode and cbz_output_path and translated_files:
                translation_logs.append("\n📦 Compiling translated images into CBZ archive...")
                try:
                    with zipfile.ZipFile(cbz_output_path, 'w', zipfile.ZIP_DEFLATED) as cbz:
                        for img_path in translated_files:
                            # Preserve original filename structure
                            arcname = os.path.basename(img_path).replace("translated_", "")
                            cbz.write(img_path, arcname)
                    
                    translation_logs.append(f"✅ CBZ archive created: {os.path.basename(cbz_output_path)}")
                    translation_logs.append(f"📁 Archive location: {cbz_output_path}")
                    final_output_for_display = cbz_output_path
                except Exception as e:
                    translation_logs.append(f"❌ Error creating CBZ: {str(e)}")
            
            # Build final status
            final_status_lines = []
            if translated_files:
                final_status_lines.append(f"✅ Successfully translated {len(translated_files)}/{total_images} image(s)!")
                if cbz_mode and cbz_output_path:
                    final_status_lines.append(f"\n📦 CBZ Output: {cbz_output_path}")
                else:
                    final_status_lines.append(f"\nOutput directory: {output_dir}")
            else:
                final_status_lines.append("❌ Translation failed - no images were processed")
            
            final_status_text = "\n".join(final_status_lines)
            
            # Final yield with complete logs, image, CBZ, and final status
            # Format: (logs_textbox, output_image, cbz_file, status_textbox)
            if translated_files:
                # If CBZ mode, show CBZ file for download; otherwise show first image
                if cbz_mode and cbz_output_path and os.path.exists(cbz_output_path):
                    yield "\n".join(translation_logs), gr.update(value=translated_files[0], visible=True), gr.update(value=cbz_output_path, visible=True), gr.update(value=final_status_text, visible=True)
                else:
                    yield "\n".join(translation_logs), gr.update(value=translated_files[0], visible=True), gr.update(visible=False), gr.update(value=final_status_text, visible=True)
            else:
                yield "\n".join(translation_logs), gr.update(visible=False), gr.update(visible=False), gr.update(value=final_status_text, visible=True)
                
        except Exception as e:
            import traceback
            error_msg = f"❌ Error during manga translation:\n{str(e)}\n\n{traceback.format_exc()}"
            yield error_msg, gr.update(visible=False), gr.update(visible=False), gr.update(value=error_msg, visible=True)
    
    def create_interface(self):
        """Create Gradio interface"""
        
        # Load and encode icon as base64
        icon_base64 = ""
        icon_path = "Halgakos.png" if os.path.exists("Halgakos.png") else "Halgakos.ico"
        if os.path.exists(icon_path):
            with open(icon_path, "rb") as f:
                icon_base64 = base64.b64encode(f.read()).decode()
        
        # Custom CSS to hide Gradio footer and add favicon
        custom_css = """
        footer {display: none !important;}
        .gradio-container {min-height: 100vh;}
        """
        
        with gr.Blocks(
            title="Glossarion - AI Translation",
            theme=gr.themes.Soft(),
            css=custom_css
        ) as app:
            
            # Add custom HTML with favicon link and title with icon
            icon_img_tag = f'<img src="data:image/png;base64,{icon_base64}" alt="Glossarion">' if icon_base64 else ''
            
            gr.HTML(f"""
            <link rel="icon" type="image/x-icon" href="file/Halgakos.ico">
            <link rel="shortcut icon" type="image/x-icon" href="file/Halgakos.ico">
            <style>
                .title-with-icon {{
                    display: flex;
                    align-items: center;
                    gap: 15px;
                    margin-bottom: 10px;
                }}
                .title-with-icon img {{
                    width: 48px;
                    height: 48px;
                }}
            </style>
            <div class="title-with-icon">
                {icon_img_tag}
                <h1>Glossarion - AI-Powered Translation</h1>
            </div>
            """)
            
            gr.Markdown("""
            Translate novels and books using advanced AI models (GPT-5, Claude, etc.)
            """)
            
            with gr.Tabs():
                # Manga Translation Tab - DEFAULT/FIRST
                with gr.Tab("🎨 Manga Translation"):
                    with gr.Row():
                        with gr.Column():
                            manga_images = gr.File(
                                label="🖼️ Upload Manga Images or CBZ",
                                file_types=[".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".cbz", ".zip"],
                                file_count="multiple"
                            )
                            
                            translate_manga_btn = gr.Button(
                                "🚀 Translate Manga",
                                variant="primary",
                                size="lg"
                            )
                            
                            manga_model = gr.Dropdown(
                                choices=self.models,
                                value=self.config.get('model', 'gpt-4-turbo'),
                                label="🤖 AI Model"
                            )
                            
                            manga_api_key = gr.Textbox(
                                label="🔑 API Key",
                                type="password",
                                placeholder="Enter your API key",
                                value=self.config.get('api_key', '')  # Pre-fill from config
                            )
                            
                            # Filter manga-specific profiles
                            manga_profile_choices = [k for k in self.profiles.keys() if k.startswith('Manga_')]
                            if not manga_profile_choices:
                                manga_profile_choices = list(self.profiles.keys())  # Fallback to all
                            
                            default_manga_profile = "Manga_JP" if "Manga_JP" in self.profiles else manga_profile_choices[0] if manga_profile_choices else ""
                            
                            manga_profile = gr.Dropdown(
                                choices=manga_profile_choices,
                                value=default_manga_profile,
                                label="📝 Translation Profile"
                            )
                            
                            # Editable manga system prompt
                            manga_system_prompt = gr.Textbox(
                                label="Manga System Prompt (Translation Instructions)",
                                lines=8,
                                max_lines=15,
                                interactive=True,
                                placeholder="Select a manga profile to load translation instructions...",
                                value=self.profiles.get(default_manga_profile, '') if default_manga_profile else ''
                            )
                            
                            with gr.Accordion("⚙️ OCR Settings", open=False):
                                ocr_provider = gr.Radio(
                                    choices=["google", "azure", "custom-api"],
                                    value="custom-api",  # Default to custom-api
                                    label="OCR Provider"
                                )
                                
                                google_creds = gr.File(
                                    label="Google Cloud Credentials JSON (if using Google)",
                                    file_types=[".json"]
                                )
                                
                                azure_key = gr.Textbox(
                                    label="Azure Vision API Key (if using Azure)",
                                    type="password",
                                    placeholder="Enter Azure API key"
                                )
                                
                                azure_endpoint = gr.Textbox(
                                    label="Azure Vision Endpoint (if using Azure)",
                                    placeholder="https://your-resource.cognitiveservices.azure.com/"
                                )
                                
                                bubble_detection = gr.Checkbox(
                                    label="Enable Bubble Detection",
                                    value=True
                                )
                                
                                inpainting = gr.Checkbox(
                                    label="Enable Text Removal (Inpainting)",
                                    value=True
                                )
                            
                            with gr.Accordion("✨ Text Visibility Settings", open=False):
                                gr.Markdown("### Font Settings")
                                
                                font_size_mode = gr.Radio(
                                    choices=["auto", "fixed", "multiplier"],
                                    value=self.config.get('manga_font_size_mode', 'auto'),
                                    label="Font Size Mode"
                                )
                                
                                font_size = gr.Slider(
                                    minimum=0,
                                    maximum=72,
                                    value=self.config.get('manga_font_size', 24),
                                    step=1,
                                    label="Fixed Font Size (0=auto, used when mode=fixed)"
                                )
                                
                                font_multiplier = gr.Slider(
                                    minimum=0.5,
                                    maximum=2.0,
                                    value=self.config.get('manga_font_size_multiplier', 1.0),
                                    step=0.1,
                                    label="Font Size Multiplier (when mode=multiplier)"
                                )
                                
                                min_font_size = gr.Slider(
                                    minimum=0,
                                    maximum=100,
                                    value=self.config.get('manga_settings', {}).get('rendering', {}).get('auto_min_size', 12),
                                    step=1,
                                    label="Minimum Font Size (0=no limit)"
                                )
                                
                                max_font_size = gr.Slider(
                                    minimum=20,
                                    maximum=100,
                                    value=self.config.get('manga_max_font_size', 48),
                                    step=1,
                                    label="Maximum Font Size"
                                )
                                
                                gr.Markdown("### Text Color")
                                
                                text_color_rgb = gr.ColorPicker(
                                    label="Font Color",
                                    value="#000000"  # Default black
                                )
                                
                                gr.Markdown("### Shadow Settings")
                                
                                shadow_enabled = gr.Checkbox(
                                    label="Enable Text Shadow",
                                    value=self.config.get('manga_shadow_enabled', True)
                                )
                                
                                shadow_color = gr.ColorPicker(
                                    label="Shadow Color",
                                    value="#FFFFFF"  # Default white
                                )
                                
                                shadow_offset_x = gr.Slider(
                                    minimum=-10,
                                    maximum=10,
                                    value=self.config.get('manga_shadow_offset_x', 2),
                                    step=1,
                                    label="Shadow Offset X"
                                )
                                
                                shadow_offset_y = gr.Slider(
                                    minimum=-10,
                                    maximum=10,
                                    value=self.config.get('manga_shadow_offset_y', 2),
                                    step=1,
                                    label="Shadow Offset Y"
                                )
                                
                                shadow_blur = gr.Slider(
                                    minimum=0,
                                    maximum=10,
                                    value=self.config.get('manga_shadow_blur', 0),
                                    step=1,
                                    label="Shadow Blur"
                                )
                                
                                gr.Markdown("### Background Settings")
                                
                                bg_opacity = gr.Slider(
                                    minimum=0,
                                    maximum=255,
                                    value=self.config.get('manga_bg_opacity', 130),
                                    step=1,
                                    label="Background Opacity"
                                )
                                
                                bg_style = gr.Radio(
                                    choices=["box", "circle", "wrap"],
                                    value=self.config.get('manga_bg_style', 'circle'),
                                    label="Background Style"
                                )
                        
                        with gr.Column():
                            # Add logo and loading message at top
                            with gr.Row():
                                gr.Image(
                                    value="Halgakos.png",
                                    label=None,
                                    show_label=False,
                                    width=80,
                                    height=80,
                                    interactive=False,
                                    show_download_button=False,
                                    container=False
                                )
                                status_message = gr.Markdown(
                                    value="### Ready to translate\nUpload an image and click 'Translate Manga' to begin.",
                                    visible=True
                                )
                            manga_logs = gr.Textbox(
                                label="📋 Translation Logs",
                                lines=20,
                                max_lines=30,
                                value="Ready to translate. Click 'Translate Manga' to begin.",
                                visible=True,
                                interactive=False
                            )
                            
                            manga_output_image = gr.Image(label="📷 Translated Image Preview", visible=False)
                            manga_cbz_output = gr.File(label="📦 Download Translated CBZ", visible=False)
                            manga_status = gr.Textbox(
                                label="Final Status",
                                lines=8,
                                max_lines=15,
                                visible=False
                            )
                    
                    # Update manga system prompt when profile changes
                    def update_manga_system_prompt(profile_name):
                        return self.profiles.get(profile_name, "")
                    
                    manga_profile.change(
                        fn=update_manga_system_prompt,
                        inputs=[manga_profile],
                        outputs=[manga_system_prompt]
                    )
                    
                    translate_manga_btn.click(
                        fn=self.translate_manga,
                        inputs=[
                            manga_images,
                            manga_model,
                            manga_api_key,
                            manga_profile,
                            manga_system_prompt,
                            ocr_provider,
                            google_creds,
                            azure_key,
                            azure_endpoint,
                            bubble_detection,
                            inpainting,
                            font_size_mode,
                            font_size,
                            font_multiplier,
                            min_font_size,
                            max_font_size,
                            text_color_rgb,
                            shadow_enabled,
                            shadow_color,
                            shadow_offset_x,
                            shadow_offset_y,
                            shadow_blur,
                            bg_opacity,
                            bg_style
                        ],
                        outputs=[manga_logs, manga_output_image, manga_cbz_output, manga_status]
                    )
                
                # Manga Settings Tab - NEW
                with gr.Tab("🎬 Manga Settings"):
                    gr.Markdown("### Advanced Manga Translation Settings")
                    gr.Markdown("Configure bubble detection, inpainting, preprocessing, and rendering options.")
                    
                    with gr.Accordion("🕹️ Bubble Detection & Inpainting", open=True):
                        gr.Markdown("#### Bubble Detection")
                        
                        detector_type = gr.Radio(
                            choices=["rtdetr_onnx", "rtdetr", "yolo"],
                            value=self.config.get('manga_settings', {}).get('ocr', {}).get('detector_type', 'rtdetr_onnx'),
                            label="Detector Type",
                            interactive=True
                        )
                        
                        rtdetr_confidence = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=self.config.get('manga_settings', {}).get('ocr', {}).get('rtdetr_confidence', 0.3),
                            step=0.05,
                            label="RT-DETR Confidence Threshold",
                            interactive=True
                        )
                        
                        bubble_confidence = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=self.config.get('manga_settings', {}).get('ocr', {}).get('bubble_confidence', 0.3),
                            step=0.05,
                            label="YOLO Bubble Confidence Threshold",
                            interactive=True
                        )
                        
                        detect_text_bubbles = gr.Checkbox(
                            label="Detect Text Bubbles",
                            value=self.config.get('manga_settings', {}).get('ocr', {}).get('detect_text_bubbles', True)
                        )
                        
                        detect_empty_bubbles = gr.Checkbox(
                            label="Detect Empty Bubbles",
                            value=self.config.get('manga_settings', {}).get('ocr', {}).get('detect_empty_bubbles', True)
                        )
                        
                        detect_free_text = gr.Checkbox(
                            label="Detect Free Text (outside bubbles)",
                            value=self.config.get('manga_settings', {}).get('ocr', {}).get('detect_free_text', True)
                        )
                        
                        gr.Markdown("#### Inpainting")
                        
                        local_inpaint_method = gr.Radio(
                            choices=["anime_onnx", "anime", "lama", "lama_onnx", "aot", "aot_onnx"],
                            value=self.config.get('manga_settings', {}).get('inpainting', {}).get('local_method', 'anime_onnx'),
                            label="Local Inpainting Model",
                            interactive=True
                        )
                        
                        with gr.Row():
                            download_models_btn = gr.Button(
                                "📥 Download Models",
                                variant="secondary",
                                size="sm"
                            )
                            load_models_btn = gr.Button(
                                "📂 Load Models",
                                variant="secondary",
                                size="sm"
                            )
                        
                        gr.Markdown("#### Mask Dilation")
                        
                        auto_iterations = gr.Checkbox(
                            label="Auto Iterations (Recommended)",
                            value=self.config.get('manga_settings', {}).get('auto_iterations', True)
                        )
                        
                        mask_dilation = gr.Slider(
                            minimum=0,
                            maximum=20,
                            value=self.config.get('manga_settings', {}).get('mask_dilation', 0),
                            step=1,
                            label="General Mask Dilation",
                            interactive=True
                        )
                        
                        text_bubble_dilation = gr.Slider(
                            minimum=0,
                            maximum=20,
                            value=self.config.get('manga_settings', {}).get('text_bubble_dilation_iterations', 2),
                            step=1,
                            label="Text Bubble Dilation Iterations",
                            interactive=True
                        )
                        
                        empty_bubble_dilation = gr.Slider(
                            minimum=0,
                            maximum=20,
                            value=self.config.get('manga_settings', {}).get('empty_bubble_dilation_iterations', 3),
                            step=1,
                            label="Empty Bubble Dilation Iterations",
                            interactive=True
                        )
                        
                        free_text_dilation = gr.Slider(
                            minimum=0,
                            maximum=20,
                            value=self.config.get('manga_settings', {}).get('free_text_dilation_iterations', 3),
                            step=1,
                            label="Free Text Dilation Iterations",
                            interactive=True
                        )
                    
                    with gr.Accordion("🖌️ Image Preprocessing", open=False):
                        preprocessing_enabled = gr.Checkbox(
                            label="Enable Preprocessing",
                            value=self.config.get('manga_settings', {}).get('preprocessing', {}).get('enabled', False)
                        )
                        
                        auto_detect_quality = gr.Checkbox(
                            label="Auto Detect Image Quality",
                            value=self.config.get('manga_settings', {}).get('preprocessing', {}).get('auto_detect_quality', True)
                        )
                        
                        enhancement_strength = gr.Slider(
                            minimum=1.0,
                            maximum=3.0,
                            value=self.config.get('manga_settings', {}).get('preprocessing', {}).get('enhancement_strength', 1.5),
                            step=0.1,
                            label="Enhancement Strength",
                            interactive=True
                        )
                        
                        denoise_strength = gr.Slider(
                            minimum=0,
                            maximum=50,
                            value=self.config.get('manga_settings', {}).get('preprocessing', {}).get('denoise_strength', 10),
                            step=1,
                            label="Denoise Strength",
                            interactive=True
                        )
                        
                        max_image_dimension = gr.Number(
                            label="Max Image Dimension (pixels)",
                            value=self.config.get('manga_settings', {}).get('preprocessing', {}).get('max_image_dimension', 2000),
                            minimum=500
                        )
                        
                        chunk_height = gr.Number(
                            label="Chunk Height for Large Images",
                            value=self.config.get('manga_settings', {}).get('preprocessing', {}).get('chunk_height', 1000),
                            minimum=500
                        )
                        
                        gr.Markdown("#### HD Strategy for Inpainting")
                        gr.Markdown("*Controls how large images are processed during inpainting*")
                        
                        hd_strategy = gr.Radio(
                            choices=["original", "resize", "crop"],
                            value=self.config.get('manga_settings', {}).get('advanced', {}).get('hd_strategy', 'resize'),
                            label="HD Strategy",
                            interactive=True,
                            info="original = legacy full-image; resize/crop = faster"
                        )
                        
                        hd_strategy_resize_limit = gr.Slider(
                            minimum=512,
                            maximum=4096,
                            value=self.config.get('manga_settings', {}).get('advanced', {}).get('hd_strategy_resize_limit', 1536),
                            step=64,
                            label="Resize Limit (long edge, px)",
                            info="For resize strategy",
                            interactive=True
                        )
                        
                        hd_strategy_crop_margin = gr.Slider(
                            minimum=0,
                            maximum=256,
                            value=self.config.get('manga_settings', {}).get('advanced', {}).get('hd_strategy_crop_margin', 16),
                            step=2,
                            label="Crop Margin (px)",
                            info="For crop strategy",
                            interactive=True
                        )
                        
                        hd_strategy_crop_trigger = gr.Slider(
                            minimum=256,
                            maximum=4096,
                            value=self.config.get('manga_settings', {}).get('advanced', {}).get('hd_strategy_crop_trigger_size', 1024),
                            step=64,
                            label="Crop Trigger Size (px)",
                            info="Apply crop only if long edge exceeds this",
                            interactive=True
                        )
                        
                        gr.Markdown("#### Image Tiling")
                        gr.Markdown("*Alternative tiling strategy (note: HD Strategy takes precedence)*")
                        
                        tiling_enabled = gr.Checkbox(
                            label="Enable Tiling",
                            value=self.config.get('manga_settings', {}).get('tiling', {}).get('enabled', False)
                        )
                        
                        tiling_tile_size = gr.Slider(
                            minimum=256,
                            maximum=1024,
                            value=self.config.get('manga_settings', {}).get('tiling', {}).get('tile_size', 480),
                            step=64,
                            label="Tile Size (px)",
                            interactive=True
                        )
                        
                        tiling_tile_overlap = gr.Slider(
                            minimum=0,
                            maximum=128,
                            value=self.config.get('manga_settings', {}).get('tiling', {}).get('tile_overlap', 64),
                            step=16,
                            label="Tile Overlap (px)",
                            interactive=True
                        )
                    
                    with gr.Accordion("🎨 Font & Text Rendering", open=False):
                        gr.Markdown("#### Font Sizing Algorithm")
                        
                        font_algorithm = gr.Radio(
                            choices=["smart", "simple"],
                            value=self.config.get('manga_settings', {}).get('font_sizing', {}).get('algorithm', 'smart'),
                            label="Font Sizing Algorithm",
                            interactive=True
                        )
                        
                        prefer_larger = gr.Checkbox(
                            label="Prefer Larger Fonts",
                            value=self.config.get('manga_settings', {}).get('font_sizing', {}).get('prefer_larger', True)
                        )
                        
                        max_lines = gr.Slider(
                            minimum=1,
                            maximum=20,
                            value=self.config.get('manga_settings', {}).get('font_sizing', {}).get('max_lines', 10),
                            step=1,
                            label="Maximum Lines Per Bubble",
                            interactive=True
                        )
                        
                        line_spacing = gr.Slider(
                            minimum=0.5,
                            maximum=3.0,
                            value=self.config.get('manga_settings', {}).get('font_sizing', {}).get('line_spacing', 1.3),
                            step=0.1,
                            label="Line Spacing Multiplier",
                            interactive=True
                        )
                        
                        bubble_size_factor = gr.Checkbox(
                            label="Use Bubble Size Factor",
                            value=self.config.get('manga_settings', {}).get('font_sizing', {}).get('bubble_size_factor', True)
                        )
                        
                        auto_fit_style = gr.Radio(
                            choices=["balanced", "aggressive", "conservative"],
                            value=self.config.get('manga_settings', {}).get('rendering', {}).get('auto_fit_style', 'balanced'),
                            label="Auto Fit Style",
                            interactive=True
                        )
                    
                    with gr.Accordion("⚙️ Advanced Options", open=False):
                        gr.Markdown("#### Format Detection")
                        
                        format_detection = gr.Checkbox(
                            label="Enable Format Detection (manga/webtoon)",
                            value=self.config.get('manga_settings', {}).get('advanced', {}).get('format_detection', True)
                        )
                        
                        webtoon_mode = gr.Radio(
                            choices=["auto", "force_manga", "force_webtoon"],
                            value=self.config.get('manga_settings', {}).get('advanced', {}).get('webtoon_mode', 'auto'),
                            label="Webtoon Mode",
                            interactive=True
                        )
                        
                        gr.Markdown("#### Performance")
                        
                        parallel_processing = gr.Checkbox(
                            label="Enable Parallel Processing",
                            value=self.config.get('manga_settings', {}).get('advanced', {}).get('parallel_processing', True)
                        )
                        
                        max_workers = gr.Slider(
                            minimum=1,
                            maximum=8,
                            value=self.config.get('manga_settings', {}).get('advanced', {}).get('max_workers', 2),
                            step=1,
                            label="Max Worker Threads",
                            interactive=True
                        )
                        
                        parallel_panel_translation = gr.Checkbox(
                            label="Parallel Panel Translation",
                            value=self.config.get('manga_settings', {}).get('advanced', {}).get('parallel_panel_translation', False)
                        )
                        
                        panel_max_workers = gr.Slider(
                            minimum=1,
                            maximum=20,
                            value=self.config.get('manga_settings', {}).get('advanced', {}).get('panel_max_workers', 10),
                            step=1,
                            label="Panel Max Workers",
                            interactive=True
                        )
                        
                        gr.Markdown("#### Model Optimization")
                        
                        torch_precision = gr.Radio(
                            choices=["fp32", "fp16"],
                            value=self.config.get('manga_settings', {}).get('advanced', {}).get('torch_precision', 'fp16'),
                            label="Torch Precision",
                            interactive=True
                        )
                        
                        auto_cleanup_models = gr.Checkbox(
                            label="Auto Cleanup Models from Memory",
                            value=self.config.get('manga_settings', {}).get('advanced', {}).get('auto_cleanup_models', False)
                        )
                        
                        gr.Markdown("#### Debug Options")
                        
                        debug_mode = gr.Checkbox(
                            label="Enable Debug Mode",
                            value=self.config.get('manga_settings', {}).get('advanced', {}).get('debug_mode', False)
                        )
                        
                        save_intermediate = gr.Checkbox(
                            label="Save Intermediate Files",
                            value=self.config.get('manga_settings', {}).get('advanced', {}).get('save_intermediate', False)
                        )
                        
                        concise_pipeline_logs = gr.Checkbox(
                            label="Concise Pipeline Logs",
                            value=self.config.get('concise_pipeline_logs', True)
                        )
                    
                    # Button handlers for model management
                    def download_models_handler(detector_type_val, inpaint_method_val):
                        """Download selected models"""
                        messages = []
                        
                        try:
                            # Download bubble detection model
                            if detector_type_val:
                                messages.append(f"📥 Downloading {detector_type_val} bubble detector...")
                                try:
                                    from bubble_detector import BubbleDetector
                                    bd = BubbleDetector()
                                    
                                    if detector_type_val == "rtdetr_onnx":
                                        if bd.load_rtdetr_onnx_model():
                                            messages.append("✅ RT-DETR ONNX model downloaded successfully")
                                        else:
                                            messages.append("❌ Failed to download RT-DETR ONNX model")
                                    elif detector_type_val == "rtdetr":
                                        if bd.load_rtdetr_model():
                                            messages.append("✅ RT-DETR model downloaded successfully")
                                        else:
                                            messages.append("❌ Failed to download RT-DETR model")
                                    elif detector_type_val == "yolo":
                                        messages.append("ℹ️ YOLO models are downloaded automatically on first use")
                                except Exception as e:
                                    messages.append(f"❌ Error downloading detector: {str(e)}")
                            
                            # Download inpainting model
                            if inpaint_method_val:
                                messages.append(f"\n📥 Downloading {inpaint_method_val} inpainting model...")
                                try:
                                    from local_inpainter import LocalInpainter, LAMA_JIT_MODELS
                                    
                                    inpainter = LocalInpainter({})
                                    
                                    # Map method names to download keys
                                    method_map = {
                                        'anime_onnx': 'anime_onnx',
                                        'anime': 'anime',
                                        'lama': 'lama',
                                        'lama_onnx': 'lama_onnx',
                                        'aot': 'aot',
                                        'aot_onnx': 'aot_onnx'
                                    }
                                    
                                    method_key = method_map.get(inpaint_method_val)
                                    if method_key and method_key in LAMA_JIT_MODELS:
                                        model_info = LAMA_JIT_MODELS[method_key]
                                        messages.append(f"Downloading {model_info['name']}...")
                                        
                                        model_path = inpainter.download_jit_model(method_key)
                                        if model_path:
                                            messages.append(f"✅ {model_info['name']} downloaded to: {model_path}")
                                        else:
                                            messages.append(f"❌ Failed to download {model_info['name']}")
                                    else:
                                        messages.append(f"ℹ️ {inpaint_method_val} is downloaded automatically on first use")
                                        
                                except Exception as e:
                                    messages.append(f"❌ Error downloading inpainting model: {str(e)}")
                            
                            if not messages:
                                messages.append("ℹ️ No models selected for download")
                                
                        except Exception as e:
                            messages.append(f"❌ Error during download: {str(e)}")
                        
                        return gr.Info("\n".join(messages))
                    
                    def load_models_handler(detector_type_val, inpaint_method_val):
                        """Load selected models into memory"""
                        messages = []
                        
                        try:
                            # Load bubble detection model
                            if detector_type_val:
                                messages.append(f"📦 Loading {detector_type_val} bubble detector...")
                                try:
                                    from bubble_detector import BubbleDetector
                                    bd = BubbleDetector()
                                    
                                    if detector_type_val == "rtdetr_onnx":
                                        if bd.load_rtdetr_onnx_model():
                                            messages.append("✅ RT-DETR ONNX model loaded successfully")
                                        else:
                                            messages.append("❌ Failed to load RT-DETR ONNX model")
                                    elif detector_type_val == "rtdetr":
                                        if bd.load_rtdetr_model():
                                            messages.append("✅ RT-DETR model loaded successfully")
                                        else:
                                            messages.append("❌ Failed to load RT-DETR model")
                                    elif detector_type_val == "yolo":
                                        messages.append("ℹ️ YOLO models are loaded automatically when needed")
                                except Exception as e:
                                    messages.append(f"❌ Error loading detector: {str(e)}")
                            
                            # Load inpainting model
                            if inpaint_method_val:
                                messages.append(f"\n📦 Loading {inpaint_method_val} inpainting model...")
                                try:
                                    from local_inpainter import LocalInpainter, LAMA_JIT_MODELS
                                    import os
                                    
                                    inpainter = LocalInpainter({})
                                    
                                    # Map method names to model keys
                                    method_map = {
                                        'anime_onnx': 'anime_onnx',
                                        'anime': 'anime',
                                        'lama': 'lama',
                                        'lama_onnx': 'lama_onnx',
                                        'aot': 'aot',
                                        'aot_onnx': 'aot_onnx'
                                    }
                                    
                                    method_key = method_map.get(inpaint_method_val)
                                    if method_key:
                                        # First check if model exists, download if not
                                        if method_key in LAMA_JIT_MODELS:
                                            model_info = LAMA_JIT_MODELS[method_key]
                                            cache_dir = os.path.expanduser('~/.cache/inpainting')
                                            model_filename = os.path.basename(model_info['url'])
                                            model_path = os.path.join(cache_dir, model_filename)
                                            
                                            if not os.path.exists(model_path):
                                                messages.append(f"Model not found, downloading first...")
                                                model_path = inpainter.download_jit_model(method_key)
                                                if not model_path:
                                                    messages.append(f"❌ Failed to download model")
                                                    return gr.Info("\n".join(messages))
                                            
                                            # Now load the model
                                            if inpainter.load_model(method_key, model_path):
                                                messages.append(f"✅ {model_info['name']} loaded successfully")
                                            else:
                                                messages.append(f"❌ Failed to load {model_info['name']}")
                                        else:
                                            messages.append(f"ℹ️ {inpaint_method_val} will be loaded automatically when needed")
                                    else:
                                        messages.append(f"ℹ️ Unknown method: {inpaint_method_val}")
                                        
                                except Exception as e:
                                    messages.append(f"❌ Error loading inpainting model: {str(e)}")
                            
                            if not messages:
                                messages.append("ℹ️ No models selected for loading")
                                
                        except Exception as e:
                            messages.append(f"❌ Error during loading: {str(e)}")
                        
                        return gr.Info("\n".join(messages))
                    
                    download_models_btn.click(
                        fn=download_models_handler,
                        inputs=[detector_type, local_inpaint_method],
                        outputs=None
                    )
                    
                    load_models_btn.click(
                        fn=load_models_handler,
                        inputs=[detector_type, local_inpaint_method],
                        outputs=None
                    )
                    
                    gr.Markdown("\n---\n**Note:** These settings will be saved to your config and applied to all manga translations.")
                
                # Glossary Extraction Tab - TEMPORARILY HIDDEN
                with gr.Tab("📝 Glossary Extraction", visible=False):
                    with gr.Row():
                        with gr.Column():
                            glossary_epub = gr.File(
                                label="📖 Upload EPUB File",
                                file_types=[".epub"]
                            )
                            
                            glossary_model = gr.Dropdown(
                                choices=self.models,
                                value="gpt-4-turbo",
                                label="🤖 AI Model"
                            )
                            
                            glossary_api_key = gr.Textbox(
                                label="🔑 API Key",
                                type="password",
                                placeholder="Enter API key"
                            )
                            
                            min_freq = gr.Slider(
                                minimum=1,
                                maximum=10,
                                value=2,
                                step=1,
                                label="Minimum Frequency"
                            )
                            
                            max_names_slider = gr.Slider(
                                minimum=10,
                                maximum=200,
                                value=50,
                                step=10,
                                label="Max Character Names"
                            )
                            
                            extract_btn = gr.Button(
                                "🔍 Extract Glossary",
                                variant="primary"
                            )
                        
                        with gr.Column():
                            glossary_output = gr.File(label="📥 Download Glossary CSV")
                            glossary_status = gr.Textbox(
                                label="Status",
                                lines=10
                            )
                    
                    extract_btn.click(
                        fn=self.extract_glossary,
                        inputs=[
                            glossary_epub,
                            glossary_model,
                            glossary_api_key,
                            min_freq,
                            max_names_slider
                        ],
                        outputs=[glossary_output, glossary_status]
                    )
                
                # Settings Tab
                with gr.Tab("⚙️ Settings"):
                    gr.Markdown("### Configuration")
                    
                    gr.Markdown("#### Translation Profiles")
                    gr.Markdown("Profiles are loaded from your `config.json` file. You can manage profiles in the desktop application.")
                    
                    with gr.Accordion("View All Profiles", open=False):
                        profiles_text = "\n\n".join(
                            [f"**{name}**:\n```\n{prompt[:200]}...\n```" 
                             for name, prompt in self.profiles.items()]
                        )
                        gr.Markdown(profiles_text if profiles_text else "No profiles found")
                    
                    gr.Markdown("---")
                    gr.Markdown("#### Advanced Translation Settings")
                    
                    with gr.Row():
                        with gr.Column():
                            thread_delay = gr.Slider(
                                minimum=0,
                                maximum=5,
                                value=self.config.get('thread_submission_delay', 0.5),
                                step=0.1,
                                label="Threading delay (s)"
                            )
                            
                            api_delay = gr.Slider(
                                minimum=0,
                                maximum=10,
                                value=self.config.get('delay', 2),
                                step=0.5,
                                label="API call delay (s)"
                            )
                            
                            chapter_range = gr.Textbox(
                                label="Chapter range (e.g., 5-10)",
                                value=self.config.get('chapter_range', ''),
                                placeholder="Leave empty for all chapters"
                            )
                            
                            token_limit = gr.Number(
                                label="Input Token limit",
                                value=self.config.get('token_limit', 200000),
                                minimum=0
                            )
                            
                            disable_token_limit = gr.Checkbox(
                                label="Disable Input Token Limit",
                                value=self.config.get('token_limit_disabled', False)
                            )
                            
                            output_token_limit = gr.Number(
                                label="Output Token limit",
                                value=self.config.get('max_output_tokens', 16000),
                                minimum=0
                            )
                        
                        with gr.Column():
                            contextual = gr.Checkbox(
                                label="Contextual Translation",
                                value=self.config.get('contextual', False)
                            )
                            
                            history_limit = gr.Number(
                                label="Translation History Limit",
                                value=self.config.get('translation_history_limit', 2),
                                minimum=0
                            )
                            
                            rolling_history = gr.Checkbox(
                                label="Rolling History Window",
                                value=self.config.get('translation_history_rolling', False)
                            )
                            
                            batch_translation = gr.Checkbox(
                                label="Batch Translation",
                                value=self.config.get('batch_translation', False)
                            )
                            
                            batch_size = gr.Number(
                                label="Batch Size",
                                value=self.config.get('batch_size', 3),
                                minimum=1
                            )
                    
                    gr.Markdown("---")
                    
                    save_api_key = gr.Checkbox(
                        label="Save API Key (⚠️ Warning: Stores key in plain text)",
                        value=False
                    )
                    
                    save_status = gr.Textbox(label="Settings Status", value="Settings auto-save on change", interactive=False)
                    
                    def save_settings(save_key, t_delay, a_delay, ch_range, tok_limit, disable_tok_limit, out_tok_limit, ctx, hist_lim, roll_hist, batch, b_size):
                        """Auto-save settings when changed"""
                        try:
                            # Reload latest config first to avoid overwriting other changes
                            current_config = self.load_config()
                            
                            # Update only the fields we're managing
                            current_config.update({
                                'save_api_key': save_key,
                                'thread_submission_delay': float(t_delay),
                                'delay': float(a_delay),
                                'chapter_range': str(ch_range),
                                'token_limit': int(tok_limit) if tok_limit else 200000,
                                'token_limit_disabled': bool(disable_tok_limit),
                                'max_output_tokens': int(out_tok_limit) if out_tok_limit else 16000,
                                'contextual': bool(ctx),
                                'translation_history_limit': int(hist_lim) if hist_lim else 2,
                                'translation_history_rolling': bool(roll_hist),
                                'batch_translation': bool(batch),
                                'batch_size': int(b_size) if b_size else 3
                            })
                            
                            # Save with the merged config
                            result = self.save_config(current_config)
                            return f"✅ {result}"
                        except Exception as e:
                            import traceback
                            error_trace = traceback.format_exc()
                            print(f"Settings save error:\n{error_trace}")
                            return f"❌ Save failed: {str(e)}"
                    
                    # Auto-save on any change
                    for component in [save_api_key, thread_delay, api_delay, chapter_range, token_limit, disable_token_limit, 
                                     output_token_limit, contextual, history_limit, rolling_history, batch_translation, batch_size]:
                        component.change(
                            fn=save_settings,
                            inputs=[
                                save_api_key,
                                thread_delay,
                                api_delay,
                                chapter_range,
                                token_limit,
                                disable_token_limit,
                                output_token_limit,
                                contextual,
                                history_limit,
                                rolling_history,
                                batch_translation,
                                batch_size
                            ],
                            outputs=[save_status]
                        )
                
                # Help Tab
                with gr.Tab("❓ Help"):
                    gr.Markdown("""
                    ## How to Use Glossarion
                    
                    ### Translation
                    1. Upload an EPUB file
                    2. Select AI model (GPT-4, Claude, etc.)
                    3. Enter your API key
                    4. Click "Translate"
                    5. Download the translated EPUB
                    
                    ### Manga Translation
                    1. Upload manga image(s) (PNG, JPG, etc.)
                    2. Select AI model and enter API key
                    3. Choose translation profile (e.g., Manga_JP, Manga_KR)
                    4. Configure OCR settings (Google Cloud Vision recommended)
                    5. Enable bubble detection and inpainting for best results
                    6. Click "Translate Manga"
                    
                    ### Glossary Extraction
                    1. Upload an EPUB file
                    2. Configure extraction settings
                    3. Click "Extract Glossary"
                    4. Use the CSV in future translations
                    
                    ### API Keys
                    - **OpenAI**: Get from https://platform.openai.com/api-keys
                    - **Anthropic**: Get from https://console.anthropic.com/
                    
                    ### Translation Profiles
                    Profiles contain detailed translation instructions and rules.
                    Select a profile that matches your source language and style preferences.
                    
                    You can create and edit profiles in the desktop application.
                    
                    ### Tips
                    - Use glossaries for consistent character name translation
                    - Lower temperature (0.1-0.3) for more literal translations
                    - Higher temperature (0.5-0.7) for more creative translations
                    """)
        
        return app


def main():
    """Launch Gradio web app"""
    print("🚀 Starting Glossarion Web Interface...")
    
    web_app = GlossarionWeb()
    app = web_app.create_interface()
    
    # Set favicon with absolute path if available
    favicon_path = None
    if os.path.exists("Halgakos.ico"):
        favicon_path = os.path.abspath("Halgakos.ico")
        print(f"✅ Using favicon: {favicon_path}")
    else:
        print("⚠️ Halgakos.ico not found")
    
    # Launch with options
    app.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,
        share=False,  # Set to True to create public link
        show_error=True,
        favicon_path=favicon_path
    )


if __name__ == "__main__":
    main()