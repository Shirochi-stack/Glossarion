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
    from api_key_encryption import APIKeyEncryption
    API_KEY_ENCRYPTION_AVAILABLE = True
    # Create web-specific encryption handler with its own key file
    _web_encryption_handler = None
    def get_web_encryption_handler():
        global _web_encryption_handler
        if _web_encryption_handler is None:
            _web_encryption_handler = APIKeyEncryption()
            # Use web-specific key file
            from pathlib import Path
            _web_encryption_handler.key_file = Path('.glossarion_web_key')
            _web_encryption_handler.cipher = _web_encryption_handler._get_or_create_cipher()
            # Add web-specific fields to encrypt
            _web_encryption_handler.api_key_fields.extend([
                'azure_vision_key',
                'google_vision_credentials'
            ])
        return _web_encryption_handler
    
    def decrypt_config(config):
        return get_web_encryption_handler().decrypt_config(config)
    
    def encrypt_config(config):
        return get_web_encryption_handler().encrypt_config(config)
except ImportError:
    API_KEY_ENCRYPTION_AVAILABLE = False
    def decrypt_config(config):
        return config  # Fallback: return config as-is
    def encrypt_config(config):
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
    print("✅ Manga translation modules loaded successfully")
except ImportError as e:
    MANGA_TRANSLATION_AVAILABLE = False
    print(f"⚠️ Manga translation modules not found: {e}")
    print(f"⚠️ Current working directory: {os.getcwd()}")
    print(f"⚠️ Python path: {sys.path[:3]}...")
    
    # Check if files exist
    files_to_check = ['manga_translator.py', 'unified_api_client.py', 'bubble_detector.py', 'local_inpainter.py']
    for file in files_to_check:
        if os.path.exists(file):
            print(f"✅ Found: {file}")
        else:
            print(f"❌ Missing: {file}")


class GlossarionWeb:
    """Web interface for Glossarion translator"""
    
    def __init__(self):
        self.config_file = "config_web.json"
        # Load raw config first
        self.config = self.load_config()
        
        # Create a decrypted version for display/use in the UI
        # but keep the original for saving
        self.decrypted_config = self.config.copy()
        if API_KEY_ENCRYPTION_AVAILABLE:
            self.decrypted_config = decrypt_config(self.decrypted_config)
        
        self.models = get_model_options() if TRANSLATION_AVAILABLE else ["gpt-4", "claude-3-5-sonnet"]
        print(f"🤖 Loaded {len(self.models)} models: {self.models[:5]}{'...' if len(self.models) > 5 else ''}")
        
        # Translation state management
        import threading
        self.is_translating = False
        self.stop_flag = threading.Event()
        self.translation_thread = None
        self.current_unified_client = None  # Track active client to allow cancellation
        self.current_translator = None     # Track active translator to allow shutdown
        
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
        
        # Load profiles from config and merge with defaults
        # Always include default prompts, then overlay any custom ones from config
        self.profiles = self.default_prompts.copy()
        config_profiles = self.config.get('prompt_profiles', {})
        if config_profiles:
            self.profiles.update(config_profiles)
    
    def get_config_value(self, key, default=None):
        """Get value from decrypted config with fallback"""
        return self.decrypted_config.get(key, default)
    
    def get_current_config_for_update(self):
        """Get the current config for updating (uses in-memory version)"""
        # Return a copy of the in-memory config, not loaded from file
        return self.config.copy()
    
    def get_default_config(self):
        """Get default configuration for Hugging Face Spaces"""
        return {
            'model': 'gpt-4-turbo',
            'api_key': '',
            'ocr_provider': 'custom-api',
            'bubble_detection_enabled': True,
            'inpainting_enabled': True,
            'manga_font_size_mode': 'auto',
            'manga_font_size': 24,
            'manga_font_size_multiplier': 1.0,
            'manga_max_font_size': 48,
            'manga_text_color': [255, 255, 255],
            'manga_shadow_enabled': True,
            'manga_shadow_color': [0, 0, 0],
            'manga_shadow_offset_x': 1,
            'manga_shadow_offset_y': 1,
            'manga_shadow_blur': 2,
            'manga_bg_opacity': 180,
            'manga_bg_style': 'circle',
            'manga_settings': {
                'ocr': {
                    'detector_type': 'rtdetr_onnx',
                    'rtdetr_confidence': 0.3,
                    'bubble_confidence': 0.3,
                    'detect_text_bubbles': True,
                    'detect_empty_bubbles': True,
                    'detect_free_text': True,
                    'bubble_max_detections_yolo': 100
                },
                'inpainting': {
                    'local_method': 'anime',
                    'method': 'local',
                    'batch_size': 10,
                    'enable_cache': True
                },
                'advanced': {
                    'parallel_processing': True,
                    'max_workers': 2,
                    'parallel_panel_translation': False,
                    'panel_max_workers': 7,
                    'format_detection': True,
                    'webtoon_mode': 'auto',
                    'torch_precision': 'fp16',
                    'auto_cleanup_models': False,
                    'debug_mode': False,
                    'save_intermediate': False
                },
                'rendering': {
                    'auto_min_size': 12,
                    'auto_max_size': 48,
                    'auto_fit_style': 'balanced'
                },
                'font_sizing': {
                    'algorithm': 'smart',
                    'prefer_larger': True,
                    'max_lines': 10,
                    'line_spacing': 1.3,
                    'bubble_size_factor': True,
                    'min_size': 12,
                    'max_size': 48
                },
                'tiling': {
                    'enabled': False,
                    'tile_size': 480,
                    'tile_overlap': 64
                }
            }
        }
    
    def load_config(self):
        """Load configuration - from file locally, from localStorage on HF"""
        is_hf_spaces = os.getenv('SPACE_ID') is not None or os.getenv('HF_SPACES') == 'true'
        
        if not is_hf_spaces:
            # Running locally - use config file
            try:
                if os.path.exists(self.config_file):
                    with open(self.config_file, 'r', encoding='utf-8') as f:
                        loaded_config = json.load(f)
                        # Start with defaults
                        default_config = self.get_default_config()
                        # Deep merge - preserve nested structures from loaded config
                        self._deep_merge_config(default_config, loaded_config)
                        return default_config
            except Exception as e:
                print(f"Could not load config: {e}")
        
        # HF Spaces or if loading fails - return defaults
        return self.get_default_config()
    
    def _deep_merge_config(self, base, override):
        """Deep merge override config into base config"""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                # Recursively merge nested dicts
                self._deep_merge_config(base[key], value)
            else:
                # Override the value
                base[key] = value
    
    def save_config(self, config):
        """Save configuration - to file locally, to localStorage on HF"""
        is_hf_spaces = os.getenv('SPACE_ID') is not None or os.getenv('HF_SPACES') == 'true'
        
        if not is_hf_spaces:
            # Running locally - save to file
            try:
                config_to_save = config.copy()
                
                # Only encrypt if we have the encryption module AND keys aren't already encrypted
                if API_KEY_ENCRYPTION_AVAILABLE:
                    # Check if keys need encryption (not already encrypted)
                    needs_encryption = False
                    for key in ['api_key', 'azure_vision_key', 'google_vision_credentials']:
                        if key in config_to_save:
                            value = config_to_save[key]
                            # If it's a non-empty string that doesn't start with 'ENC:', it needs encryption
                            if value and isinstance(value, str) and not value.startswith('ENC:'):
                                needs_encryption = True
                                break
                    
                    if needs_encryption:
                        config_to_save = encrypt_config(config_to_save)
                
                print(f"DEBUG save_config called with model={config.get('model')}, batch_size={config.get('batch_size')}")
                print(f"DEBUG self.config before={self.config.get('model') if hasattr(self, 'config') else 'N/A'}")
                print(f"DEBUG self.decrypted_config before={self.decrypted_config.get('model') if hasattr(self, 'decrypted_config') else 'N/A'}")
                with open(self.config_file, 'w', encoding='utf-8') as f:
                    json.dump(config_to_save, f, ensure_ascii=False, indent=2)
                
                # IMPORTANT: Update the in-memory configs so the UI reflects the changes immediately
                self.config = config_to_save
                # Update decrypted config too
                self.decrypted_config = config.copy()  # Use the original (unencrypted) version
                if API_KEY_ENCRYPTION_AVAILABLE:
                    # Make sure decrypted_config has decrypted values
                    self.decrypted_config = decrypt_config(self.decrypted_config)
                print(f"DEBUG self.config after={self.config.get('model')}")
                print(f"DEBUG self.decrypted_config after={self.decrypted_config.get('model')}")
                
                print(f"✅ Saved to {self.config_file}")
                return "✅ Settings saved successfully!"
            except Exception as e:
                print(f"❌ Save error: {e}")
                return f"❌ Failed to save: {str(e)}"
        
        # HF Spaces - indicate localStorage usage
        return "Settings saved to browser localStorage"
    
    def translate_epub(
        self,
        epub_file,
        model,
        api_key,
        profile_name,
        system_prompt,
        temperature,
        glossary_file=None
    ):
        """Translate EPUB file - yields progress updates"""
        
        if not TRANSLATION_AVAILABLE:
            yield None, None, None, "❌ Translation modules not loaded", None, "Error", 0
            return
        
        if not epub_file:
            yield None, None, None, "❌ Please upload an EPUB file", None, "Error", 0
            return
        
        if not api_key:
            yield None, None, None, "❌ Please provide an API key", None, "Error", 0
            return
        
        if not profile_name:
            yield None, None, None, "❌ Please select a translation profile", None, "Error", 0
            return
        
        # Initialize logs list
        translation_logs = []
        
        try:
            # Initial status
            translation_logs.append("📚 Starting EPUB translation...")
            yield None, None, gr.update(visible=True), "\n".join(translation_logs), gr.update(visible=True), "Starting...", 0
            
            # Save uploaded file to temp location if needed
            input_path = epub_file.name if hasattr(epub_file, 'name') else epub_file
            epub_base = os.path.splitext(os.path.basename(input_path))[0]
            
            translation_logs.append(f"📖 Input: {os.path.basename(input_path)}")
            translation_logs.append(f"🤖 Model: {model}")
            translation_logs.append(f"📝 Profile: {profile_name}")
            yield None, None, gr.update(visible=True), "\n".join(translation_logs), gr.update(visible=True), "Initializing...", 5
            
            # Use the provided system prompt (user may have edited it)
            translation_prompt = system_prompt if system_prompt else self.profiles.get(profile_name, "")
            
            # Set the input path as a command line argument simulation
            import sys
            original_argv = sys.argv.copy()
            sys.argv = ['glossarion_web.py', input_path]
            
            # Set environment variables for TransateKRtoEN.main()
            os.environ['INPUT_PATH'] = input_path
            os.environ['MODEL'] = model
            os.environ['TRANSLATION_TEMPERATURE'] = str(temperature)
            os.environ['MAX_OUTPUT_TOKENS'] = str(self.get_config_value('max_output_tokens', 16000))
            
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
            
            translation_logs.append("⚙️ Configuration set")
            yield None, None, gr.update(visible=True), "\n".join(translation_logs), gr.update(visible=True), "Starting translation...", 10
            
            # Create a thread-safe queue for capturing logs
            import queue
            import threading
            import time
            log_queue = queue.Queue()
            translation_complete = threading.Event()
            translation_error = [None]
            
            def log_callback(msg):
                """Capture log messages"""
                if msg and msg.strip():
                    log_queue.put(msg.strip())
            
            # Run translation in a separate thread
            def run_translation():
                try:
                    result = TransateKRtoEN.main(
                        log_callback=log_callback,
                        stop_callback=None
                    )
                    translation_error[0] = None
                except Exception as e:
                    translation_error[0] = e
                finally:
                    translation_complete.set()
            
            translation_thread = threading.Thread(target=run_translation, daemon=True)
            translation_thread.start()
            
            # Monitor progress
            last_yield_time = time.time()
            progress_percent = 10
            
            while not translation_complete.is_set() or not log_queue.empty():
                # Collect logs
                new_logs = []
                while not log_queue.empty():
                    try:
                        msg = log_queue.get_nowait()
                        new_logs.append(msg)
                    except queue.Empty:
                        break
                
                # Add new logs
                if new_logs:
                    translation_logs.extend(new_logs)
                    
                    # Update progress based on log content
                    for log in new_logs:
                        if 'Chapter' in log or 'chapter' in log:
                            progress_percent = min(progress_percent + 5, 90)
                        elif '✅' in log or 'Complete' in log:
                            progress_percent = min(progress_percent + 10, 95)
                        elif 'Translating' in log:
                            progress_percent = min(progress_percent + 2, 85)
                
                # Yield updates periodically
                current_time = time.time()
                if new_logs or (current_time - last_yield_time) > 1.0:
                    status_text = new_logs[-1] if new_logs else "Processing..."
                    # Keep only last 100 logs to avoid UI overflow
                    display_logs = translation_logs[-100:] if len(translation_logs) > 100 else translation_logs
                    yield None, None, gr.update(visible=True), "\n".join(display_logs), gr.update(visible=True), status_text, progress_percent
                    last_yield_time = current_time
                
                # Small delay to avoid CPU spinning
                time.sleep(0.1)
            
            # Wait for thread to complete
            translation_thread.join(timeout=5)
            
            # Restore original sys.argv
            sys.argv = original_argv
            
            # Check for errors
            if translation_error[0]:
                error_msg = f"❌ Translation error: {str(translation_error[0])}"
                translation_logs.append(error_msg)
                yield None, None, gr.update(visible=False), "\n".join(translation_logs), gr.update(visible=True), error_msg, 0
                return
            
            # Check for output EPUB
            output_dir = epub_base
            compiled_epub = None
            
            if os.path.exists(output_dir):
                # Look for compiled EPUB
                potential_epub = os.path.join(output_dir, f"{epub_base}_translated.epub")
                if os.path.exists(potential_epub):
                    compiled_epub = potential_epub
                    translation_logs.append(f"✅ Translation complete: {os.path.basename(compiled_epub)}")
                    yield compiled_epub, gr.update(visible=True), gr.update(visible=False), "\n".join(translation_logs), gr.update(visible=True), "Translation complete!", 100
                    return
            
            # Translation failed
            translation_logs.append("❌ Translation failed - output file not created")
            yield None, None, gr.update(visible=False), "\n".join(translation_logs), gr.update(visible=True), "Translation failed", 0
                
        except Exception as e:
            import traceback
            error_msg = f"❌ Error during translation:\n{str(e)}\n\n{traceback.format_exc()}"
            translation_logs.append(error_msg)
            yield None, None, gr.update(visible=False), "\n".join(translation_logs), gr.update(visible=True), "Error occurred", 0
    
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
    
    def stop_translation(self):
        """Stop the ongoing translation process"""
        print(f"DEBUG: stop_translation called, was_translating={self.is_translating}")
        if self.is_translating:
            print("DEBUG: Setting stop flag and cancellation")
            self.stop_flag.set()
            self.is_translating = False
            
            # Best-effort: cancel any in-flight API operation on the active client
            try:
                if getattr(self, 'current_unified_client', None):
                    self.current_unified_client.cancel_current_operation()
                    print("DEBUG: Requested UnifiedClient cancellation")
            except Exception as e:
                print(f"DEBUG: UnifiedClient cancel failed: {e}")
            
            # Also propagate to MangaTranslator class if available
            try:
                if MANGA_TRANSLATION_AVAILABLE:
                    from manga_translator import MangaTranslator
                    MangaTranslator.set_global_cancellation(True)
                    print("DEBUG: Set MangaTranslator global cancellation")
            except ImportError:
                pass
            
            # Also propagate to UnifiedClient if available
            try:
                if MANGA_TRANSLATION_AVAILABLE:
                    from unified_api_client import UnifiedClient
                    UnifiedClient.set_global_cancellation(True)
                    print("DEBUG: Set UnifiedClient global cancellation")
            except ImportError:
                pass
            
            # Kick off translator shutdown to free resources quickly
            try:
                tr = getattr(self, 'current_translator', None)
                if tr and hasattr(tr, 'shutdown'):
                    import threading as _th
                    _th.Thread(target=tr.shutdown, name="WebMangaTranslatorShutdown", daemon=True).start()
                    print("DEBUG: Initiated translator shutdown thread")
                    # Clear reference so a new start creates a fresh instance
                    self.current_translator = None
            except Exception as e:
                print(f"DEBUG: Failed to start translator shutdown: {e}")
        else:
            print("DEBUG: stop_translation called but not translating")
    
    def _reset_translation_flags(self):
        """Reset all translation flags for new translation"""
        self.is_translating = False
        self.stop_flag.clear()
        
        # Reset global cancellation flags
        try:
            if MANGA_TRANSLATION_AVAILABLE:
                from manga_translator import MangaTranslator
                MangaTranslator.set_global_cancellation(False)
        except ImportError:
            pass
            
        try:
            if MANGA_TRANSLATION_AVAILABLE:
                from unified_api_client import UnifiedClient
                UnifiedClient.set_global_cancellation(False)
        except ImportError:
            pass
    
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
        bg_style,
        parallel_panel_translation=False,
        panel_max_workers=10
    ):
        """Translate manga images - GENERATOR that yields (logs, image, cbz_file, status, progress_group, progress_text, progress_bar) updates"""
        
        # Reset translation flags and set running state
        self._reset_translation_flags()
        self.is_translating = True
        
        if not MANGA_TRANSLATION_AVAILABLE:
            self.is_translating = False
            yield "❌ Manga translation modules not loaded", None, None, gr.update(value="❌ Error", visible=True), gr.update(visible=False), gr.update(value="Error"), gr.update(value=0)
            return
        
        if not image_files:
            self.is_translating = False
            yield "❌ Please upload at least one image", gr.update(visible=False), gr.update(visible=False), gr.update(value="❌ Error", visible=True), gr.update(visible=False), gr.update(value="Error"), gr.update(value=0)
            return
        
        if not api_key:
            self.is_translating = False
            yield "❌ Please provide an API key", gr.update(visible=False), gr.update(visible=False), gr.update(value="❌ Error", visible=True), gr.update(visible=False), gr.update(value="Error"), gr.update(value=0)
            return
        
        # Check for stop request
        if self.stop_flag.is_set():
            self.is_translating = False
            yield "⏹️ Translation stopped by user", gr.update(visible=False), gr.update(visible=False), gr.update(value="⏹️ Stopped", visible=True), gr.update(visible=False), gr.update(value="Stopped"), gr.update(value=0)
            return
        
        if ocr_provider == "google":
            # Check if credentials are provided or saved in config
            if not google_creds_path and not self.get_config_value('google_vision_credentials'):
                yield "❌ Please provide Google Cloud credentials JSON file", gr.update(visible=False), gr.update(visible=False), gr.update(value="❌ Error", visible=True), gr.update(visible=False), gr.update(value="Error"), gr.update(value=0)
                return
        
        if ocr_provider == "azure":
            # Ensure azure credentials are strings
            azure_key_str = str(azure_key) if azure_key else ''
            azure_endpoint_str = str(azure_endpoint) if azure_endpoint else ''
            if not azure_key_str.strip() or not azure_endpoint_str.strip():
                yield "❌ Please provide Azure API key and endpoint", gr.update(visible=False), gr.update(visible=False), gr.update(value="❌ Error", visible=True), gr.update(visible=False), gr.update(value="Error"), gr.update(value=0)
                return
        
        try:
            
            # Set API key environment variable
            if 'gpt' in model.lower() or 'openai' in model.lower():
                os.environ['OPENAI_API_KEY'] = api_key
            elif 'claude' in model.lower():
                os.environ['ANTHROPIC_API_KEY'] = api_key
            elif 'gemini' in model.lower():
                os.environ['GOOGLE_API_KEY'] = api_key
            
            # Set Google Cloud credentials if provided and save to config
            if ocr_provider == "google":
                if google_creds_path:
                    # New file provided - save it
                    creds_path = google_creds_path.name if hasattr(google_creds_path, 'name') else google_creds_path
                    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = creds_path
                    # Auto-save to config
                    self.config['google_vision_credentials'] = creds_path
                    self.save_config(self.config)
                elif self.get_config_value('google_vision_credentials'):
                    # Use saved credentials from config
                    creds_path = self.get_config_value('google_vision_credentials')
                    if os.path.exists(creds_path):
                        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = creds_path
                    else:
                        yield f"❌ Saved Google credentials not found: {creds_path}", gr.update(visible=False), gr.update(visible=False), gr.update(value="❌ Error", visible=True), gr.update(visible=False), gr.update(value="Error"), gr.update(value=0)
                        return
            
            # Set Azure credentials if provided and save to config
            if ocr_provider == "azure":
                # Convert to strings and strip whitespace
                azure_key_str = str(azure_key).strip() if azure_key else ''
                azure_endpoint_str = str(azure_endpoint).strip() if azure_endpoint else ''
                
                os.environ['AZURE_VISION_KEY'] = azure_key_str
                os.environ['AZURE_VISION_ENDPOINT'] = azure_endpoint_str
                # Auto-save to config
                self.config['azure_vision_key'] = azure_key_str
                self.config['azure_vision_endpoint'] = azure_endpoint_str
                self.save_config(self.config)
            
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
            
            # Initialize translation logs early (needed for CBZ processing)
            translation_logs = []
            
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
            if 'advanced' not in merged_config['manga_settings']:
                merged_config['manga_settings']['advanced'] = {}
            
            merged_config['manga_settings']['ocr']['provider'] = ocr_provider
            merged_config['manga_settings']['ocr']['bubble_detection_enabled'] = enable_bubble_detection
            merged_config['manga_settings']['inpainting']['method'] = 'local' if enable_inpainting else 'none'
            # Make sure local_method is set from config (defaults to anime)
            if 'local_method' not in merged_config['manga_settings']['inpainting']:
                merged_config['manga_settings']['inpainting']['local_method'] = self.get_config_value('manga_settings', {}).get('inpainting', {}).get('local_method', 'anime')
            
            # Set parallel panel translation settings from config (Manga Settings tab)
            # These are controlled in the Manga Settings tab, so reload config to get latest values
            current_config = self.load_config()
            if API_KEY_ENCRYPTION_AVAILABLE:
                current_config = decrypt_config(current_config)
            
            config_parallel = current_config.get('manga_settings', {}).get('advanced', {}).get('parallel_panel_translation', False)
            config_max_workers = current_config.get('manga_settings', {}).get('advanced', {}).get('panel_max_workers', 10)
            
            # Map web UI settings to MangaTranslator expected names
            merged_config['manga_settings']['advanced']['parallel_panel_translation'] = config_parallel
            merged_config['manga_settings']['advanced']['panel_max_workers'] = int(config_max_workers)
            # CRITICAL: Also set the setting names that MangaTranslator actually checks
            merged_config['manga_settings']['advanced']['parallel_processing'] = config_parallel
            merged_config['manga_settings']['advanced']['max_workers'] = int(config_max_workers)
            
            # Log the parallel settings being used
            print(f"🔧 Reloaded config - Using parallel panel translation: {config_parallel}")
            print(f"🔧 Reloaded config - Using panel max workers: {config_max_workers}")
            
            # CRITICAL: Set skip_inpainting flag to False when inpainting is enabled
            merged_config['manga_skip_inpainting'] = not enable_inpainting
            
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
                            self.profile = str(profile) if profile else ''
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
                            # Ensure val is properly typed
                            self.val = val
                        def get(self):
                            return self.val
                    self.delay_entry = MockVar(float(config.get('delay', 2.0)))
                    self.trans_temp = MockVar(float(config.get('translation_temperature', 0.3)))
                    self.contextual_var = MockVar(bool(config.get('contextual', False)))
                    self.trans_history = MockVar(int(config.get('translation_history_limit', 2)))
                    self.translation_history_rolling_var = MockVar(bool(config.get('translation_history_rolling', False)))
                    self.token_limit_disabled = bool(config.get('token_limit_disabled', False))
                    # IMPORTANT: token_limit_entry must return STRING because manga_translator calls .strip() on it
                    self.token_limit_entry = MockVar(str(config.get('token_limit', 200000)))
                    # Add API key and model for custom-api OCR provider - ensure strings
                    self.api_key_entry = MockVar(str(api_key) if api_key else '')
                    self.model_var = MockVar(str(model) if model else '')
            
            simple_config = SimpleConfig(merged_config)
            # Get max_output_tokens from config or use from web app config
            web_max_tokens = merged_config.get('max_output_tokens', 16000)
            mock_gui = MockGUI(simple_config.config, profile_name, system_prompt, web_max_tokens, api_key, model)
            
            # Ensure model path is in config for local inpainting
            if enable_inpainting:
                local_method = merged_config.get('manga_settings', {}).get('inpainting', {}).get('local_method', 'anime')
                # Set the model path key that MangaTranslator expects
                model_path_key = f'manga_{local_method}_model_path'
                if model_path_key not in merged_config:
                    # Use default model path or empty string
                    default_model_path = self.get_config_value(model_path_key, '')
                    merged_config[model_path_key] = default_model_path
                    print(f"Set {model_path_key} to: {default_model_path}")
            
            # Setup OCR configuration
            ocr_config = {
                'provider': ocr_provider
            }
            
            if ocr_provider == 'google':
                ocr_config['google_credentials_path'] = google_creds_path.name if google_creds_path else None
            elif ocr_provider == 'azure':
                # Use string versions
                azure_key_str = str(azure_key).strip() if azure_key else ''
                azure_endpoint_str = str(azure_endpoint).strip() if azure_endpoint else ''
                ocr_config['azure_key'] = azure_key_str
                ocr_config['azure_endpoint'] = azure_endpoint_str
            
            # Create UnifiedClient for translation API calls
            try:
                unified_client = UnifiedClient(
                    api_key=api_key,
                    model=model,
                    output_dir=output_dir
                )
                # Store reference for stop() cancellation support
                self.current_unified_client = unified_client
            except Exception as e:
                error_log = f"❌ Failed to initialize API client: {str(e)}"
                yield error_log, gr.update(visible=False), gr.update(visible=False), gr.update(value=error_log, visible=True), gr.update(visible=False), gr.update(value="Error"), gr.update(value=0)
                return
            
            # Log storage - will be yielded as live updates
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
                # Debug: Log inpainting config
                inpaint_cfg = merged_config.get('manga_settings', {}).get('inpainting', {})
                print(f"\n=== INPAINTING CONFIG DEBUG ===")
                print(f"Inpainting enabled checkbox: {enable_inpainting}")
                print(f"Inpainting method: {inpaint_cfg.get('method')}")
                print(f"Local method: {inpaint_cfg.get('local_method')}")
                print(f"Full inpainting config: {inpaint_cfg}")
                print("=== END DEBUG ===\n")
                
                translator = MangaTranslator(
                    ocr_config=ocr_config,
                    unified_client=unified_client,
                    main_gui=mock_gui,
                    log_callback=capture_log
                )
                
                # Keep a reference for stop/shutdown support
                self.current_translator = translator
                
                # Connect stop flag so translator can react immediately to stop requests
                if hasattr(translator, 'set_stop_flag'):
                    try:
                        translator.set_stop_flag(self.stop_flag)
                    except Exception:
                        pass
                
                # CRITICAL: Set skip_inpainting flag directly on translator instance
                translator.skip_inpainting = not enable_inpainting
                print(f"Set translator.skip_inpainting = {translator.skip_inpainting}")
                
                # Explicitly initialize local inpainting if enabled
                if enable_inpainting:
                    print(f"🎨 Initializing local inpainting...")
                    try:
                        # Force initialization of the inpainter
                        init_result = translator._initialize_local_inpainter()
                        if init_result:
                            print(f"✅ Local inpainter initialized successfully")
                        else:
                            print(f"⚠️ Local inpainter initialization returned False")
                    except Exception as init_error:
                        print(f"❌ Failed to initialize inpainter: {init_error}")
                        import traceback
                        traceback.print_exc()
                
            except Exception as e:
                import traceback
                full_error = traceback.format_exc()
                print(f"\n\n=== MANGA TRANSLATOR INIT ERROR ===")
                print(full_error)
                print(f"\nocr_config: {ocr_config}")
                print(f"\nmock_gui.model_var.get(): {mock_gui.model_var.get()}")
                print(f"\nmock_gui.api_key_entry.get(): {type(mock_gui.api_key_entry.get())}")
                print("=== END ERROR ===")
                error_log = f"❌ Failed to initialize manga translator: {str(e)}\n\nCheck console for full traceback"
                yield error_log, gr.update(visible=False), gr.update(visible=False), gr.update(value=error_log, visible=True), gr.update(visible=False), gr.update(value="Error"), gr.update(value=0)
                return
            
            # Process each image with real progress tracking
            for idx, img_file in enumerate(files_to_process, 1):
                try:
                    # Check for stop request before processing each image
                    if self.stop_flag.is_set():
                        translation_logs.append(f"\n⏹️ Translation stopped by user before image {idx}/{total_images}")
                        self.is_translating = False
                        yield "\n".join(translation_logs), gr.update(visible=False), gr.update(visible=False), gr.update(value="⏹️ Translation stopped", visible=True), gr.update(visible=True), gr.update(value="Stopped"), gr.update(value=0)
                        return
                    
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
                    
                    # Yield initial log update with progress
                    progress_percent = int(((idx - 1) / total_images) * 100)
                    status_text = f"Processing {idx}/{total_images}: {filename}"
                    last_yield_log_count[0] = len(translation_logs)
                    last_yield_time[0] = time.time()
                    yield "\n".join(translation_logs), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(value=status_text), gr.update(value=progress_percent)
                    
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
                        
                        # Check for stop request during processing
                        if self.stop_flag.is_set():
                            translation_logs.append(f"\n⏹️ Translation stopped by user while processing image {idx}/{total_images}")
                            self.is_translating = False
                            yield "\n".join(translation_logs), gr.update(visible=False), gr.update(visible=False), gr.update(value="⏹️ Translation stopped", visible=True), gr.update(visible=True), gr.update(value="Stopped"), gr.update(value=0)
                            return
                        
                        if should_yield_logs():
                            progress_percent = int(((idx - 0.5) / total_images) * 100)  # Mid-processing
                            status_text = f"Processing {idx}/{total_images}: {filename} (in progress...)"
                            last_yield_log_count[0] = len(translation_logs)
                            last_yield_time[0] = time.time()
                            yield "\n".join(translation_logs), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(value=status_text), gr.update(value=progress_percent)
                    
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
                            progress_percent = int((idx / total_images) * 100)
                            status_text = f"Completed {idx}/{total_images}: {filename}"
                            yield "\n".join(translation_logs), gr.update(value=final_output, visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(value=status_text), gr.update(value=progress_percent)
                        else:
                            translation_logs.append(f"⚠️ Image {idx}/{total_images}: Output file missing for {filename}")
                            translation_logs.append(f"⚠️ Warning: Output file not found for image {idx}")
                            translation_logs.append("")
                            # Yield progress update
                            progress_percent = int((idx / total_images) * 100)
                            status_text = f"Warning: {idx}/{total_images} - Output missing for {filename}"
                            yield "\n".join(translation_logs), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(value=status_text), gr.update(value=progress_percent)
                    else:
                        errors = result.get('errors', [])
                        error_msg = errors[0] if errors else 'Unknown error'
                        translation_logs.append(f"❌ Image {idx}/{total_images} FAILED: {error_msg[:50]}")
                        translation_logs.append(f"⚠️ Error on image {idx}: {error_msg}")
                        translation_logs.append("")
                        # Yield progress update
                        progress_percent = int((idx / total_images) * 100)
                        status_text = f"Failed: {idx}/{total_images} - {filename}"
                        yield "\n".join(translation_logs), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(value=status_text), gr.update(value=progress_percent)
                        
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
            
            # Check for stop request before final processing
            if self.stop_flag.is_set():
                translation_logs.append("\n⏹️ Translation stopped by user")
                self.is_translating = False
                yield "\n".join(translation_logs), gr.update(visible=False), gr.update(visible=False), gr.update(value="⏹️ Translation stopped", visible=True), gr.update(visible=True), gr.update(value="Stopped"), gr.update(value=0)
                return
                
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
            # Format: (logs_textbox, output_image, cbz_file, status_textbox, progress_group, progress_text, progress_bar)
            final_progress_text = f"Complete! Processed {len(translated_files)}/{total_images} images"
            if translated_files:
                # If CBZ mode, show CBZ file for download; otherwise show first image
                if cbz_mode and cbz_output_path and os.path.exists(cbz_output_path):
                    yield (
                        "\n".join(translation_logs), 
                        gr.update(value=translated_files[0], visible=True), 
                        gr.update(value=cbz_output_path, visible=True),  # CBZ file for download with visibility
                        gr.update(value=final_status_text, visible=True),
                        gr.update(visible=True),
                        gr.update(value=final_progress_text),
                        gr.update(value=100)
                    )
                else:
                    yield (
                        "\n".join(translation_logs), 
                        gr.update(value=translated_files[0], visible=True), 
                        gr.update(visible=False),  # Hide CBZ component
                        gr.update(value=final_status_text, visible=True),
                        gr.update(visible=True),
                        gr.update(value=final_progress_text),
                        gr.update(value=100)
                    )
            else:
                yield (
                    "\n".join(translation_logs), 
                    gr.update(visible=False), 
                    gr.update(visible=False),  # Hide CBZ component
                    gr.update(value=final_status_text, visible=True),
                    gr.update(visible=True),
                    gr.update(value=final_progress_text),
                    gr.update(value=0)  # 0% if nothing was processed
                )
                
        except Exception as e:
            import traceback
            error_msg = f"❌ Error during manga translation:\n{str(e)}\n\n{traceback.format_exc()}"
            self.is_translating = False
            yield error_msg, gr.update(visible=False), gr.update(visible=False), gr.update(value=error_msg, visible=True), gr.update(visible=False), gr.update(value="Error occurred"), gr.update(value=0)
        finally:
            # Always reset translation state when done
            self.is_translating = False
            # Clear active references on full completion
            try:
                self.current_translator = None
                self.current_unified_client = None
            except Exception:
                pass
    
    def stop_manga_translation(self):
        """Simple function to stop manga translation"""
        print("DEBUG: Stop button clicked")
        if self.is_translating:
            print("DEBUG: Stopping active translation")
            self.stop_translation()
            # Return UI updates for button visibility and status
            return (
                gr.update(visible=True),   # translate button - show
                gr.update(visible=False),  # stop button - hide  
                "⏹️ Translation stopped by user"
            )
        else:
            print("DEBUG: No active translation to stop")
            return (
                gr.update(visible=True),   # translate button - show
                gr.update(visible=False),  # stop button - hide
                "No active translation to stop"
            )
    
    def start_manga_translation(self, *args):
        """Simple function to start manga translation - GENERATOR FUNCTION"""
        print("DEBUG: Translate button clicked")
        
        # Reset flags for new translation and mark as translating BEFORE first yield
        self._reset_translation_flags()
        self.is_translating = True
        
        # Initial yield to update button visibility
        yield (
            "🚀 Starting translation...",
            gr.update(visible=False),  # manga_output_image
            gr.update(visible=False),  # manga_cbz_output  
            gr.update(value="Starting...", visible=True),  # manga_status
            gr.update(visible=False),  # manga_progress_group
            gr.update(value="Initializing..."),  # manga_progress_text
            gr.update(value=0),  # manga_progress_bar
            gr.update(visible=False),  # translate button - hide during translation
            gr.update(visible=True)   # stop button - show during translation
        )
        
        # Call the translate function and yield all its results
        last_result = None
        try:
            for result in self.translate_manga(*args):
                # Check if stop was requested during iteration
                if self.stop_flag.is_set():
                    print("DEBUG: Stop flag detected, breaking translation loop")
                    break
                    
                last_result = result
                # Pad result to include button states (translate_visible=False, stop_visible=True)
                if len(result) >= 7:
                    yield result + (gr.update(visible=False), gr.update(visible=True))
                else:
                    # Pad result to match expected length (7 values) then add button states
                    padded_result = list(result) + [gr.update(visible=False)] * (7 - len(result))
                    yield tuple(padded_result) + (gr.update(visible=False), gr.update(visible=True))
                    
        except GeneratorExit:
            print("DEBUG: Translation generator was closed")
            self.is_translating = False
            return
        except Exception as e:
            print(f"DEBUG: Exception during translation: {e}")
            self.is_translating = False
            # Show error and reset buttons
            error_msg = f"❌ Error during translation: {str(e)}"
            yield (
                error_msg,
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(value=error_msg, visible=True),
                gr.update(visible=False),
                gr.update(value="Error occurred"),
                gr.update(value=0),
                gr.update(visible=True),   # translate button - show after error
                gr.update(visible=False)   # stop button - hide after error
            )
            return
        finally:
            # Clear active references when the loop exits
            self.is_translating = False
            try:
                self.current_translator = None
                self.current_unified_client = None
            except Exception:
                pass
        
        # Check if we stopped early
        if self.stop_flag.is_set():
            yield (
                "⏹️ Translation stopped by user",
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(value="⏹️ Translation stopped", visible=True),
                gr.update(visible=False),
                gr.update(value="Stopped"),
                gr.update(value=0),
                gr.update(visible=True),   # translate button - show after stop
                gr.update(visible=False)   # stop button - hide after stop
            )
            return
        
        # Final yield to reset buttons after successful completion
        print("DEBUG: Translation completed normally, resetting buttons")
        if last_result is None:
            last_result = ("", gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(value="Complete"), gr.update(value=100))
        
        if len(last_result) >= 7:
            yield last_result[:7] + (gr.update(visible=True), gr.update(visible=False))
        else:
            # Pad result to match expected length then add button states
            padded_result = list(last_result) + [gr.update(visible=False)] * (7 - len(last_result))
            yield tuple(padded_result) + (gr.update(visible=True), gr.update(visible=False))
    
    def create_interface(self):
        """Create and return the Gradio interface"""
        # Reload config before creating interface to get latest values
        self.config = self.load_config()
        self.decrypted_config = decrypt_config(self.config.copy()) if API_KEY_ENCRYPTION_AVAILABLE else self.config.copy()
        
        # Load and encode icon as base64
        icon_base64 = ""
        icon_path = "Halgakos.ico" if os.path.exists("Halgakos.ico") else "Halgakos.ico"
        if os.path.exists(icon_path):
            with open(icon_path, "rb") as f:
                icon_base64 = base64.b64encode(f.read()).decode()
        
        # Custom CSS to hide Gradio footer and add favicon
        custom_css = """
        footer {display: none !important;}
        .gradio-container {min-height: 100vh;}
        
        /* Stop button styling */
        .gr-button[data-variant="stop"] {
            background-color: #dc3545 !important;
            border-color: #dc3545 !important;
            color: white !important;
        }
        .gr-button[data-variant="stop"]:hover {
            background-color: #c82333 !important;
            border-color: #bd2130 !important;
            color: white !important;
        }
        """
        
        # JavaScript for localStorage persistence - SIMPLE VERSION
        localStorage_js = """
        <script>
        console.log('Glossarion localStorage script loading...');
        
        // Simple localStorage functions
        function saveToLocalStorage(key, value) {
            try {
                localStorage.setItem('glossarion_' + key, JSON.stringify(value));
                console.log('Saved:', key, '=', value);
                return true;
            } catch (e) {
                console.error('Save failed:', e);
                return false;
            }
        }
        
        function loadFromLocalStorage(key, defaultValue) {
            try {
                const item = localStorage.getItem('glossarion_' + key);
                return item ? JSON.parse(item) : defaultValue;
            } catch (e) {
                console.error('Load failed:', e);
                return defaultValue;
            }
        }
        
        // Manual save current form values to localStorage
        function saveCurrentSettings() {
            const settings = {};
            
            // Find all input elements in Gradio
            document.querySelectorAll('input, select, textarea').forEach(el => {
                // Skip file inputs
                if (el.type === 'file') return;
                
                // Get a unique key based on element properties
                let key = el.id || el.name || el.placeholder || '';
                if (!key) {
                    // Try to get label text
                    const label = el.closest('div')?.querySelector('label');
                    if (label) key = label.textContent;
                }
                
                if (key) {
                    key = key.trim().replace(/[^a-zA-Z0-9]/g, '_');
                    if (el.type === 'checkbox') {
                        settings[key] = el.checked;
                    } else if (el.type === 'radio') {
                        if (el.checked) settings[key] = el.value;
                    } else if (el.value) {
                        settings[key] = el.value;
                    }
                }
            });
            
            // Save all settings
            Object.keys(settings).forEach(key => {
                saveToLocalStorage(key, settings[key]);
            });
            
            console.log('Saved', Object.keys(settings).length, 'settings');
            return settings;
        }
        
        // Export settings from localStorage
        function exportSettings() {
            console.log('Export started');
            
            // First save current form state
            saveCurrentSettings();
            
            // Then export from localStorage
            const settings = {};
            for (let i = 0; i < localStorage.length; i++) {
                const key = localStorage.key(i);
                if (key && key.startsWith('glossarion_')) {
                    try {
                        settings[key.replace('glossarion_', '')] = JSON.parse(localStorage.getItem(key));
                    } catch (e) {
                        // Store as-is if not JSON
                        settings[key.replace('glossarion_', '')] = localStorage.getItem(key);
                    }
                }
            }
            
            if (Object.keys(settings).length === 0) {
                alert('No settings to export. Try saving some settings first.');
                return;
            }
            
            // Download as JSON
            const blob = new Blob([JSON.stringify(settings, null, 2)], {type: 'application/json'});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'glossarion_settings_' + new Date().toISOString().slice(0,19).replace(/:/g, '-') + '.json';
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
            
            console.log('Exported', Object.keys(settings).length, 'settings');
        }
        
        function importSettings(fileContent) {
            try {
                const settings = JSON.parse(fileContent);
                Object.keys(settings).forEach(key => {
                    saveToLocalStorage(key, settings[key]);
                });
                location.reload(); // Reload to apply settings
            } catch (e) {
                alert('Invalid settings file format');
            }
        }
        
        // Expose to global scope
        window.exportSettings = exportSettings;
        window.importSettings = importSettings;
        window.saveCurrentSettings = saveCurrentSettings;
        window.saveToLocalStorage = saveToLocalStorage;
        window.loadFromLocalStorage = loadFromLocalStorage;
        
        // Load settings from localStorage on page load for HF Spaces
        function loadSettingsFromLocalStorage() {
            console.log('Attempting to load settings from localStorage...');
            try {
                // Get all localStorage items with glossarion_ prefix
                const settings = {};
                for (let i = 0; i < localStorage.length; i++) {
                    const key = localStorage.key(i);
                    if (key && key.startsWith('glossarion_')) {
                        const cleanKey = key.replace('glossarion_', '');
                        try {
                            settings[cleanKey] = JSON.parse(localStorage.getItem(key));
                        } catch (e) {
                            settings[cleanKey] = localStorage.getItem(key);
                        }
                    }
                }
                
                if (Object.keys(settings).length > 0) {
                    console.log('Found', Object.keys(settings).length, 'settings in localStorage');
                    
                    // Try to update Gradio components
                    // This is tricky because Gradio components are rendered dynamically
                    // We'll need to find them by their labels or other identifiers
                    
                    // For now, just log what we found
                    console.log('Settings:', settings);
                }
            } catch (e) {
                console.error('Error loading from localStorage:', e);
            }
        }
        
        // Try loading settings at various points
        window.addEventListener('load', function() {
            console.log('Page loaded');
            setTimeout(loadSettingsFromLocalStorage, 1000);
            setTimeout(loadSettingsFromLocalStorage, 3000);
        });
        
        document.addEventListener('DOMContentLoaded', function() {
            console.log('DOM ready');
            setTimeout(loadSettingsFromLocalStorage, 500);
        });
        </script>
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
            {localStorage_js}
            """)
            
            gr.Markdown("""
            Translate novels and books using advanced AI models (GPT-5, Claude, etc.)
            """)
            
            with gr.Tabs() as main_tabs:
                # EPUB Translation Tab
                with gr.Tab("📚 EPUB Translation"):
                    with gr.Row():
                        with gr.Column():
                            epub_file = gr.File(
                                label="📖 Upload EPUB File",
                                file_types=[".epub"]
                            )
                            
                            translate_btn = gr.Button(
                                "🚀 Translate EPUB",
                                variant="primary",
                                size="lg"
                            )
                            
                            epub_model = gr.Dropdown(
                                choices=self.models,
                                value=self.get_config_value('model', 'gpt-4-turbo'),
                                label="🤖 AI Model",
                                interactive=True,
                                allow_custom_value=True,
                                filterable=True
                            )
                            
                            epub_api_key = gr.Textbox(
                                label="🔑 API Key",
                                type="password",
                                placeholder="Enter your API key",
                                value=self.get_config_value('api_key', '')
                            )
                            
                            # Use all profiles without filtering
                            profile_choices = list(self.profiles.keys())
                            # Use saved active_profile instead of hardcoded default
                            default_profile = self.get_config_value('active_profile', profile_choices[0] if profile_choices else '')
                            
                            epub_profile = gr.Dropdown(
                                choices=profile_choices,
                                value=default_profile,
                                label="📝 Translation Profile"
                            )
                            
                            epub_system_prompt = gr.Textbox(
                                label="System Prompt (Translation Instructions)",
                                lines=8,
                                max_lines=15,
                                interactive=True,
                                placeholder="Select a profile to load translation instructions...",
                                value=self.profiles.get(default_profile, '') if default_profile else ''
                            )
                            
                            with gr.Accordion("⚙️ Advanced Settings", open=False):
                                epub_temperature = gr.Slider(
                                    minimum=0,
                                    maximum=1,
                                    value=self.get_config_value('temperature', 0.3),
                                    step=0.1,
                                    label="Temperature"
                                )
                                
                                glossary_file = gr.File(
                                    label="📋 Glossary CSV (optional)",
                                    file_types=[".csv"]
                                )
                        
                        with gr.Column():
                            # Add logo and status at top
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
                                epub_status_message = gr.Markdown(
                                    value="### Ready to translate\nUpload an EPUB file and click 'Translate EPUB' to begin.",
                                    visible=True
                                )
                            
                            # Progress section (similar to manga tab)
                            with gr.Group(visible=False) as epub_progress_group:
                                gr.Markdown("### Progress")
                                epub_progress_text = gr.Textbox(
                                    label="📨 Current Status",
                                    value="Ready to start",
                                    interactive=False,
                                    lines=1
                                )
                                epub_progress_bar = gr.Slider(
                                    minimum=0,
                                    maximum=100,
                                    value=0,
                                    step=1,
                                    label="📋 Translation Progress",
                                    interactive=False,
                                    show_label=True
                                )
                            
                            epub_logs = gr.Textbox(
                                label="📋 Translation Logs",
                                lines=20,
                                max_lines=30,
                                value="Ready to translate. Upload an EPUB file and configure settings.",
                                visible=True,
                                interactive=False
                            )
                            
                            epub_output = gr.File(
                                label="📥 Download Translated EPUB",
                                visible=False
                            )
                            
                            epub_status = gr.Textbox(
                                label="Final Status",
                                lines=3,
                                max_lines=5,
                                visible=False,
                                interactive=False
                            )
                    
                    # Sync handlers will be connected after manga components are created
                    
                    # Translation button handler - now with progress outputs
                    translate_btn.click(
                        fn=self.translate_epub,
                        inputs=[
                            epub_file,
                            epub_model,
                            epub_api_key,
                            epub_profile,
                            epub_system_prompt,
                            epub_temperature,
                            glossary_file
                        ],
                        outputs=[
                            epub_output,          # Download file
                            epub_status_message,  # Top status message
                            epub_progress_group,  # Progress group visibility
                            epub_logs,            # Translation logs
                            epub_status,          # Final status
                            epub_progress_text,   # Progress text
                            epub_progress_bar     # Progress bar
                        ]
                    )
                
                # Manga Translation Tab
                with gr.Tab("🎨 Manga Translation"):
                    with gr.Row():
                        with gr.Column():
                            manga_images = gr.File(
                                label="🖼️ Upload Manga Images or CBZ",
                                file_types=[".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".cbz", ".zip"],
                                file_count="multiple"
                            )
                            
                            with gr.Row():
                                translate_manga_btn = gr.Button(
                                    "🚀 Translate Manga",
                                    variant="primary",
                                    size="lg",
                                    scale=2
                                )
                                
                                stop_manga_btn = gr.Button(
                                    "⏹️ Stop Translation",
                                    variant="stop",
                                    size="lg",
                                    visible=False,
                                    scale=1
                                )
                            
                            manga_model = gr.Dropdown(
                                choices=self.models,
                                value=self.get_config_value('model', 'gpt-4-turbo'),
                                label="🤖 AI Model",
                                interactive=True,
                                allow_custom_value=True,
                                filterable=True
                            )
                            
                            manga_api_key = gr.Textbox(
                                label="🔑 API Key",
                                type="password",
                                placeholder="Enter your API key",
                                value=self.get_config_value('api_key', '')  # Pre-fill from config
                            )
                            
                            # Use all profiles without filtering
                            profile_choices = list(self.profiles.keys())
                            # Use the active profile from config, same as EPUB tab
                            default_profile = self.get_config_value('active_profile', profile_choices[0] if profile_choices else '')
                            
                            manga_profile = gr.Dropdown(
                                choices=profile_choices,
                                value=default_profile,
                                label="📝 Translation Profile"
                            )
                            
                            # Editable manga system prompt
                            manga_system_prompt = gr.Textbox(
                                label="Manga System Prompt (Translation Instructions)",
                                lines=8,
                                max_lines=15,
                                interactive=True,
                                placeholder="Select a manga profile to load translation instructions...",
                                value=self.profiles.get(default_profile, '') if default_profile else ''
                            )
                            
                            with gr.Accordion("⚙️ OCR Settings", open=False):
                                gr.Markdown("🔒 **Credentials are auto-saved** to your config (encrypted) after first use.")
                                
                                ocr_provider = gr.Radio(
                                    choices=["google", "azure", "custom-api"],
                                    value=self.get_config_value('ocr_provider', 'custom-api'),
                                    label="OCR Provider"
                                )
                                
                                # Show saved Google credentials path if available
                                saved_google_path = self.get_config_value('google_vision_credentials', '')
                                if saved_google_path and os.path.exists(saved_google_path):
                                    gr.Markdown(f"✅ **Saved credentials found:** `{os.path.basename(saved_google_path)}`")
                                    gr.Markdown("💡 *Using saved credentials. Upload a new file only if you want to change them.*")
                                else:
                                    gr.Markdown("⚠️ No saved Google credentials found. Please upload your JSON file.")
                                
                                # Note: File component doesn't support pre-filling paths due to browser security
                                google_creds = gr.File(
                                    label="Google Cloud Credentials JSON (upload to update)",
                                    file_types=[".json"]
                                )
                                
                                azure_key = gr.Textbox(
                                    label="Azure Vision API Key (if using Azure)",
                                    type="password",
                                    placeholder="Enter Azure API key",
                                    value=self.get_config_value('azure_vision_key', '')
                                )
                                
                                azure_endpoint = gr.Textbox(
                                    label="Azure Vision Endpoint (if using Azure)",
                                    placeholder="https://your-resource.cognitiveservices.azure.com/",
                                    value=self.get_config_value('azure_vision_endpoint', '')
                                )
                                
                                bubble_detection = gr.Checkbox(
                                    label="Enable Bubble Detection",
                                    value=self.get_config_value('bubble_detection_enabled', True)
                                )
                                
                                inpainting = gr.Checkbox(
                                    label="Enable Text Removal (Inpainting)",
                                    value=self.get_config_value('inpainting_enabled', True)
                                )
                            
                            with gr.Accordion("⚡ Parallel Processing", open=False):
                                gr.Markdown("### Parallel Panel Translation")
                                gr.Markdown("*Process multiple panels simultaneously for faster translation*")
                                
                                parallel_panel_translation = gr.Checkbox(
                                    label="Enable Parallel Panel Translation",
                                    value=self.get_config_value('manga_settings', {}).get('advanced', {}).get('parallel_panel_translation', False),
                                    info="Translates multiple panels at once instead of sequentially"
                                )
                                
                                panel_max_workers = gr.Slider(
                                    minimum=1,
                                    maximum=20,
                                    value=self.get_config_value('manga_settings', {}).get('advanced', {}).get('panel_max_workers', 7),
                                    step=1,
                                    label="Max concurrent panels",
                                    interactive=True,
                                    info="Number of panels to process simultaneously (higher = faster but more memory)"
                                )
                            
                            with gr.Accordion("✨ Text Visibility Settings", open=False):
                                gr.Markdown("### Font Settings")
                                
                                font_size_mode = gr.Radio(
                                    choices=["auto", "fixed", "multiplier"],
                                    value=self.get_config_value('manga_font_size_mode', 'auto'),
                                    label="Font Size Mode"
                                )
                                
                                font_size = gr.Slider(
                                    minimum=0,
                                    maximum=72,
                                    value=self.get_config_value('manga_font_size', 24),
                                    step=1,
                                    label="Fixed Font Size (0=auto, used when mode=fixed)"
                                )
                                
                                font_multiplier = gr.Slider(
                                    minimum=0.5,
                                    maximum=2.0,
                                    value=self.get_config_value('manga_font_size_multiplier', 1.0),
                                    step=0.1,
                                    label="Font Size Multiplier (when mode=multiplier)"
                                )
                                
                                min_font_size = gr.Slider(
                                    minimum=0,
                                    maximum=100,
                                    value=self.get_config_value('manga_settings', {}).get('rendering', {}).get('auto_min_size', 12),
                                    step=1,
                                    label="Minimum Font Size (0=no limit)"
                                )
                                
                                max_font_size = gr.Slider(
                                    minimum=20,
                                    maximum=100,
                                    value=self.get_config_value('manga_max_font_size', 48),
                                    step=1,
                                    label="Maximum Font Size"
                                )
                                
                                gr.Markdown("### Text Color")
                                
                                text_color_rgb = gr.ColorPicker(
                                    label="Font Color",
                                    value=self.get_config_value('manga_text_color', '#000000')  # Default black
                                )
                                
                                gr.Markdown("### Shadow Settings")
                                
                                shadow_enabled = gr.Checkbox(
                                    label="Enable Text Shadow",
                                    value=self.get_config_value('manga_shadow_enabled', True)
                                )
                                
                                shadow_color = gr.ColorPicker(
                                    label="Shadow Color",
                                    value=self.get_config_value('manga_shadow_color', '#FFFFFF')  # Default white
                                )
                                
                                shadow_offset_x = gr.Slider(
                                    minimum=-10,
                                    maximum=10,
                                    value=self.get_config_value('manga_shadow_offset_x', 2),
                                    step=1,
                                    label="Shadow Offset X"
                                )
                                
                                shadow_offset_y = gr.Slider(
                                    minimum=-10,
                                    maximum=10,
                                    value=self.get_config_value('manga_shadow_offset_y', 2),
                                    step=1,
                                    label="Shadow Offset Y"
                                )
                                
                                shadow_blur = gr.Slider(
                                    minimum=0,
                                    maximum=10,
                                    value=self.get_config_value('manga_shadow_blur', 0),
                                    step=1,
                                    label="Shadow Blur"
                                )
                                
                                gr.Markdown("### Background Settings")
                                
                                bg_opacity = gr.Slider(
                                    minimum=0,
                                    maximum=255,
                                    value=self.get_config_value('manga_bg_opacity', 130),
                                    step=1,
                                    label="Background Opacity"
                                )
                                
                            # Ensure bg_style value is valid
                            bg_style_value = self.get_config_value('manga_bg_style', 'circle')
                            if bg_style_value not in ["box", "circle", "wrap"]:
                                bg_style_value = 'circle'  # Default fallback
                            
                            bg_style = gr.Radio(
                                choices=["box", "circle", "wrap"],
                                value=bg_style_value,
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
                            
                            # Progress section for manga translation (similar to manga integration script)
                            with gr.Group(visible=False) as manga_progress_group:
                                gr.Markdown("### Progress")
                                manga_progress_text = gr.Textbox(
                                    label="📈 Current Status",
                                    value="Ready to start",
                                    interactive=False,
                                    lines=1
                                )
                                manga_progress_bar = gr.Slider(
                                    minimum=0,
                                    maximum=100,
                                    value=0,
                                    step=1,
                                    label="📋 Translation Progress",
                                    interactive=False,
                                    show_label=True
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
                    
                    # Global sync flag to prevent loops
                    self._syncing_active = False
                    
                    # Auto-save Azure credentials on change
                    def save_azure_credentials(key, endpoint):
                        """Save Azure credentials to config"""
                        try:
                            current_config = self.get_current_config_for_update()
                            # Don't decrypt - just update what we need
                            if key and key.strip():
                                current_config['azure_vision_key'] = str(key).strip()
                            if endpoint and endpoint.strip():
                                current_config['azure_vision_endpoint'] = str(endpoint).strip()
                            self.save_config(current_config)
                            return None
                        except Exception as e:
                            print(f"Failed to save Azure credentials: {e}")
                            return None
                    
                    azure_key.change(
                        fn=lambda k, e: save_azure_credentials(k, e),
                        inputs=[azure_key, azure_endpoint],
                        outputs=None
                    )
                    
                    azure_endpoint.change(
                        fn=lambda k, e: save_azure_credentials(k, e),
                        inputs=[azure_key, azure_endpoint],
                        outputs=None
                    )
                    
                    # Auto-save OCR provider on change
                    def save_ocr_provider(provider):
                        """Save OCR provider to config"""
                        try:
                            current_config = self.get_current_config_for_update()
                            # Don't decrypt - just update what we need
                            current_config['ocr_provider'] = provider
                            self.save_config(current_config)
                            return None
                        except Exception as e:
                            print(f"Failed to save OCR provider: {e}")
                            return None
                    
                    ocr_provider.change(
                        fn=save_ocr_provider,
                        inputs=[ocr_provider],
                        outputs=None
                    )
                    
                    # Auto-save bubble detection and inpainting on change
                    def save_detection_settings(bubble_det, inpaint):
                        """Save bubble detection and inpainting settings"""
                        try:
                            current_config = self.get_current_config_for_update()
                            # Don't decrypt - just update what we need
                            current_config['bubble_detection_enabled'] = bubble_det
                            current_config['inpainting_enabled'] = inpaint
                            self.save_config(current_config)
                            return None
                        except Exception as e:
                            print(f"Failed to save detection settings: {e}")
                            return None
                    
                    bubble_detection.change(
                        fn=lambda b, i: save_detection_settings(b, i),
                        inputs=[bubble_detection, inpainting],
                        outputs=None
                    )
                    
                    inpainting.change(
                        fn=lambda b, i: save_detection_settings(b, i),
                        inputs=[bubble_detection, inpainting],
                        outputs=None
                    )
                    
                    # Auto-save font size mode on change
                    def save_font_mode(mode):
                        """Save font size mode to config"""
                        try:
                            current_config = self.get_current_config_for_update()
                            # Don't decrypt - just update what we need
                            current_config['manga_font_size_mode'] = mode
                            self.save_config(current_config)
                            return None
                        except Exception as e:
                            print(f"Failed to save font mode: {e}")
                            return None
                    
                    font_size_mode.change(
                        fn=save_font_mode,
                        inputs=[font_size_mode],
                        outputs=None
                    )
                    
                    # Auto-save background style on change
                    def save_bg_style(style):
                        """Save background style to config"""
                        try:
                            current_config = self.get_current_config_for_update()
                            # Don't decrypt - just update what we need
                            current_config['manga_bg_style'] = style
                            self.save_config(current_config)
                            return None
                        except Exception as e:
                            print(f"Failed to save bg style: {e}")
                            return None
                    
                    bg_style.change(
                        fn=save_bg_style,
                        inputs=[bg_style],
                        outputs=None
                    )
                    
                    # Auto-save parallel panel translation settings  
                    def save_parallel_panel_settings(parallel_enabled, max_workers):
                        """Save parallel panel translation settings to config"""
                        try:
                            current_config = self.get_current_config_for_update()
                            # Don't decrypt - just update what we need
                            
                            # Initialize nested structure if not exists
                            if 'manga_settings' not in current_config:
                                current_config['manga_settings'] = {}
                            if 'advanced' not in current_config['manga_settings']:
                                current_config['manga_settings']['advanced'] = {}
                            
                            current_config['manga_settings']['advanced']['parallel_panel_translation'] = parallel_enabled
                            current_config['manga_settings']['advanced']['panel_max_workers'] = int(max_workers)
                            
                            self.save_config(current_config)
                            return None
                        except Exception as e:
                            print(f"Failed to save parallel panel settings: {e}")
                            return None
                    
                    parallel_panel_translation.change(
                        fn=lambda p, w: save_parallel_panel_settings(p, w),
                        inputs=[parallel_panel_translation, panel_max_workers],
                        outputs=None
                    )
                    
                    panel_max_workers.change(
                        fn=lambda p, w: save_parallel_panel_settings(p, w),
                        inputs=[parallel_panel_translation, panel_max_workers],
                        outputs=None
                    )
                    
                    # Comprehensive sync system for ALL components
                    def sync_all_components(source_tab, model, api_key, profile):
                        """Sync all main components between tabs"""
                        if self._syncing_active:
                            return [gr.update()] * 6  # Return no updates to prevent loops
                        
                        try:
                            self._syncing_active = True
                            
                            # Save to config
                            current_config = self.get_current_config_for_update()
                            current_config['model'] = model
                            if api_key:  # Only save non-empty API keys
                                current_config['api_key'] = api_key
                            current_config['active_profile'] = profile
                            self.save_config(current_config)
                            
                            # Get prompt for profile
                            prompt = self.profiles.get(profile, '')
                            
                            # Return values to sync to the OTHER tab
                            # Order: model, api_key, profile, system_prompt for target tab
                            return model, api_key, profile, prompt
                            
                        except Exception as e:
                            print(f"Sync error: {e}")
                            return [gr.update()] * 4
                        finally:
                            self._syncing_active = False
                    
                    # Connect EPUB components to sync to Manga
                    epub_model.change(
                        fn=lambda m, k, p: sync_all_components('epub', m, k, p),
                        inputs=[epub_model, epub_api_key, epub_profile],
                        outputs=[manga_model, manga_api_key, manga_profile, manga_system_prompt]
                    )
                    
                    epub_api_key.change(
                        fn=lambda m, k, p: sync_all_components('epub', m, k, p),
                        inputs=[epub_model, epub_api_key, epub_profile],
                        outputs=[manga_model, manga_api_key, manga_profile, manga_system_prompt]
                    )
                    
                    epub_profile.change(
                        fn=lambda m, k, p: sync_all_components('epub', m, k, p),
                        inputs=[epub_model, epub_api_key, epub_profile],
                        outputs=[manga_model, manga_api_key, manga_profile, manga_system_prompt]
                    )
                    
                    # Also update epub system prompt when profile changes
                    epub_profile.change(
                        fn=lambda p: self.profiles.get(p, ''),
                        inputs=[epub_profile],
                        outputs=[epub_system_prompt]
                    )
                    
                    # Connect Manga components to sync to EPUB
                    manga_model.change(
                        fn=lambda m, k, p: sync_all_components('manga', m, k, p),
                        inputs=[manga_model, manga_api_key, manga_profile],
                        outputs=[epub_model, epub_api_key, epub_profile, epub_system_prompt]
                    )
                    
                    manga_api_key.change(
                        fn=lambda m, k, p: sync_all_components('manga', m, k, p),
                        inputs=[manga_model, manga_api_key, manga_profile],
                        outputs=[epub_model, epub_api_key, epub_profile, epub_system_prompt]
                    )
                    
                    manga_profile.change(
                        fn=lambda m, k, p: sync_all_components('manga', m, k, p),
                        inputs=[manga_model, manga_api_key, manga_profile],
                        outputs=[epub_model, epub_api_key, epub_profile, epub_system_prompt]
                    )
                    
                    # Also update manga system prompt when profile changes
                    manga_profile.change(
                        fn=lambda p: self.profiles.get(p, ''),
                        inputs=[manga_profile],
                        outputs=[manga_system_prompt]
                    )
                    
                    # Save active tab when it changes
                    def save_active_tab(evt: gr.SelectData):
                        """Save the active tab index to config"""
                        try:
                            print(f"Tab changed to index: {evt.index}")
                            current_config = self.get_current_config_for_update()
                            current_config['active_tab_index'] = evt.index
                            self.save_config(current_config)
                        except Exception as e:
                            print(f"Failed to save active tab: {e}")
                    
                    main_tabs.select(
                        fn=save_active_tab,
                        inputs=None,
                        outputs=None
                    )
                    
                    # Generic field saver for all remaining fields
                    def save_field(field_name):
                        def _save(value):
                            if not self._syncing_active:
                                try:
                                    current_config = self.get_current_config_for_update()
                                    current_config[field_name] = value
                                    self.save_config(current_config)
                                except Exception as e:
                                    print(f"Failed to save {field_name}: {e}")
                            # Don't return anything when outputs=None
                        return _save
                    
                    # Save temperature
                    epub_temperature.change(
                        fn=save_field('temperature'),
                        inputs=[epub_temperature],
                        outputs=None
                    )
                    
                    # Save all font/color fields that aren't already saved
                    text_color_rgb.change(
                        fn=save_field('manga_text_color'),
                        inputs=[text_color_rgb],
                        outputs=None
                    )
                    
                    shadow_color.change(
                        fn=save_field('manga_shadow_color'),
                        inputs=[shadow_color],
                        outputs=None
                    )
                    
                    # Translate button click handler
                    translate_manga_btn.click(
                        fn=self.start_manga_translation,
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
                            bg_style,
                            parallel_panel_translation,
                            panel_max_workers
                        ],
                        outputs=[manga_logs, manga_output_image, manga_cbz_output, manga_status, manga_progress_group, manga_progress_text, manga_progress_bar, translate_manga_btn, stop_manga_btn]
                    )
                    
                    # Stop button click handler
                    stop_manga_btn.click(
                        fn=self.stop_manga_translation,
                        inputs=[],
                        outputs=[translate_manga_btn, stop_manga_btn, manga_status]
                    )
                    
                    # Load settings from localStorage on page load
                    def load_settings_from_storage():
                        """Load settings from localStorage or config file"""
                        is_hf_spaces = os.getenv('SPACE_ID') is not None or os.getenv('HF_SPACES') == 'true'
                        
                        if not is_hf_spaces:
                            # Load from config file locally
                            config = self.load_config()
                            # Decrypt API keys if needed
                            if API_KEY_ENCRYPTION_AVAILABLE:
                                config = decrypt_config(config)
                            return [
                                config.get('model', 'gpt-4-turbo'),
                                config.get('api_key', ''),
                                config.get('active_profile', list(self.profiles.keys())[0] if self.profiles else ''),  # profile
                                self.profiles.get(config.get('active_profile', list(self.profiles.keys())[0] if self.profiles else ''), ''),  # prompt
                                config.get('ocr_provider', 'custom-api'),
                                None,  # google_creds (file component - can't be pre-filled)
                                config.get('azure_vision_key', ''),
                                config.get('azure_vision_endpoint', ''),
                                config.get('bubble_detection_enabled', True),
                                config.get('inpainting_enabled', True),
                                config.get('manga_font_size_mode', 'auto'),
                                config.get('manga_font_size', 24),
                                config.get('manga_font_multiplier', 1.0),
                                config.get('manga_min_font_size', 12),
                                config.get('manga_max_font_size', 48),
                                config.get('manga_text_color', '#FFFFFF'),
                                config.get('manga_shadow_enabled', True),
                                config.get('manga_shadow_color', '#000000'),
                                config.get('manga_shadow_offset_x', 1),
                                config.get('manga_shadow_offset_y', 1),
                                config.get('manga_shadow_blur', 2),
                                config.get('manga_bg_opacity', 180),
                                config.get('manga_bg_style', 'auto'),
                                config.get('manga_settings', {}).get('advanced', {}).get('parallel_panel_translation', False),
                                config.get('manga_settings', {}).get('advanced', {}).get('panel_max_workers', 7)
                            ]
                        else:
                            # For HF Spaces, return defaults (will be overridden by JS)
                            return [
                                'gpt-4-turbo',  # model
                                '',  # api_key
                                list(self.profiles.keys())[0] if self.profiles else '',  # profile
                                self.profiles.get(list(self.profiles.keys())[0] if self.profiles else '', ''),  # prompt
                                'custom-api',  # ocr_provider
                                None,  # google_creds (file component - can't be pre-filled)
                                '',  # azure_key
                                '',  # azure_endpoint
                                True,  # bubble_detection
                                True,  # inpainting
                                'auto',  # font_size_mode
                                24,  # font_size
                                1.0,  # font_multiplier
                                12,  # min_font_size
                                48,  # max_font_size
                                '#FFFFFF',  # text_color
                                True,  # shadow_enabled
                                '#000000',  # shadow_color
                                1,  # shadow_offset_x
                                1,  # shadow_offset_y
                                2,  # shadow_blur
                                180,  # bg_opacity
                                'auto',  # bg_style
                                False,  # parallel_panel_translation
                                7  # panel_max_workers
                            ]
                    
                    # Store references for load handler
                    self.manga_components = {
                        'model': manga_model,
                        'api_key': manga_api_key,
                        'profile': manga_profile,
                        'prompt': manga_system_prompt,
                        'ocr_provider': ocr_provider,
                        'google_creds': google_creds,
                        'azure_key': azure_key,
                        'azure_endpoint': azure_endpoint,
                        'bubble_detection': bubble_detection,
                        'inpainting': inpainting,
                        'font_size_mode': font_size_mode,
                        'font_size': font_size,
                        'font_multiplier': font_multiplier,
                        'min_font_size': min_font_size,
                        'max_font_size': max_font_size,
                        'text_color_rgb': text_color_rgb,
                        'shadow_enabled': shadow_enabled,
                        'shadow_color': shadow_color,
                        'shadow_offset_x': shadow_offset_x,
                        'shadow_offset_y': shadow_offset_y,
                        'shadow_blur': shadow_blur,
                        'bg_opacity': bg_opacity,
                        'bg_style': bg_style,
                        'parallel_panel_translation': parallel_panel_translation,
                        'panel_max_workers': panel_max_workers
                    }
                    self.load_settings_fn = load_settings_from_storage
                
                # Manga Settings Tab - NEW
                with gr.Tab("🎬 Manga Settings"):
                    gr.Markdown("### Advanced Manga Translation Settings")
                    gr.Markdown("Configure bubble detection, inpainting, preprocessing, and rendering options.")
                    
                    with gr.Accordion("🕹️ Bubble Detection & Inpainting", open=True):
                        gr.Markdown("#### Bubble Detection")
                        
                        detector_type = gr.Radio(
                            choices=["rtdetr_onnx", "rtdetr", "yolo"],
                            value=self.get_config_value('manga_settings', {}).get('ocr', {}).get('detector_type', 'rtdetr_onnx'),
                            label="Detector Type",
                            interactive=True
                        )
                        
                        rtdetr_confidence = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=self.get_config_value('manga_settings', {}).get('ocr', {}).get('rtdetr_confidence', 0.3),
                            step=0.05,
                            label="RT-DETR Confidence Threshold",
                            interactive=True
                        )
                        
                        bubble_confidence = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=self.get_config_value('manga_settings', {}).get('ocr', {}).get('bubble_confidence', 0.3),
                            step=0.05,
                            label="YOLO Bubble Confidence Threshold",
                            interactive=True
                        )
                        
                        detect_text_bubbles = gr.Checkbox(
                            label="Detect Text Bubbles",
                            value=self.get_config_value('manga_settings', {}).get('ocr', {}).get('detect_text_bubbles', True)
                        )
                        
                        detect_empty_bubbles = gr.Checkbox(
                            label="Detect Empty Bubbles",
                            value=self.get_config_value('manga_settings', {}).get('ocr', {}).get('detect_empty_bubbles', True)
                        )
                        
                        detect_free_text = gr.Checkbox(
                            label="Detect Free Text (outside bubbles)",
                            value=self.get_config_value('manga_settings', {}).get('ocr', {}).get('detect_free_text', True)
                        )
                        
                        bubble_max_detections = gr.Slider(
                            minimum=1,
                            maximum=2000,
                            value=self.get_config_value('manga_settings', {}).get('ocr', {}).get('bubble_max_detections_yolo', 100),
                            step=1,
                            label="Max detections (YOLO only)",
                            interactive=True,
                            info="Maximum number of bubble detections for YOLO detector"
                        )
                        
                        gr.Markdown("#### Inpainting")
                        
                        local_inpaint_method = gr.Radio(
                            choices=["anime_onnx", "anime", "lama", "lama_onnx", "aot", "aot_onnx"],
                            value=self.get_config_value('manga_settings', {}).get('inpainting', {}).get('local_method', 'anime_onnx'),
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
                            value=self.get_config_value('manga_settings', {}).get('auto_iterations', True)
                        )
                        
                        mask_dilation = gr.Slider(
                            minimum=0,
                            maximum=20,
                            value=self.get_config_value('manga_settings', {}).get('mask_dilation', 0),
                            step=1,
                            label="General Mask Dilation",
                            interactive=True
                        )
                        
                        text_bubble_dilation = gr.Slider(
                            minimum=0,
                            maximum=20,
                            value=self.get_config_value('manga_settings', {}).get('text_bubble_dilation_iterations', 2),
                            step=1,
                            label="Text Bubble Dilation Iterations",
                            interactive=True
                        )
                        
                        empty_bubble_dilation = gr.Slider(
                            minimum=0,
                            maximum=20,
                            value=self.get_config_value('manga_settings', {}).get('empty_bubble_dilation_iterations', 3),
                            step=1,
                            label="Empty Bubble Dilation Iterations",
                            interactive=True
                        )
                        
                        free_text_dilation = gr.Slider(
                            minimum=0,
                            maximum=20,
                            value=self.get_config_value('manga_settings', {}).get('free_text_dilation_iterations', 3),
                            step=1,
                            label="Free Text Dilation Iterations",
                            interactive=True
                        )
                    
                    with gr.Accordion("🖌️ Image Preprocessing", open=False):
                        preprocessing_enabled = gr.Checkbox(
                            label="Enable Preprocessing",
                            value=self.get_config_value('manga_settings', {}).get('preprocessing', {}).get('enabled', False)
                        )
                        
                        auto_detect_quality = gr.Checkbox(
                            label="Auto Detect Image Quality",
                            value=self.get_config_value('manga_settings', {}).get('preprocessing', {}).get('auto_detect_quality', True)
                        )
                        
                        enhancement_strength = gr.Slider(
                            minimum=1.0,
                            maximum=3.0,
                            value=self.get_config_value('manga_settings', {}).get('preprocessing', {}).get('enhancement_strength', 1.5),
                            step=0.1,
                            label="Enhancement Strength",
                            interactive=True
                        )
                        
                        denoise_strength = gr.Slider(
                            minimum=0,
                            maximum=50,
                            value=self.get_config_value('manga_settings', {}).get('preprocessing', {}).get('denoise_strength', 10),
                            step=1,
                            label="Denoise Strength",
                            interactive=True
                        )
                        
                        max_image_dimension = gr.Number(
                            label="Max Image Dimension (pixels)",
                            value=self.get_config_value('manga_settings', {}).get('preprocessing', {}).get('max_image_dimension', 2000),
                            minimum=500
                        )
                        
                        chunk_height = gr.Number(
                            label="Chunk Height for Large Images",
                            value=self.get_config_value('manga_settings', {}).get('preprocessing', {}).get('chunk_height', 1000),
                            minimum=500
                        )
                        
                        gr.Markdown("#### HD Strategy for Inpainting")
                        gr.Markdown("*Controls how large images are processed during inpainting*")
                        
                        hd_strategy = gr.Radio(
                            choices=["original", "resize", "crop"],
                            value=self.get_config_value('manga_settings', {}).get('advanced', {}).get('hd_strategy', 'resize'),
                            label="HD Strategy",
                            interactive=True,
                            info="original = legacy full-image; resize/crop = faster"
                        )
                        
                        hd_strategy_resize_limit = gr.Slider(
                            minimum=512,
                            maximum=4096,
                            value=self.get_config_value('manga_settings', {}).get('advanced', {}).get('hd_strategy_resize_limit', 1536),
                            step=64,
                            label="Resize Limit (long edge, px)",
                            info="For resize strategy",
                            interactive=True
                        )
                        
                        hd_strategy_crop_margin = gr.Slider(
                            minimum=0,
                            maximum=256,
                            value=self.get_config_value('manga_settings', {}).get('advanced', {}).get('hd_strategy_crop_margin', 16),
                            step=2,
                            label="Crop Margin (px)",
                            info="For crop strategy",
                            interactive=True
                        )
                        
                        hd_strategy_crop_trigger = gr.Slider(
                            minimum=256,
                            maximum=4096,
                            value=self.get_config_value('manga_settings', {}).get('advanced', {}).get('hd_strategy_crop_trigger_size', 1024),
                            step=64,
                            label="Crop Trigger Size (px)",
                            info="Apply crop only if long edge exceeds this",
                            interactive=True
                        )
                        
                        gr.Markdown("#### Image Tiling")
                        gr.Markdown("*Alternative tiling strategy (note: HD Strategy takes precedence)*")
                        
                        tiling_enabled = gr.Checkbox(
                            label="Enable Tiling",
                            value=self.get_config_value('manga_settings', {}).get('tiling', {}).get('enabled', False)
                        )
                        
                        tiling_tile_size = gr.Slider(
                            minimum=256,
                            maximum=1024,
                            value=self.get_config_value('manga_settings', {}).get('tiling', {}).get('tile_size', 480),
                            step=64,
                            label="Tile Size (px)",
                            interactive=True
                        )
                        
                        tiling_tile_overlap = gr.Slider(
                            minimum=0,
                            maximum=128,
                            value=self.get_config_value('manga_settings', {}).get('tiling', {}).get('tile_overlap', 64),
                            step=16,
                            label="Tile Overlap (px)",
                            interactive=True
                        )
                    
                    with gr.Accordion("🎨 Font & Text Rendering", open=False):
                        gr.Markdown("#### Font Sizing Algorithm")
                        
                        font_algorithm = gr.Radio(
                            choices=["smart", "simple"],
                            value=self.get_config_value('manga_settings', {}).get('font_sizing', {}).get('algorithm', 'smart'),
                            label="Font Sizing Algorithm",
                            interactive=True
                        )
                        
                        prefer_larger = gr.Checkbox(
                            label="Prefer Larger Fonts",
                            value=self.get_config_value('manga_settings', {}).get('font_sizing', {}).get('prefer_larger', True)
                        )
                        
                        max_lines = gr.Slider(
                            minimum=1,
                            maximum=20,
                            value=self.get_config_value('manga_settings', {}).get('font_sizing', {}).get('max_lines', 10),
                            step=1,
                            label="Maximum Lines Per Bubble",
                            interactive=True
                        )
                        
                        line_spacing = gr.Slider(
                            minimum=0.5,
                            maximum=3.0,
                            value=self.get_config_value('manga_settings', {}).get('font_sizing', {}).get('line_spacing', 1.3),
                            step=0.1,
                            label="Line Spacing Multiplier",
                            interactive=True
                        )
                        
                        bubble_size_factor = gr.Checkbox(
                            label="Use Bubble Size Factor",
                            value=self.get_config_value('manga_settings', {}).get('font_sizing', {}).get('bubble_size_factor', True)
                        )
                        
                        auto_fit_style = gr.Radio(
                            choices=["balanced", "aggressive", "conservative"],
                            value=self.get_config_value('manga_settings', {}).get('rendering', {}).get('auto_fit_style', 'balanced'),
                            label="Auto Fit Style",
                            interactive=True
                        )
                    
                    with gr.Accordion("⚙️ Advanced Options", open=False):
                        gr.Markdown("#### Format Detection")
                        
                        format_detection = gr.Checkbox(
                            label="Enable Format Detection (manga/webtoon)",
                            value=self.get_config_value('manga_settings', {}).get('advanced', {}).get('format_detection', True)
                        )
                        
                        webtoon_mode = gr.Radio(
                            choices=["auto", "force_manga", "force_webtoon"],
                            value=self.get_config_value('manga_settings', {}).get('advanced', {}).get('webtoon_mode', 'auto'),
                            label="Webtoon Mode",
                            interactive=True
                        )
                        
                        gr.Markdown("#### Inpainting Performance")
                        
                        inpaint_batch_size = gr.Slider(
                            minimum=1,
                            maximum=32,
                            value=self.get_config_value('manga_settings', {}).get('inpainting', {}).get('batch_size', 10),
                            step=1,
                            label="Batch Size",
                            interactive=True,
                            info="Process multiple regions at once"
                        )
                        
                        inpaint_cache_enabled = gr.Checkbox(
                            label="Enable inpainting cache (speeds up repeated processing)",
                            value=self.get_config_value('manga_settings', {}).get('inpainting', {}).get('enable_cache', True)
                        )
                        
                        gr.Markdown("#### Performance")
                        
                        parallel_processing = gr.Checkbox(
                            label="Enable Parallel Processing",
                            value=self.get_config_value('manga_settings', {}).get('advanced', {}).get('parallel_processing', True)
                        )
                        
                        max_workers = gr.Slider(
                            minimum=1,
                            maximum=8,
                            value=self.get_config_value('manga_settings', {}).get('advanced', {}).get('max_workers', 2),
                            step=1,
                            label="Max Worker Threads",
                            interactive=True
                        )
                        
                        gr.Markdown("**⚡ Advanced Performance**")
                        
                        preload_local_inpainting = gr.Checkbox(
                            label="Preload local inpainting instances for panel-parallel runs",
                            value=self.get_config_value('manga_settings', {}).get('advanced', {}).get('preload_local_inpainting_for_panels', True),
                            info="Preloads inpainting models to speed up parallel processing"
                        )
                        
                        panel_start_stagger = gr.Slider(
                            minimum=0,
                            maximum=1000,
                            value=self.get_config_value('manga_settings', {}).get('advanced', {}).get('panel_start_stagger_ms', 30),
                            step=10,
                            label="Panel start stagger",
                            interactive=True,
                            info="Milliseconds delay between panel starts"
                        )
                        
                        gr.Markdown("#### Model Optimization")
                        
                        torch_precision = gr.Radio(
                            choices=["fp32", "fp16"],
                            value=self.get_config_value('manga_settings', {}).get('advanced', {}).get('torch_precision', 'fp16'),
                            label="Torch Precision",
                            interactive=True
                        )
                        
                        auto_cleanup_models = gr.Checkbox(
                            label="Auto Cleanup Models from Memory",
                            value=self.get_config_value('manga_settings', {}).get('advanced', {}).get('auto_cleanup_models', False)
                        )
                        
                        gr.Markdown("#### Debug Options")
                        
                        debug_mode = gr.Checkbox(
                            label="Enable Debug Mode",
                            value=self.get_config_value('manga_settings', {}).get('advanced', {}).get('debug_mode', False)
                        )
                        
                        save_intermediate = gr.Checkbox(
                            label="Save Intermediate Files",
                            value=self.get_config_value('manga_settings', {}).get('advanced', {}).get('save_intermediate', False)
                        )
                        
                        concise_pipeline_logs = gr.Checkbox(
                            label="Concise Pipeline Logs",
                            value=self.get_config_value('concise_pipeline_logs', True)
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
                    
                    # Auto-save parallel panel translation settings
                    def save_parallel_settings(preload_enabled, parallel_enabled, max_workers, stagger_ms):
                        """Save parallel panel translation settings to config"""
                        try:
                            current_config = self.get_current_config_for_update()
                            # Don't decrypt - just update what we need
                            
                            # Initialize nested structure if not exists
                            if 'manga_settings' not in current_config:
                                current_config['manga_settings'] = {}
                            if 'advanced' not in current_config['manga_settings']:
                                current_config['manga_settings']['advanced'] = {}
                            
                            current_config['manga_settings']['advanced']['preload_local_inpainting_for_panels'] = bool(preload_enabled)
                            current_config['manga_settings']['advanced']['parallel_panel_translation'] = bool(parallel_enabled)
                            current_config['manga_settings']['advanced']['panel_max_workers'] = int(max_workers)
                            current_config['manga_settings']['advanced']['panel_start_stagger_ms'] = int(stagger_ms)
                            
                            self.save_config(current_config)
                            return None
                        except Exception as e:
                            print(f"Failed to save parallel panel settings: {e}")
                            return None
                    
                    # Auto-save inpainting performance settings
                    def save_inpainting_settings(batch_size, cache_enabled):
                        """Save inpainting performance settings to config"""
                        try:
                            current_config = self.get_current_config_for_update()
                            # Don't decrypt - just update what we need
                            
                            # Initialize nested structure if not exists
                            if 'manga_settings' not in current_config:
                                current_config['manga_settings'] = {}
                            if 'inpainting' not in current_config['manga_settings']:
                                current_config['manga_settings']['inpainting'] = {}
                            
                            current_config['manga_settings']['inpainting']['batch_size'] = int(batch_size)
                            current_config['manga_settings']['inpainting']['enable_cache'] = bool(cache_enabled)
                            
                            self.save_config(current_config)
                            return None
                        except Exception as e:
                            print(f"Failed to save inpainting settings: {e}")
                            return None
                    
                    # Auto-save preload local inpainting setting
                    def save_preload_setting(preload_enabled):
                        """Save preload local inpainting setting to config"""
                        try:
                            current_config = self.get_current_config_for_update()
                            # Don't decrypt - just update what we need
                            
                            # Initialize nested structure if not exists
                            if 'manga_settings' not in current_config:
                                current_config['manga_settings'] = {}
                            if 'advanced' not in current_config['manga_settings']:
                                current_config['manga_settings']['advanced'] = {}
                            
                            current_config['manga_settings']['advanced']['preload_local_inpainting_for_panels'] = bool(preload_enabled)
                            
                            self.save_config(current_config)
                            return None
                        except Exception as e:
                            print(f"Failed to save preload setting: {e}")
                            return None
                    
                    # Auto-save bubble detection settings
                    def save_bubble_detection_settings(detector_type_val, rtdetr_conf, bubble_conf, detect_text, detect_empty, detect_free, max_detections, local_method_val):
                        """Save bubble detection settings to config"""
                        try:
                            current_config = self.get_current_config_for_update()
                            # Don't decrypt - just update what we need
                            
                            # Initialize nested structure
                            if 'manga_settings' not in current_config:
                                current_config['manga_settings'] = {}
                            if 'ocr' not in current_config['manga_settings']:
                                current_config['manga_settings']['ocr'] = {}
                            if 'inpainting' not in current_config['manga_settings']:
                                current_config['manga_settings']['inpainting'] = {}
                            
                            # Save bubble detection settings
                            current_config['manga_settings']['ocr']['detector_type'] = detector_type_val
                            current_config['manga_settings']['ocr']['rtdetr_confidence'] = float(rtdetr_conf)
                            current_config['manga_settings']['ocr']['bubble_confidence'] = float(bubble_conf)
                            current_config['manga_settings']['ocr']['detect_text_bubbles'] = bool(detect_text)
                            current_config['manga_settings']['ocr']['detect_empty_bubbles'] = bool(detect_empty)
                            current_config['manga_settings']['ocr']['detect_free_text'] = bool(detect_free)
                            current_config['manga_settings']['ocr']['bubble_max_detections_yolo'] = int(max_detections)
                            
                            # Save inpainting method
                            current_config['manga_settings']['inpainting']['local_method'] = local_method_val
                            
                            self.save_config(current_config)
                            return None
                        except Exception as e:
                            print(f"Failed to save bubble detection settings: {e}")
                            return None
                    
                    # Auto-save handlers for bubble detection settings
                    detector_type.change(
                        fn=lambda dt, rc, bc, det_text, det_empty, det_free, max_det, local_method: save_bubble_detection_settings(dt, rc, bc, det_text, det_empty, det_free, max_det, local_method),
                        inputs=[detector_type, rtdetr_confidence, bubble_confidence, detect_text_bubbles, detect_empty_bubbles, detect_free_text, bubble_max_detections, local_inpaint_method],
                        outputs=None
                    )
                    
                    rtdetr_confidence.change(
                        fn=lambda dt, rc, bc, det_text, det_empty, det_free, max_det, local_method: save_bubble_detection_settings(dt, rc, bc, det_text, det_empty, det_free, max_det, local_method),
                        inputs=[detector_type, rtdetr_confidence, bubble_confidence, detect_text_bubbles, detect_empty_bubbles, detect_free_text, bubble_max_detections, local_inpaint_method],
                        outputs=None
                    )
                    
                    bubble_confidence.change(
                        fn=lambda dt, rc, bc, det_text, det_empty, det_free, max_det, local_method: save_bubble_detection_settings(dt, rc, bc, det_text, det_empty, det_free, max_det, local_method),
                        inputs=[detector_type, rtdetr_confidence, bubble_confidence, detect_text_bubbles, detect_empty_bubbles, detect_free_text, bubble_max_detections, local_inpaint_method],
                        outputs=None
                    )
                    
                    detect_text_bubbles.change(
                        fn=lambda dt, rc, bc, det_text, det_empty, det_free, max_det, local_method: save_bubble_detection_settings(dt, rc, bc, det_text, det_empty, det_free, max_det, local_method),
                        inputs=[detector_type, rtdetr_confidence, bubble_confidence, detect_text_bubbles, detect_empty_bubbles, detect_free_text, bubble_max_detections, local_inpaint_method],
                        outputs=None
                    )
                    
                    detect_empty_bubbles.change(
                        fn=lambda dt, rc, bc, det_text, det_empty, det_free, max_det, local_method: save_bubble_detection_settings(dt, rc, bc, det_text, det_empty, det_free, max_det, local_method),
                        inputs=[detector_type, rtdetr_confidence, bubble_confidence, detect_text_bubbles, detect_empty_bubbles, detect_free_text, bubble_max_detections, local_inpaint_method],
                        outputs=None
                    )
                    
                    detect_free_text.change(
                        fn=lambda dt, rc, bc, det_text, det_empty, det_free, max_det, local_method: save_bubble_detection_settings(dt, rc, bc, det_text, det_empty, det_free, max_det, local_method),
                        inputs=[detector_type, rtdetr_confidence, bubble_confidence, detect_text_bubbles, detect_empty_bubbles, detect_free_text, bubble_max_detections, local_inpaint_method],
                        outputs=None
                    )
                    
                    bubble_max_detections.change(
                        fn=lambda dt, rc, bc, det_text, det_empty, det_free, max_det, local_method: save_bubble_detection_settings(dt, rc, bc, det_text, det_empty, det_free, max_det, local_method),
                        inputs=[detector_type, rtdetr_confidence, bubble_confidence, detect_text_bubbles, detect_empty_bubbles, detect_free_text, bubble_max_detections, local_inpaint_method],
                        outputs=None
                    )
                    
                    local_inpaint_method.change(
                        fn=lambda dt, rc, bc, det_text, det_empty, det_free, max_det, local_method: save_bubble_detection_settings(dt, rc, bc, det_text, det_empty, det_free, max_det, local_method),
                        inputs=[detector_type, rtdetr_confidence, bubble_confidence, detect_text_bubbles, detect_empty_bubbles, detect_free_text, bubble_max_detections, local_inpaint_method],
                        outputs=None
                    )
                    
                    # Auto-save handlers for inpainting performance settings
                    inpaint_batch_size.change(
                        fn=lambda bs, ce: save_inpainting_settings(bs, ce),
                        inputs=[inpaint_batch_size, inpaint_cache_enabled],
                        outputs=None
                    )
                    
                    inpaint_cache_enabled.change(
                        fn=lambda bs, ce: save_inpainting_settings(bs, ce),
                        inputs=[inpaint_batch_size, inpaint_cache_enabled],
                        outputs=None
                    )
                    
                    # Auto-save handler for preload local inpainting setting
                    preload_local_inpainting.change(
                        fn=lambda pl: save_preload_setting(pl),
                        inputs=[preload_local_inpainting],
                        outputs=None
                    )
                    
                    gr.Markdown("\n---\n**Note:** These settings will be saved to your config and applied to all manga translations.")
                
                # Glossary Extraction Tab
                with gr.Tab("📝 Glossary Extraction"):
                    with gr.Row():
                        with gr.Column():
                            glossary_epub = gr.File(
                                label="📖 Upload EPUB File",
                                file_types=[".epub"]
                            )
                            
                            extract_btn = gr.Button(
                                "🔍 Extract Glossary",
                                variant="primary",
                                size="lg"
                            )
                            
                            glossary_model = gr.Dropdown(
                                choices=self.models,
                                value=self.get_config_value('model', 'gpt-4-turbo'),
                                label="🤖 AI Model",
                                interactive=True,
                                allow_custom_value=True,
                                filterable=True
                            )
                            
                            glossary_api_key = gr.Textbox(
                                label="🔑 API Key",
                                type="password",
                                placeholder="Enter your API key",
                                value=self.get_config_value('api_key', '')
                            )
                            
                            with gr.Accordion("⚙️ Extraction Settings", open=True):
                                min_freq = gr.Slider(
                                    minimum=1,
                                    maximum=10,
                                    value=2,
                                    step=1,
                                    label="Minimum Frequency",
                                    info="Minimum times a name must appear to be included"
                                )
                                
                                max_names_slider = gr.Slider(
                                    minimum=10,
                                    maximum=200,
                                    value=50,
                                    step=10,
                                    label="Max Character Names",
                                    info="Maximum number of character names to extract"
                                )
                        
                        with gr.Column():
                            # Add logo and status at top
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
                                glossary_status_message = gr.Markdown(
                                    value="### Ready to extract\nUpload an EPUB file and click 'Extract Glossary' to begin.",
                                    visible=True
                                )
                            
                            glossary_output = gr.File(
                                label="📥 Download Glossary CSV",
                                visible=False
                            )
                            
                            glossary_status = gr.Textbox(
                                label="📋 Extraction Status",
                                lines=20,
                                max_lines=30,
                                value="Ready to extract. Upload an EPUB file and configure settings.",
                                interactive=False
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
                    gr.Markdown("Profiles are loaded from your `config_web.json` file. The web interface has its own separate configuration.")
                    
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
                                value=self.get_config_value('thread_submission_delay', 0.1),
                                step=0.1,
                                label="Threading delay (s)",
                                interactive=True
                            )
                            
                            api_delay = gr.Slider(
                                minimum=0,
                                maximum=10,
                                value=self.get_config_value('delay', 1),
                                step=0.5,
                                label="API call delay (s)",
                                interactive=True
                            )
                            
                            chapter_range = gr.Textbox(
                                label="Chapter range (e.g., 5-10)",
                                value=self.get_config_value('chapter_range', ''),
                                placeholder="Leave empty for all chapters"
                            )
                            
                            token_limit = gr.Number(
                                label="Input Token limit",
                                value=self.get_config_value('token_limit', 200000),
                                minimum=0
                            )
                            
                            disable_token_limit = gr.Checkbox(
                                label="Disable Input Token Limit",
                                value=self.get_config_value('token_limit_disabled', False)
                            )
                            
                            output_token_limit = gr.Number(
                                label="Output Token limit",
                                value=self.get_config_value('max_output_tokens', 16000),
                                minimum=0
                            )
                        
                        with gr.Column():
                            contextual = gr.Checkbox(
                                label="Contextual Translation",
                                value=self.get_config_value('contextual', False)
                            )
                            
                            history_limit = gr.Number(
                                label="Translation History Limit",
                                value=self.get_config_value('translation_history_limit', 2),
                                minimum=0
                            )
                            
                            rolling_history = gr.Checkbox(
                                label="Rolling History Window",
                                value=self.get_config_value('translation_history_rolling', False)
                            )
                            
                            batch_translation = gr.Checkbox(
                                label="Batch Translation",
                                value=self.get_config_value('batch_translation', True)
                            )
                            
                            batch_size = gr.Number(
                                label="Batch Size",
                                value=self.get_config_value('batch_size', 10),
                                minimum=1
                            )
                    
                    gr.Markdown("---")
                    gr.Markdown("🔒 **API keys are encrypted** when saved to config using AES encryption.")
                    
                    save_api_key = gr.Checkbox(
                        label="Save API Key (Encrypted)",
                        value=True
                    )
                    
                    save_status = gr.Textbox(label="Settings Status", value="Settings are automatically saved when changed", interactive=False)
                    
                    # Hidden HTML component for JavaScript execution
                    js_executor = gr.HTML("", visible=False)
                    
                    # Auto-save function for settings tab
                    def save_settings_tab(thread_delay_val, api_delay_val, chapter_range_val, token_limit_val, disable_token_limit_val, output_token_limit_val, contextual_val, history_limit_val, rolling_history_val, batch_translation_val, batch_size_val, save_api_key_val):
                        """Save settings from the Settings tab"""
                        try:
                            current_config = self.get_current_config_for_update()
                            # Don't decrypt - just update non-encrypted fields
                            
                            # Update settings
                            current_config['thread_submission_delay'] = float(thread_delay_val)
                            current_config['delay'] = float(api_delay_val)
                            current_config['chapter_range'] = str(chapter_range_val)
                            current_config['token_limit'] = int(token_limit_val)
                            current_config['token_limit_disabled'] = bool(disable_token_limit_val)
                            current_config['max_output_tokens'] = int(output_token_limit_val)
                            current_config['contextual'] = bool(contextual_val)
                            current_config['translation_history_limit'] = int(history_limit_val)
                            current_config['translation_history_rolling'] = bool(rolling_history_val)
                            current_config['batch_translation'] = bool(batch_translation_val)
                            current_config['batch_size'] = int(batch_size_val)
                            
                            # Save to file
                            self.save_config(current_config)
                            
                            # JavaScript to save to localStorage
                            js_code = """
                            <script>
                            (function() {
                                // Save individual settings to localStorage
                                window.saveToLocalStorage('thread_delay', %f);
                                window.saveToLocalStorage('api_delay', %f);
                                window.saveToLocalStorage('chapter_range', '%s');
                                window.saveToLocalStorage('token_limit', %d);
                                window.saveToLocalStorage('disable_token_limit', %s);
                                window.saveToLocalStorage('output_token_limit', %d);
                                window.saveToLocalStorage('contextual', %s);
                                window.saveToLocalStorage('history_limit', %d);
                                window.saveToLocalStorage('rolling_history', %s);
                                window.saveToLocalStorage('batch_translation', %s);
                                window.saveToLocalStorage('batch_size', %d);
                                console.log('Settings saved to localStorage');
                            })();
                            </script>
                            """ % (
                                thread_delay_val, api_delay_val, chapter_range_val, token_limit_val,
                                str(disable_token_limit_val).lower(), output_token_limit_val,
                                str(contextual_val).lower(), history_limit_val,
                                str(rolling_history_val).lower(), str(batch_translation_val).lower(),
                                batch_size_val
                            )
                            
                            return "✅ Settings saved successfully", js_code
                        except Exception as e:
                            return f"❌ Failed to save: {str(e)}", ""
                    
                    # Add change handlers to all settings components
                    settings_components = [
                        thread_delay, api_delay, chapter_range, token_limit, 
                        disable_token_limit, output_token_limit, contextual, 
                        history_limit, rolling_history, batch_translation, 
                        batch_size, save_api_key
                    ]
                    
                    for component in settings_components:
                        component.change(
                            fn=save_settings_tab,
                            inputs=settings_components,
                            outputs=[save_status, js_executor]
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
            
            # Create a comprehensive load function that refreshes ALL values
            def load_all_settings():
                """Load all settings from config file on page refresh"""
                # Reload config to get latest values
                self.config = self.load_config()
                self.decrypted_config = decrypt_config(self.config.copy()) if API_KEY_ENCRYPTION_AVAILABLE else self.config.copy()
                
                # Return values for all tracked components
                return [
                    self.get_config_value('model', 'gpt-4-turbo'),  # epub_model
                    self.get_config_value('api_key', ''),  # epub_api_key
                    self.get_config_value('active_profile', list(self.profiles.keys())[0] if self.profiles else ''),  # epub_profile
                    self.profiles.get(self.get_config_value('active_profile', ''), ''),  # epub_system_prompt
                    self.get_config_value('temperature', 0.3),  # epub_temperature
                    self.get_config_value('model', 'gpt-4-turbo'),  # manga_model
                    self.get_config_value('api_key', ''),  # manga_api_key
                    self.get_config_value('active_profile', list(self.profiles.keys())[0] if self.profiles else ''),  # manga_profile
                    self.profiles.get(self.get_config_value('active_profile', ''), ''),  # manga_system_prompt
                    self.get_config_value('ocr_provider', 'custom-api'),  # ocr_provider
                    self.get_config_value('azure_vision_key', ''),  # azure_key
                    self.get_config_value('azure_vision_endpoint', ''),  # azure_endpoint
                    self.get_config_value('bubble_detection_enabled', True),  # bubble_detection
                    self.get_config_value('inpainting_enabled', True),  # inpainting
                    self.get_config_value('manga_font_size_mode', 'auto'),  # font_size_mode
                    self.get_config_value('manga_font_size', 24),  # font_size
                    self.get_config_value('manga_font_multiplier', 1.0),  # font_multiplier
                    self.get_config_value('manga_min_font_size', 12),  # min_font_size
                    self.get_config_value('manga_max_font_size', 48),  # max_font_size
                    self.get_config_value('manga_text_color', '#000000'),  # text_color_rgb
                    self.get_config_value('manga_shadow_enabled', True),  # shadow_enabled
                    self.get_config_value('manga_shadow_color', '#FFFFFF'),  # shadow_color
                    self.get_config_value('manga_shadow_offset_x', 2),  # shadow_offset_x
                    self.get_config_value('manga_shadow_offset_y', 2),  # shadow_offset_y
                    self.get_config_value('manga_shadow_blur', 0),  # shadow_blur
                    self.get_config_value('manga_bg_opacity', 130),  # bg_opacity
                    self.get_config_value('manga_bg_style', 'circle'),  # bg_style
                    self.get_config_value('manga_settings', {}).get('advanced', {}).get('parallel_panel_translation', False),  # parallel_panel_translation
                    self.get_config_value('manga_settings', {}).get('advanced', {}).get('panel_max_workers', 7),  # panel_max_workers
                ]
            
            # Add load handler to restore settings on page load
            app.load(
                fn=load_all_settings,
                inputs=[],
                outputs=[
                    epub_model, epub_api_key, epub_profile, epub_system_prompt, epub_temperature,
                    manga_model, manga_api_key, manga_profile, manga_system_prompt,
                    ocr_provider, azure_key, azure_endpoint, bubble_detection, inpainting,
                    font_size_mode, font_size, font_multiplier, min_font_size, max_font_size,
                    text_color_rgb, shadow_enabled, shadow_color, shadow_offset_x, shadow_offset_y,
                    shadow_blur, bg_opacity, bg_style, parallel_panel_translation, panel_max_workers
                ]
            )
        
        return app


def main():
    """Launch Gradio web app"""
    print("🚀 Starting Glossarion Web Interface...")
    
    # Check if running on Hugging Face Spaces
    is_spaces = os.getenv('SPACE_ID') is not None or os.getenv('HF_SPACES') == 'true'
    if is_spaces:
        print("🤗 Running on Hugging Face Spaces")
        print(f"📁 Space ID: {os.getenv('SPACE_ID', 'Unknown')}")
        print(f"📁 Files in current directory: {len(os.listdir('.'))} items")
        print(f"📁 Working directory: {os.getcwd()}")
        print(f"😎 Available manga modules: {MANGA_TRANSLATION_AVAILABLE}")
    else:
        print("🏠 Running locally")
    
    web_app = GlossarionWeb()
    app = web_app.create_interface()
    
    # Set favicon with absolute path if available (skip for Spaces)
    favicon_path = None
    if not is_spaces and os.path.exists("Halgakos.ico"):
        favicon_path = os.path.abspath("Halgakos.ico")
        print(f"✅ Using favicon: {favicon_path}")
    elif not is_spaces:
        print("⚠️ Halgakos.ico not found")
    
    # Launch with options appropriate for environment
    launch_args = {
        "server_name": "0.0.0.0",  # Allow external access
        "server_port": 7860,
        "share": False,
        "show_error": True,
    }
    
    # Only add favicon for non-Spaces environments
    if not is_spaces and favicon_path:
        launch_args["favicon_path"] = favicon_path
    
    app.launch(**launch_args)


if __name__ == "__main__":
    main()