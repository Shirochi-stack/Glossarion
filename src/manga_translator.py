# manga_translator.py
"""
Enhanced Manga Translation Pipeline with improved text visibility controls
Handles OCR, translation, and advanced text rendering for manga panels
Now with proper history management and full page context support
"""

import os
import json
import base64
import logging
import time
import traceback
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Google Cloud Vision imports
try:
    from google.cloud import vision
    GOOGLE_CLOUD_VISION_AVAILABLE = True
except ImportError:
    GOOGLE_CLOUD_VISION_AVAILABLE = False
    print("Warning: Google Cloud Vision not installed. Install with: pip install google-cloud-vision")

# Import HistoryManager for proper context management
try:
    from history_manager import HistoryManager
except ImportError:
    HistoryManager = None
    print("Warning: HistoryManager not available. Context tracking will be limited.")

logger = logging.getLogger(__name__)

@dataclass
class TextRegion:
    """Represents a detected text region (speech bubble, narration box, etc.)"""
    text: str
    vertices: List[Tuple[int, int]]  # Polygon vertices from Cloud Vision
    bounding_box: Tuple[int, int, int, int]  # x, y, width, height
    confidence: float
    region_type: str  # 'text_block' from Cloud Vision
    translated_text: Optional[str] = None
    
    def to_dict(self):
        return {
            'text': self.text,
            'vertices': self.vertices,
            'bounding_box': self.bounding_box,
            'confidence': self.confidence,
            'region_type': self.region_type,
            'translated_text': self.translated_text
        }

class MangaTranslator:
    """Main class for manga translation pipeline using Google Cloud Vision + API Key"""
    
    def __init__(self, google_credentials_path: str, unified_client, main_gui, log_callback=None):
        """Initialize with Google Cloud Vision credentials and API client from main GUI"""
        
        if not GOOGLE_CLOUD_VISION_AVAILABLE:
            raise ImportError("Google Cloud Vision required. Install with: pip install google-cloud-vision")
        
        # Set up Google Cloud Vision
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = google_credentials_path
        self.vision_client = vision.ImageAnnotatorClient()
        
        # API client from main GUI
        self.client = unified_client
        self.main_gui = main_gui
        self.log_callback = log_callback
        
        # Get all settings from GUI
        self.api_delay = float(self.main_gui.delay_entry.get() if hasattr(main_gui, 'delay_entry') else 2.0)
        self.temperature = float(main_gui.trans_temp.get() if hasattr(main_gui, 'trans_temp') else 0.3)
        self.max_tokens = int(main_gui.max_output_tokens if hasattr(main_gui, 'max_output_tokens') else 4000)
        if hasattr(main_gui, 'token_limit_disabled') and main_gui.token_limit_disabled:
            self.input_token_limit = None  # None means no limit
            self._log("📊 Input token limit: DISABLED (unlimited)")
        else:
            token_limit_value = main_gui.token_limit_entry.get() if hasattr(main_gui, 'token_limit_entry') else '120000'
            if token_limit_value and token_limit_value.strip().isdigit():
                self.input_token_limit = int(token_limit_value.strip())
            else:
                self.input_token_limit = 120000  # Default
            self._log(f"📊 Input token limit: {self.input_token_limit} tokens")
        
        # Get contextual settings from GUI
        self.contextual_enabled = main_gui.contextual_var.get() if hasattr(main_gui, 'contextual_var') else False
        self.translation_history_limit = int(main_gui.trans_history.get() if hasattr(main_gui, 'trans_history') else 3)
        self.rolling_history_enabled = main_gui.translation_history_rolling_var.get() if hasattr(main_gui, 'translation_history_rolling_var') else False
        
        # Initialize HistoryManager for proper context tracking
        self.history_manager = None
        if HistoryManager:
            # Create a manga-specific directory for history
            manga_history_dir = os.path.join(os.getcwd(), "manga_translation_history")
            os.makedirs(manga_history_dir, exist_ok=True)
            self.history_manager = HistoryManager(manga_history_dir)
            self._log(f"📚 Initialized HistoryManager with directory: {manga_history_dir}")
        
        # Full page context translation settings
        self.full_page_context_enabled = True
        
        # Default prompt for full page context mode
        self.full_page_context_prompt = (
            "You will receive multiple text segments from a manga page. "
            "Translate each segment considering the context of all segments together. "
            "Maintain consistency in character names, tone, and style across all translations.\n\n"
            "IMPORTANT: Return your response as a JSON object where each key is the original text "
            "and each value is the translation. Example:\n"
            '{\n'
            '  "こんにちは": "Hello",\n'
            '  "ありがとう": "Thank you"\n'
            '}'
        )
        
        # Store context for contextual translation (backwards compatibility)
        self.translation_context = []
        
        # Font settings for text rendering
        self.font_path = self._find_font()
        self.min_font_size = 8
        self.max_font_size = 36
        
        # Enhanced text rendering settings - Load from config if available
        config = main_gui.config if hasattr(main_gui, 'config') else {}
        
        self.text_bg_opacity = config.get('manga_bg_opacity', 255)  # 0-255, default fully opaque
        self.text_bg_style = config.get('manga_bg_style', 'box')  # 'box', 'circle', 'wrap'
        self.text_bg_reduction = config.get('manga_bg_reduction', 1.0)  # Size reduction factor (0.5-1.0)
        
        # Text color from config
        manga_text_color = config.get('manga_text_color', [0, 0, 0])
        self.text_color = tuple(manga_text_color)  # Convert list to tuple
        
        self.outline_color = (255, 255, 255)  # White outline
        self.outline_width_factor = 15  # Divider for font_size to get outline width
        self.selected_font_style = config.get('manga_font_path', None)  # Will store selected font path
        self.custom_font_size = config.get('manga_font_size', None) if config.get('manga_font_size', 0) > 0 else None
        
        # Text shadow settings from config
        self.shadow_enabled = config.get('manga_shadow_enabled', False)
        manga_shadow_color = config.get('manga_shadow_color', [128, 128, 128])
        self.shadow_color = tuple(manga_shadow_color)  # Convert list to tuple
        self.shadow_offset_x = config.get('manga_shadow_offset_x', 2)
        self.shadow_offset_y = config.get('manga_shadow_offset_y', 2)
        self.shadow_blur = config.get('manga_shadow_blur', 0)  # 0 = sharp shadow, higher = more blur
        self.skip_inpainting = config.get('manga_skip_inpainting', True)

        # Font size multiplier mode - Load from config
        self.font_size_mode = config.get('manga_font_size_mode', 'fixed')  # 'fixed' or 'multiplier'
        self.font_size_multiplier = config.get('manga_font_size_multiplier', 1.0)  # Default multiplierr        
        
        # Stop flag for interruption
        self.stop_flag = None
        
        self._log("\n🔧 MangaTranslator initialized with settings:")
        self._log(f"   API Delay: {self.api_delay}s")
        self._log(f"   Temperature: {self.temperature}")
        self._log(f"   Max Output Tokens: {self.max_tokens}")
        self._log(f"   Input Token Limit: {'DISABLED' if self.input_token_limit is None else self.input_token_limit}")
        self._log(f"   Contextual Translation: {'ENABLED' if self.contextual_enabled else 'DISABLED'}")
        self._log(f"   Translation History Limit: {self.translation_history_limit}")
        self._log(f"   Rolling History: {'ENABLED' if self.rolling_history_enabled else 'DISABLED'}")
        self._log(f"   Font Path: {self.font_path or 'Default'}")
        self._log(f"   Text Rendering: BG {self.text_bg_style}, Opacity {int(self.text_bg_opacity/255*100)}%")
        self._log(f"   Shadow: {'ENABLED' if self.shadow_enabled else 'DISABLED'}\n")
 
    def set_stop_flag(self, stop_flag):
        """Set the stop flag for checking interruptions"""
        self.stop_flag = stop_flag

    def _check_stop(self):
        """Check if stop has been requested"""
        if self.stop_flag and self.stop_flag.is_set():
            return True
        return False
    
    def set_full_page_context(self, enabled: bool, custom_prompt: str = None):
        """Configure full page context translation mode
        
        Args:
            enabled: Whether to translate all text regions in a single contextual request
            custom_prompt: Optional custom prompt for full page context mode
        """
        self.full_page_context_enabled = enabled
        if custom_prompt:
            self.full_page_context_prompt = custom_prompt
        
        self._log(f"📄 Full page context mode: {'ENABLED' if enabled else 'DISABLED'}")
        if enabled:
            self._log("   All text regions will be sent together for contextual translation")
        else:
            self._log("   Text regions will be translated individually")
    
    def update_text_rendering_settings(self, 
                                     bg_opacity: int = None,
                                     bg_style: str = None,
                                     bg_reduction: float = None,
                                     font_style: str = None,
                                     font_size: int = None,
                                     text_color: tuple = None,
                                     shadow_enabled: bool = None,
                                     shadow_color: tuple = None,
                                     shadow_offset_x: int = None,
                                     shadow_offset_y: int = None,
                                     shadow_blur: int = None):
        """Update text rendering settings"""
        self._log("📐 Updating text rendering settings:", "info")
        
        if bg_opacity is not None:
            self.text_bg_opacity = max(0, min(255, bg_opacity))
            self._log(f"  Background opacity: {int(self.text_bg_opacity/255*100)}%", "info")
        if bg_style is not None and bg_style in ['box', 'circle', 'wrap']:
            self.text_bg_style = bg_style
            self._log(f"  Background style: {bg_style}", "info")
        if bg_reduction is not None:
            self.text_bg_reduction = max(0.5, min(2.0, bg_reduction))
            self._log(f"  Background size: {int(self.text_bg_reduction*100)}%", "info")
        if font_style is not None:
            self.selected_font_style = font_style
            font_name = os.path.basename(font_style) if font_style else 'Default'
            self._log(f"  Font: {font_name}", "info")
        if font_size is not None:
            if font_size < 0:
                # Negative value indicates multiplier mode
                self.font_size_mode = 'multiplier'
                self.font_size_multiplier = abs(font_size)
                self.custom_font_size = None  # Clear fixed size
                self._log(f"  Font size mode: Dynamic multiplier ({self.font_size_multiplier:.1f}x)", "info")
            else:
                # Positive value or 0 indicates fixed mode
                self.font_size_mode = 'fixed'
                self.custom_font_size = font_size if font_size > 0 else None
                self._log(f"  Font size mode: Fixed ({font_size if font_size > 0 else 'Auto'})", "info")
            self.text_color = text_color
            self._log(f"  Text color: RGB{text_color}", "info")
        if shadow_enabled is not None:
            self.shadow_enabled = shadow_enabled
            self._log(f"  Shadow: {'Enabled' if shadow_enabled else 'Disabled'}", "info")
        if shadow_color is not None:
            self.shadow_color = shadow_color
            self._log(f"  Shadow color: RGB{shadow_color}", "info")
        if shadow_offset_x is not None:
            self.shadow_offset_x = shadow_offset_x
        if shadow_offset_y is not None:
            self.shadow_offset_y = shadow_offset_y
        if shadow_blur is not None:
            self.shadow_blur = max(0, shadow_blur)
            
        self._log("✅ Rendering settings updated", "info")
    
    def _log(self, message: str, level: str = "info"):
        """Log message to GUI or console"""
        if self.log_callback:
            self.log_callback(message, level)
        else:
            print(message)
    
    def detect_text_regions(self, image_path: str) -> List[TextRegion]:
        """Detect text regions using Google Cloud Vision API"""
        self._log(f"🔍 Detecting text regions in: {os.path.basename(image_path)}")
        
        try:
            # Read image file
            with open(image_path, 'rb') as image_file:
                content = image_file.read()
            
            # Create Vision API image object
            image = vision.Image(content=content)
            
            # Perform text detection with document text detection (better for dense text)
            response = self.vision_client.document_text_detection(image=image)
            
            if response.error.message:
                raise Exception(f"Cloud Vision API error: {response.error.message}")
            
            regions = []
            
            # Process each page (usually just one for manga)
            for page in response.full_text_annotation.pages:
                for block in page.blocks:
                    # Extract vertices
                    vertices = [(v.x, v.y) for v in block.bounding_box.vertices]
                    
                    # Calculate bounding box
                    xs = [v[0] for v in vertices]
                    ys = [v[1] for v in vertices]
                    x_min, x_max = min(xs), max(xs)
                    y_min, y_max = min(ys), max(ys)
                    
                    # Extract text from block
                    block_text = ""
                    for paragraph in block.paragraphs:
                        for word in paragraph.words:
                            word_text = ''.join([symbol.text for symbol in word.symbols])
                            block_text += word_text + " "
                    
                    block_text = block_text.strip()
                    
                    if block_text:  # Only add non-empty regions
                        region = TextRegion(
                            text=block_text,
                            vertices=vertices,
                            bounding_box=(x_min, y_min, x_max - x_min, y_max - y_min),
                            confidence=block.confidence,
                            region_type='text_block'
                        )
                        regions.append(region)
                        self._log(f"   Found text region: {block_text[:50]}...")
            
            # Merge nearby regions that might be part of the same speech bubble
            regions = self._merge_nearby_regions(regions)
            
            self._log(f"✅ Detected {len(regions)} text regions")
            return regions
            
        except Exception as e:
            self._log(f"❌ Error detecting text: {str(e)}", "error")
            raise
    
    def _get_translation_history_context(self) -> List[Dict[str, str]]:
        """Get translation history context from HistoryManager"""
        if not self.history_manager or not self.contextual_enabled:
            return []
        
        try:
            # Load full history
            full_history = self.history_manager.load_history()
            
            if not full_history:
                return []
            
            # Extract only the contextual messages up to the limit
            context = []
            exchange_count = 0
            
            # Process history in pairs (user + assistant messages)
            for i in range(0, len(full_history), 2):
                if i + 1 < len(full_history):
                    user_msg = full_history[i]
                    assistant_msg = full_history[i + 1]
                    
                    if user_msg.get("role") == "user" and assistant_msg.get("role") == "assistant":
                        context.extend([user_msg, assistant_msg])
                        exchange_count += 1
                        
                        # Only keep up to the history limit
                        if exchange_count >= self.translation_history_limit:
                            # Get only the most recent exchanges
                            context = context[-(self.translation_history_limit * 2):]
                            break
            
            return context
            
        except Exception as e:
            self._log(f"⚠️ Error loading history context: {str(e)}", "warning")
            return []
    
    def translate_text(self, text: str, context: Optional[List[Dict]] = None, image_path: str = None, region: TextRegion = None) -> str:
        """Translate text using API with GUI system prompt and full image context"""
        try:
            self._log(f"\n🌐 Starting translation for text: '{text[:50]}...'")
            # CHECK 1: Before starting
            if self._check_stop():
                self._log("⏹️ Translation stopped before full page context processing", "warning")
                return {}
            
            # Get system prompt from GUI profile
            profile_name = self.main_gui.profile_var.get()
            
            # The main GUI stores prompts directly as attributes, not in a dictionary
            if profile_name == "Manga_JP":
                system_prompt = getattr(self.main_gui, 'Manga_JP', '')
            elif profile_name == "Manga_KR":
                system_prompt = getattr(self.main_gui, 'Manga_KR', '')
            elif profile_name == "Manga_CN":
                system_prompt = getattr(self.main_gui, 'Manga_CN', '')
            else:
                # For other profiles, try to get from PROFILES dict
                system_prompt = self.main_gui.PROFILES.get(profile_name, {}).get('system_prompt', '')
            
            self._log(f"📋 Using profile: {profile_name}")
            if system_prompt:
                self._log(f"📝 System prompt: {system_prompt[:100]}...")
                messages = [{"role": "system", "content": system_prompt}]
            else:
                self._log(f"📝 No system prompt configured")
                messages = []
            
            # Add contextual translations if enabled
            if self.contextual_enabled and self.history_manager:
                # Get history from HistoryManager
                history_context = self._get_translation_history_context()
                
                if history_context:
                    context_count = len(history_context) // 2  # Each exchange is 2 messages
                    self._log(f"🔗 Adding {context_count} previous exchanges from history (limit: {self.translation_history_limit})")
                    messages.extend(history_context)
                else:
                    self._log(f"🔗 Contextual enabled but no history available yet")
            else:
                self._log(f"🔗 Contextual: {'Disabled' if not self.contextual_enabled else 'No HistoryManager'}")
            
            # Add full image context if available
            if image_path:
                try:
                    import base64
                    from PIL import Image as PILImage
                    
                    self._log(f"📷 Adding full page visual context for translation")
                    
                    # Read and encode the full image
                    with open(image_path, 'rb') as img_file:
                        img_data = img_file.read()
                    
                    # Check image size
                    img_size_mb = len(img_data) / (1024 * 1024)
                    self._log(f"📊 Image size: {img_size_mb:.2f} MB")
                    
                    # Optionally resize if too large (Gemini has limits)
                    if img_size_mb > 10:  # If larger than 10MB
                        self._log(f"📉 Resizing large image for API limits...")
                        pil_image = PILImage.open(image_path)
                        
                        # Calculate new size (max 2048px on longest side)
                        max_size = 2048
                        ratio = min(max_size / pil_image.width, max_size / pil_image.height)
                        if ratio < 1:
                            new_size = (int(pil_image.width * ratio), int(pil_image.height * ratio))
                            pil_image = pil_image.resize(new_size, PILImage.Resampling.LANCZOS)
                            
                            # Re-encode
                            from io import BytesIO
                            buffered = BytesIO()
                            pil_image.save(buffered, format="PNG", optimize=True)
                            img_data = buffered.getvalue()
                            self._log(f"✅ Resized to {new_size[0]}x{new_size[1]}px ({len(img_data)/(1024*1024):.2f} MB)")
                    
                    # Encode to base64
                    img_base64 = base64.b64encode(img_data).decode('utf-8')
                    
                    # Build the message with image and text location info
                    location_description = ""
                    if region:
                        x, y, w, h = region.bounding_box
                        # Describe where on the page this text is located
                        page_width = PILImage.open(image_path).width
                        page_height = PILImage.open(image_path).height
                        
                        # Determine position
                        h_pos = "left" if x < page_width/3 else "center" if x < 2*page_width/3 else "right"
                        v_pos = "top" if y < page_height/3 else "middle" if y < 2*page_height/3 else "bottom"
                        
                        location_description = f"\n\nThe text to translate is located in the {v_pos}-{h_pos} area of the page, "
                        location_description += f"at coordinates ({x}, {y}) with size {w}x{h} pixels."
                    
                    # Add image and text to translate
                    messages.append({
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{img_base64}"
                                }
                            },
                            {
                                "type": "text",
                                "text": f"Looking at this full manga page, translate the following text: '{text}'{location_description}"
                            }
                        ]
                    })
                    
                    
                    self._log(f"✅ Added full page image as visual context")
                    
                except Exception as e:
                    self._log(f"⚠️ Failed to add image context: {str(e)}", "warning")
                    self._log(f"   Error type: {type(e).__name__}", "warning")
                    import traceback
                    self._log(traceback.format_exc(), "warning")
                    # Fall back to text-only translation
                    messages.append({"role": "user", "content": text})
            else:
                # Text-only translation
                messages.append({"role": "user", "content": text})
            
            # Check input token limit
            # For Gemini, images cost approximately 258 tokens per image (for Gemini 1.5)
            # Text tokens are roughly 1 token per 4 characters
            text_tokens = 0
            image_tokens = 0

            for msg in messages:
                if isinstance(msg.get("content"), str):
                    # Simple text message
                    text_tokens += len(msg["content"]) // 4
                elif isinstance(msg.get("content"), list):
                    # Message with mixed content (text + image)
                    for content_part in msg["content"]:
                        if content_part.get("type") == "text":
                            text_tokens += len(content_part.get("text", "")) // 4
                        elif content_part.get("type") == "image_url":
                            # Gemini charges a flat rate per image regardless of size
                            # For Gemini 1.5 Flash: 258 tokens per image
                            # For Gemini 1.5 Pro: 258 tokens per image
                            image_tokens += 258

            estimated_tokens = text_tokens + image_tokens

            # Check token limit only if it's enabled
            if self.input_token_limit is None:
                self._log(f"📊 Token estimate - Text: {text_tokens}, Images: {image_tokens} (Total: {estimated_tokens} / unlimited)")
            else:
                self._log(f"📊 Token estimate - Text: {text_tokens}, Images: {image_tokens} (Total: {estimated_tokens} / {self.input_token_limit})")
                
                if estimated_tokens > self.input_token_limit:
                    self._log(f"⚠️ Token limit exceeded, trimming context", "warning")
                    # Keep system prompt, image, and current text only
                    if image_path:
                        messages = [messages[0], messages[-1]]  
                    else:
                        messages = [messages[0], {"role": "user", "content": text}]
                    # Recalculate tokens after trimming
                    text_tokens = len(messages[0]["content"]) // 4
                    if isinstance(messages[-1].get("content"), str):
                        text_tokens += len(messages[-1]["content"]) // 4
                    else:
                        text_tokens += len(messages[-1]["content"][0]["text"]) // 4
                    estimated_tokens = text_tokens + image_tokens
                    self._log(f"📊 Trimmed token estimate: {estimated_tokens}")
            
            start_time = time.time()
            api_time = 0  # Initialize to avoid NameError
            
            try:
                response = self.client.send(
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                api_time = time.time() - start_time
                self._log(f"✅ API responded in {api_time:.2f} seconds")
                
            except Exception as api_error:
                api_time = time.time() - start_time
                error_str = str(api_error).lower()
                error_type = type(api_error).__name__
                
                # Check for specific error types
                if "429" in error_str or "rate limit" in error_str:
                    self._log(f"⚠️ RATE LIMIT ERROR (429) after {api_time:.2f}s", "error")
                    self._log(f"   The API rate limit has been exceeded", "error")
                    self._log(f"   Please wait before retrying or reduce request frequency", "error")
                    self._log(f"   Error details: {str(api_error)}", "error")
                    raise Exception(f"Rate limit exceeded (429): {str(api_error)}")
                    
                elif "401" in error_str or "unauthorized" in error_str:
                    self._log(f"❌ AUTHENTICATION ERROR (401) after {api_time:.2f}s", "error")
                    self._log(f"   Invalid API key or authentication failed", "error")
                    self._log(f"   Please check your API key in settings", "error")
                    self._log(f"   Error details: {str(api_error)}", "error")
                    raise Exception(f"Authentication failed (401): {str(api_error)}")
                    
                elif "403" in error_str or "forbidden" in error_str:
                    self._log(f"❌ FORBIDDEN ERROR (403) after {api_time:.2f}s", "error")
                    self._log(f"   Access denied - check API permissions", "error")
                    self._log(f"   Error details: {str(api_error)}", "error")
                    raise Exception(f"Access forbidden (403): {str(api_error)}")
                    
                elif "400" in error_str or "bad request" in error_str:
                    self._log(f"❌ BAD REQUEST ERROR (400) after {api_time:.2f}s", "error")
                    self._log(f"   Invalid request format or parameters", "error")
                    self._log(f"   Error details: {str(api_error)}", "error")
                    raise Exception(f"Bad request (400): {str(api_error)}")
                    
                elif "timeout" in error_str:
                    self._log(f"⏱️ TIMEOUT ERROR after {api_time:.2f}s", "error")
                    self._log(f"   API request timed out", "error")
                    self._log(f"   Consider increasing timeout or retry", "error")
                    self._log(f"   Error details: {str(api_error)}", "error")
                    raise Exception(f"Request timeout: {str(api_error)}")
                    
                else:
                    # Generic API error
                    self._log(f"❌ API ERROR ({error_type}) after {api_time:.2f}s", "error")
                    self._log(f"   Error details: {str(api_error)}", "error")
                    self._log(f"   Full traceback:", "error")
                    self._log(traceback.format_exc(), "error")
                    raise
            
            # Extract content from response
            if hasattr(response, 'content'):
                translated = response.content.strip()
            else:
                # If response is a string or other format
                translated = str(response).strip()
            
            self._log(f"🔍 Raw response type: {type(response)}")
            self._log(f"🔍 Raw response content: '{translated[:100]}...'")
            
            # Check if the response looks like a Python literal (tuple/string representation)
            if translated.startswith("('") or translated.startswith('("') or translated.startswith("('''"):
                self._log(f"⚠️ Detected Python literal in response, attempting to extract actual text", "warning")
                original = translated
                try:
                    # Try to evaluate it as a Python literal
                    import ast
                    evaluated = ast.literal_eval(translated)
                    self._log(f"📦 Evaluated type: {type(evaluated)}")
                    
                    if isinstance(evaluated, tuple):
                        # Take the first element of the tuple
                        translated = str(evaluated[0])
                        self._log(f"📦 Extracted from tuple: '{translated[:50]}...'")
                    elif isinstance(evaluated, str):
                        translated = evaluated
                        self._log(f"📦 Extracted string: '{translated[:50]}...'")
                    else:
                        self._log(f"⚠️ Unexpected type after eval: {type(evaluated)}", "warning")
                        
                except Exception as e:
                    self._log(f"⚠️ Failed to parse Python literal: {e}", "warning")
                    self._log(f"⚠️ Original content: {original[:200]}", "warning")
                    
                    # Try multiple levels of unescaping
                    temp = translated
                    for i in range(5):  # Try up to 5 levels of unescaping
                        if temp.startswith("('") or temp.startswith('("'):
                            # Try regex as fallback
                            import re
                            match = re.search(r"^\(['\"](.+)['\"]\)$", temp, re.DOTALL)
                            if match:
                                temp = match.group(1)
                                self._log(f"📦 Regex extracted (level {i+1}): '{temp[:50]}...'")
                            else:
                                break
                        else:
                            break
                    translated = temp
            
            # Additional check for escaped content
            if '\\\\' in translated or '\\n' in translated:
                self._log(f"⚠️ Detected escaped content, unescaping...", "warning")
                try:
                    # Unescape the string
                    before = translated
                    translated = translated.encode().decode('unicode_escape')
                    self._log(f"📦 Unescaped: '{before[:50]}...' -> '{translated[:50]}...'")
                except Exception as e:
                    self._log(f"⚠️ Failed to unescape: {e}", "warning")
            
            self._log(f"🎯 Final translation result: '{translated[:50]}...'")
            
            # Apply glossary if available
            if hasattr(self.main_gui, 'manual_glossary') and self.main_gui.manual_glossary:
                glossary_count = len(self.main_gui.manual_glossary)
                self._log(f"📚 Applying glossary with {glossary_count} entries")
                
                replacements = 0
                for entry in self.main_gui.manual_glossary:
                    if 'source' in entry and 'target' in entry:
                        if entry['source'] in translated:
                            translated = translated.replace(entry['source'], entry['target'])
                            replacements += 1
                
                if replacements > 0:
                    self._log(f"   ✏️ Made {replacements} glossary replacements")
            
            # Store in history if HistoryManager is available
            if self.history_manager and self.contextual_enabled:
                try:
                    # Append to history with proper limit handling
                    self.history_manager.append_to_history(
                        user_content=text,
                        assistant_content=translated,
                        hist_limit=self.translation_history_limit,
                        reset_on_limit=not self.rolling_history_enabled,
                        rolling_window=self.rolling_history_enabled
                    )
                    
                    # Check if we're about to hit the limit
                    if self.history_manager.will_reset_on_next_append(
                        self.translation_history_limit, 
                        self.rolling_history_enabled
                    ):
                        mode = "roll over" if self.rolling_history_enabled else "reset"
                        self._log(f"📚 History will {mode} on next translation (at limit: {self.translation_history_limit})")
                    
                except Exception as e:
                    self._log(f"⚠️ Failed to save to history: {str(e)}", "warning")
            
            # Also store in legacy context for compatibility
            self.translation_context.append({
                "original": text,
                "translated": translated
            })
            
            return translated
            
        except Exception as e:
            self._log(f"❌ Translation error: {str(e)}", "error")
            self._log(f"   Error type: {type(e).__name__}", "error")
            import traceback
            self._log(f"   Traceback: {traceback.format_exc()}", "error")
            return text

    def translate_full_page_context(self, regions: List[TextRegion], image_path: str) -> Dict[str, str]:
        """Translate all text regions with full page context in a single request"""
        try:
            import time
            import traceback
            
            self._log(f"\n📄 Full page context translation of {len(regions)} text regions")
            
            # Get system prompt from GUI profile
            profile_name = self.main_gui.profile_var.get()
            
            # Try to get the prompt from prompt_profiles dictionary (for all profiles including custom ones)
            system_prompt = ''
            if hasattr(self.main_gui, 'prompt_profiles') and profile_name in self.main_gui.prompt_profiles:
                system_prompt = self.main_gui.prompt_profiles[profile_name]
                self._log(f"📋 Using profile: {profile_name}")
            else:
                # Fallback to check if it's stored as a direct attribute (legacy support)
                system_prompt = getattr(self.main_gui, profile_name.replace(' ', '_'), '')
                if system_prompt:
                    self._log(f"📋 Using profile (legacy): {profile_name}")
                else:
                    self._log(f"⚠️ Profile '{profile_name}' not found, using empty prompt", "warning")
            
            # Combine with full page context instructions
            if system_prompt:
                system_prompt = f"{system_prompt}\n\n{self.full_page_context_prompt}"
            else:
                system_prompt = self.full_page_context_prompt
            
            messages = [{"role": "system", "content": system_prompt}]
            
            # CHECK 2: Before adding context
            if self._check_stop():
                self._log("⏹️ Translation stopped during context preparation", "warning")
                return {}
            
            # Add contextual translations if enabled
            if self.contextual_enabled and self.history_manager:
                history_context = self._get_translation_history_context()
                if history_context:
                    context_count = len(history_context) // 2
                    self._log(f"🔗 Adding {context_count} previous exchanges from history")
                    messages.extend(history_context)
            
            # Prepare text segments with indices
            all_texts = {}
            text_list = []
            for i, region in enumerate(regions):
                # Use index-based key to handle duplicate texts
                key = f"[{i}] {region.text}"
                all_texts[key] = region.text
                text_list.append(f"{key}")
                
            # CHECK 3: Before image processing
            if self._check_stop():
                self._log("⏹️ Translation stopped before image processing", "warning")
                return {}
                    
            # Create the request with image
            try:
                import base64
                from PIL import Image as PILImage
                
                self._log(f"📷 Adding full page visual context for translation")
                
                # Read and encode the image
                with open(image_path, 'rb') as img_file:
                    img_data = img_file.read()
                
                # Check image size
                img_size_mb = len(img_data) / (1024 * 1024)
                self._log(f"📊 Image size: {img_size_mb:.2f} MB")
                
                # Get image dimensions
                pil_image = PILImage.open(image_path)
                self._log(f"   Image dimensions: {pil_image.width}x{pil_image.height}")
 
                # CHECK 4: Before resizing (which can take time)
                if self._check_stop():
                    self._log("⏹️ Translation stopped during image preparation", "warning")
                    return {}
                
                # Resize if needed
                if img_size_mb > 10:
                    self._log(f"📉 Resizing large image for API limits...")
                    max_size = 2048
                    ratio = min(max_size / pil_image.width, max_size / pil_image.height)
                    if ratio < 1:
                        new_size = (int(pil_image.width * ratio), int(pil_image.height * ratio))
                        pil_image = pil_image.resize(new_size, PILImage.Resampling.LANCZOS)
                        from io import BytesIO
                        buffered = BytesIO()
                        pil_image.save(buffered, format="PNG", optimize=True)
                        img_data = buffered.getvalue()
                        self._log(f"✅ Resized to {new_size[0]}x{new_size[1]}px ({len(img_data)/(1024*1024):.2f} MB)")
                
                # Convert to base64
                img_b64 = base64.b64encode(img_data).decode('utf-8')
                
                # Create the full context message
                context_text = "\n".join(text_list)
                
                # Log text content info
                total_chars = sum(len(region.text) for region in regions)
                self._log(f"📝 Text content: {len(regions)} regions, {total_chars} total characters")
                
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": context_text},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
                    ]
                })
                
                self._log(f"✅ Added full page image as visual context")
                
            except Exception as e:
                self._log(f"⚠️ Failed to add image context: {str(e)}", "warning")
                self._log(f"   Error type: {type(e).__name__}", "warning")
                import traceback
                self._log(traceback.format_exc(), "warning")
                # Fall back to text-only translation
                messages.append({"role": "user", "content": context_text})

            # CHECK 5: Before API call
            if self._check_stop():
                self._log("⏹️ Translation stopped before API call", "warning")
                return {}
            
            # Check input token limit
            # For Gemini, images cost approximately 258 tokens per image (for Gemini 1.5)
            # Text tokens are roughly 1 token per 4 characters
            text_tokens = 0
            image_tokens = 0

            for msg in messages:
                if isinstance(msg.get("content"), str):
                    # Simple text message
                    text_tokens += len(msg["content"]) // 4
                elif isinstance(msg.get("content"), list):
                    # Message with mixed content (text + image)
                    for content_part in msg["content"]:
                        if content_part.get("type") == "text":
                            text_tokens += len(content_part.get("text", "")) // 4
                        elif content_part.get("type") == "image_url":
                            # Gemini charges a flat rate per image regardless of size
                            # For Gemini 1.5 Flash: 258 tokens per image
                            # For Gemini 1.5 Pro: 258 tokens per image
                            image_tokens += 258

            estimated_tokens = text_tokens + image_tokens

            # Check token limit only if it's enabled
            if self.input_token_limit is None:
                self._log(f"📊 Token estimate - Text: {text_tokens}, Images: {image_tokens} (Total: {estimated_tokens} / unlimited)")
            else:
                self._log(f"📊 Token estimate - Text: {text_tokens}, Images: {image_tokens} (Total: {estimated_tokens} / {self.input_token_limit})")
                
                if estimated_tokens > self.input_token_limit:
                    self._log(f"⚠️ Token limit exceeded, trimming context", "warning")
                    # Keep system prompt, image, and current text only
                    messages = [messages[0], messages[-1]]  
                    # Recalculate tokens
                    text_tokens = len(messages[0]["content"]) // 4 + len(context_text) // 4
                    estimated_tokens = text_tokens + image_tokens
                    self._log(f"📊 Trimmed token estimate: {estimated_tokens}")
            
            # Make API call using the client's send method (matching translate_text)
            self._log(f"🌐 Sending full page context to API...")
            self._log(f"   API Model: {self.client.model if hasattr(self.client, 'model') else 'unknown'}")
            self._log(f"   Temperature: {self.temperature}")
            self._log(f"   Max Output Tokens: {self.max_tokens}")
            
            start_time = time.time()
            api_time = 0  # Initialize to avoid NameError
            
            try:
                response = self.client.send(
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens  # Use the configured max tokens without multiplication
                )
                api_time = time.time() - start_time
                
                # CHECK 6: Immediately after API response
                if self._check_stop():
                    self._log(f"⏹️ Translation stopped after API call ({api_time:.2f}s)", "warning")
                    return {}
                
                self._log(f"✅ API responded in {api_time:.2f} seconds")
                
            except Exception as api_error:
                api_time = time.time() - start_time
                
                # CHECK 7: After API error
                if self._check_stop():
                    self._log(f"⏹️ Translation stopped during API error handling", "warning")
                    return {}
                
                error_str = str(api_error).lower()
                error_type = type(api_error).__name__
                
                # Check for specific error types
                if "429" in error_str or "rate limit" in error_str:
                    self._log(f"⚠️ RATE LIMIT ERROR (429) after {api_time:.2f}s", "error")
                    self._log(f"   The API rate limit has been exceeded", "error")
                    self._log(f"   Please wait before retrying or reduce request frequency", "error")
                    self._log(f"   Error details: {str(api_error)}", "error")
                    raise Exception(f"Rate limit exceeded (429): {str(api_error)}")
                    
                elif "401" in error_str or "unauthorized" in error_str:
                    self._log(f"❌ AUTHENTICATION ERROR (401) after {api_time:.2f}s", "error")
                    self._log(f"   Invalid API key or authentication failed", "error")
                    self._log(f"   Please check your API key in settings", "error")
                    self._log(f"   Error details: {str(api_error)}", "error")
                    raise Exception(f"Authentication failed (401): {str(api_error)}")
                    
                elif "403" in error_str or "forbidden" in error_str:
                    self._log(f"❌ FORBIDDEN ERROR (403) after {api_time:.2f}s", "error")
                    self._log(f"   Access denied - check API permissions", "error")
                    self._log(f"   Error details: {str(api_error)}", "error")
                    raise Exception(f"Access forbidden (403): {str(api_error)}")
                    
                elif "400" in error_str or "bad request" in error_str:
                    self._log(f"❌ BAD REQUEST ERROR (400) after {api_time:.2f}s", "error")
                    self._log(f"   Invalid request format or parameters", "error")
                    self._log(f"   Error details: {str(api_error)}", "error")
                    raise Exception(f"Bad request (400): {str(api_error)}")
                    
                elif "timeout" in error_str:
                    self._log(f"⏱️ TIMEOUT ERROR after {api_time:.2f}s", "error")
                    self._log(f"   API request timed out", "error")
                    self._log(f"   Consider increasing timeout or retry", "error")
                    self._log(f"   Error details: {str(api_error)}", "error")
                    raise Exception(f"Request timeout: {str(api_error)}")
                    
                else:
                    # Generic API error
                    self._log(f"❌ API ERROR ({error_type}) after {api_time:.2f}s", "error")
                    self._log(f"   Error details: {str(api_error)}", "error")
                    self._log(f"   Full traceback:", "error")
                    self._log(traceback.format_exc(), "error")
                    raise
            
            # Extract content from response (matching translate_text method)
            if hasattr(response, 'content'):
                response_text = response.content.strip()
            else:
                response_text = str(response).strip()
            
            self._log(f"📥 Received response ({len(response_text)} chars)")
            
            # CHECK 8: Before parsing response
            if self._check_stop():
                self._log("⏹️ Translation stopped before parsing response", "warning")
                return {}
            
            self._log(f"🔍 Raw response type: {type(response)}")
            self._log(f"🔍 Raw response preview: '{response_text[:100]}...'")
            
            # Check if the response looks like a Python literal (tuple/string representation)
            if response_text.startswith("('") or response_text.startswith('("') or response_text.startswith("('''"):
                self._log(f"⚠️ Detected Python literal in response, attempting to extract actual text", "warning")
                original_response = response_text
                try:
                    # Try to evaluate it as a Python literal
                    import ast
                    evaluated = ast.literal_eval(response_text)
                    self._log(f"📦 Evaluated type: {type(evaluated)}")
                    
                    if isinstance(evaluated, tuple):
                        # Take the first element of the tuple
                        response_text = str(evaluated[0])
                        self._log(f"📦 Extracted from tuple: '{response_text[:50]}...'")
                    elif isinstance(evaluated, str):
                        response_text = evaluated
                        self._log(f"📦 Extracted string: '{response_text[:50]}...'")
                    else:
                        self._log(f"⚠️ Unexpected type after eval: {type(evaluated)}", "warning")
                        
                except Exception as e:
                    self._log(f"⚠️ Failed to parse Python literal: {e}", "warning")
                    self._log(f"⚠️ Original content: {original_response[:200]}", "warning")
                    
                    # Try regex as fallback
                    import re
                    match = re.search(r"^\(['\"](.+)['\"]\)$", response_text, re.DOTALL)
                    if match:
                        response_text = match.group(1)
                        self._log(f"📦 Regex extracted: '{response_text[:50]}...'")
            
            # Additional check for escaped content
            if '\\\\' in response_text or '\\n' in response_text:
                self._log(f"⚠️ Detected escaped content, unescaping...", "warning")
                try:
                    # Unescape the string
                    before = response_text
                    response_text = response_text.encode().decode('unicode_escape')
                    self._log(f"📦 Unescaped content")
                except Exception as e:
                    self._log(f"⚠️ Failed to unescape: {e}", "warning")
            
            # Try to parse as JSON
            translations = {}
            try:
                # Clean up response if needed
                if "```json" in response_text:
                    import re
                    match = re.search(r'```json\s*(.*?)```', response_text, re.DOTALL)
                    if match:
                        response_text = match.group(1)
                elif "```" in response_text:
                    import re
                    match = re.search(r'```\s*(.*?)```', response_text, re.DOTALL)
                    if match:
                        response_text = match.group(1)
                
                # Parse JSON
                import json
                translations = json.loads(response_text)
                self._log(f"✅ Successfully parsed {len(translations)} translations")
                
            except json.JSONDecodeError as e:
                self._log(f"⚠️ Failed to parse JSON response: {str(e)}", "warning")
                self._log(f"Response preview: {response_text[:200]}...", "warning")
                
                # Fallback: try to extract translations manually
                return {}
            
            # Map translations back to regions
            result = {}
            for i, region in enumerate(regions):
                
                # CHECK 9: During mapping (in case there are many regions)
                if i % 10 == 0 and self._check_stop():  # Check every 10 regions
                    self._log(f"⏹️ Translation stopped during mapping (processed {i}/{len(regions)} regions)", "warning")
                    return result  #
                
                key = f"[{i}] {region.text}"
                
                # First try with the indexed key
                if key in translations:
                    translated = translations[key]
                    self._log(f"  ✅ Found translation with indexed key for region {i}")
                # Then try with just the text (without index)
                elif region.text in translations:
                    translated = translations[region.text]
                    self._log(f"  ✅ Found translation with text-only key for region {i}")
                else:
                    self._log(f"⚠️ No translation found for region {i}", "warning")
                    self._log(f"   Tried keys: '{key}' and '{region.text}'", "warning")
                    self._log(f"   Available keys sample: {list(translations.keys())[:2]}...", "warning")
                    translated = region.text  # Use original as fallback
                
                # Apply glossary
                if translated != region.text and hasattr(self.main_gui, 'manual_glossary') and self.main_gui.manual_glossary:
                    for entry in self.main_gui.manual_glossary:
                        if 'source' in entry and 'target' in entry:
                            if entry['source'] in translated:
                                translated = translated.replace(entry['source'], entry['target'])
                
                result[region.text] = translated
                
                if translated != region.text:
                    self._log(f"  ✅ Mapped translation: '{region.text[:30]}...' → '{translated[:30]}...'")
                
                # Store in history
                if self.history_manager and self.contextual_enabled and translated != region.text:
                    try:
                        self.history_manager.append_to_history(
                            user_content=region.text,
                            assistant_content=translated,
                            hist_limit=self.translation_history_limit,
                            reset_on_limit=not self.rolling_history_enabled,
                            rolling_window=self.rolling_history_enabled
                        )
                    except Exception as e:
                        self._log(f"⚠️ Failed to save to history: {str(e)}", "warning")
            
            return result
            
        except Exception as e:
            
            # CHECK 10: In exception handler
            if self._check_stop():
                self._log("⏹️ Translation stopped due to user request", "warning")
                return {}  
                
            self._log(f"❌ Full page context translation error: {str(e)}", "error")
            self._log(traceback.format_exc(), "error")
            return {}
            
    def create_text_mask(self, image: np.ndarray, regions: List[TextRegion]) -> np.ndarray:
        """Create a binary mask for text regions using exact vertices from Cloud Vision"""
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        for region in regions:
            # Use the exact polygon vertices from Cloud Vision
            pts = np.array(region.vertices, np.int32)
            pts = pts.reshape((-1, 1, 2))
            
            # Fill the polygon
            import cv2
            cv2.fillPoly(mask, [pts], 255)
            
            # Dilate slightly to ensure complete text coverage
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)
        
        return mask
    
    def inpaint_regions(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Enhanced inpainting with transparency support"""
        import cv2
        
        # If we want fully transparent backgrounds, we need to handle this differently
        if self.text_bg_opacity == 0:
            # For fully transparent, we'll use more sophisticated inpainting
            # that attempts to reconstruct the background
            result = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
            self._log("   Using TELEA inpainting for transparent background", "info")
        else:
            # For non-transparent, fill with white as before
            result = image.copy()
            result[mask > 0] = 255  # Set to white
            self._log("   Using white fill for non-transparent background", "info")
        
        return result
    
    def _regions_overlap(self, region1: TextRegion, region2: TextRegion) -> bool:
        """Check if two regions overlap"""
        x1, y1, w1, h1 = region1.bounding_box
        x2, y2, w2, h2 = region2.bounding_box
        
        # Check if rectangles overlap
        return not (x1 + w1 < x2 or x2 + w2 < x1 or y1 + h1 < y2 or y2 + h2 < y1)
    
    def render_translated_text(self, image: np.ndarray, regions: List[TextRegion]) -> np.ndarray:
            """Enhanced text rendering with customizable backgrounds and styles"""
            self._log(f"\n🎨 Starting ENHANCED text rendering with custom settings:", "info")
            self._log(f"  ✅ Using ENHANCED renderer (not the simple version)", "info")
            self._log(f"  Background: {self.text_bg_style} @ {int(self.text_bg_opacity/255*100)}% opacity", "info")
            self._log(f"  Text color: RGB{self.text_color}", "info")
            self._log(f"  Shadow: {'Enabled' if self.shadow_enabled else 'Disabled'}", "info")
            self._log(f"  Font: {os.path.basename(self.selected_font_style) if self.selected_font_style else 'Default'}", "info")
            
            # Convert to PIL for text rendering
            import cv2
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            # Check if any regions overlap
            has_overlaps = False
            for i, region1 in enumerate(regions):
                for region2 in regions[i+1:]:
                    if self._regions_overlap(region1, region2):
                        has_overlaps = True
                        break
                if has_overlaps:
                    break
            
            # Handle transparency settings based on overlaps
            if has_overlaps and self.text_bg_opacity < 255 and self.text_bg_opacity > 0:
                self._log("  ⚠️ Overlapping regions detected with partial transparency", "warning")
                self._log("  ℹ️ Rendering with requested transparency level", "info")
            
            region_count = 0
            
            # Decide rendering path based on transparency needs
            # For full transparency (opacity = 0) or no overlaps, use RGBA rendering
            # For overlaps with partial transparency, we still use RGBA to honor user settings
            use_rgba_rendering = True  # Always use RGBA for consistent transparency support
            
            if use_rgba_rendering:
                # Transparency-enabled rendering path
                pil_image = pil_image.convert('RGBA')
                
                for region in regions:
                    if not region.translated_text:
                        continue
                    
                    region_count += 1
                    self._log(f"  Rendering region {region_count}: {region.translated_text[:30]}...", "info")
                    
                    # Create a separate layer for this region only
                    region_overlay = Image.new('RGBA', pil_image.size, (0, 0, 0, 0))
                    region_draw = ImageDraw.Draw(region_overlay)
                    
                    x, y, w, h = region.bounding_box
                    
                    # Find optimal font size
                    if self.custom_font_size:
                        # Fixed size specified
                        font_size = self.custom_font_size
                        lines = self._wrap_text(region.translated_text, 
                                              self._get_font(font_size), 
                                              int(w * 0.8), region_draw)
                    elif self.font_size_mode == 'multiplier':
                        # Use dynamic sizing with multiplier
                        font_size, lines = self._fit_text_to_region(
                            region.translated_text, w, h, region_draw
                        )
                    else:
                        # Auto mode - use standard fitting
                        font_size, lines = self._fit_text_to_region(
                            region.translated_text, w, h, region_draw
                        )
                    
                    # Load font
                    font = self._get_font(font_size)
                    
                    # Calculate text layout
                    line_height = font_size * 1.2
                    total_height = len(lines) * line_height
                    start_y = y + (h - total_height) // 2
                    
                    # Draw background if opacity > 0
                    if self.text_bg_opacity > 0:
                        self._draw_text_background(region_draw, x, y, w, h, lines, font, 
                                                 font_size, start_y)
                    
                    # Draw text on the same region overlay
                    for i, line in enumerate(lines):
                        text_bbox = region_draw.textbbox((0, 0), line, font=font)
                        text_width = text_bbox[2] - text_bbox[0]
                        
                        text_x = x + (w - text_width) // 2
                        text_y = start_y + i * line_height
                        
                        if self.shadow_enabled:
                            self._draw_text_shadow(region_draw, text_x, text_y, line, font)
                        
                        outline_width = max(1, font_size // self.outline_width_factor)
                        
                        # Draw outline
                        for dx in range(-outline_width, outline_width + 1):
                            for dy in range(-outline_width, outline_width + 1):
                                if dx != 0 or dy != 0:
                                    region_draw.text((text_x + dx, text_y + dy), line, 
                                            font=font, fill=self.outline_color + (255,))
                        
                        # Draw main text
                        region_draw.text((text_x, text_y), line, font=font, fill=self.text_color + (255,))
                    
                    # Composite this region onto the main image
                    pil_image = Image.alpha_composite(pil_image, region_overlay)
                
                # Convert back to RGB
                pil_image = pil_image.convert('RGB')
            
            else:
                # This path is now deprecated but kept for backwards compatibility
                # Direct rendering without transparency layers
                draw = ImageDraw.Draw(pil_image)
                
                for region in regions:
                    if not region.translated_text:
                        continue
                    
                    region_count += 1
                    self._log(f"  Rendering region {region_count}: {region.translated_text[:30]}...", "info")
                    
                    x, y, w, h = region.bounding_box
                    
                    # Find optimal font size
                    if self.custom_font_size:
                        font_size = self.custom_font_size
                        lines = self._wrap_text(region.translated_text, 
                                              self._get_font(font_size), 
                                              int(w * 0.8), draw)
                    else:
                        font_size, lines = self._fit_text_to_region(
                            region.translated_text, w, h, draw
                        )
                    
                    # Load font
                    font = self._get_font(font_size)
                    
                    # Calculate text layout
                    line_height = font_size * 1.2
                    total_height = len(lines) * line_height
                    start_y = y + (h - total_height) // 2
                    
                    # Draw opaque background
                    if self.text_bg_opacity > 0:
                        self._draw_text_background(draw, x, y, w, h, lines, font, 
                                                 font_size, start_y)
                    
                    # Draw text
                    for i, line in enumerate(lines):
                        text_bbox = draw.textbbox((0, 0), line, font=font)
                        text_width = text_bbox[2] - text_bbox[0]
                        
                        text_x = x + (w - text_width) // 2
                        text_y = start_y + i * line_height
                        
                        if self.shadow_enabled:
                            self._draw_text_shadow(draw, text_x, text_y, line, font)
                        
                        outline_width = max(1, font_size // self.outline_width_factor)
                        
                        # Draw outline
                        for dx in range(-outline_width, outline_width + 1):
                            for dy in range(-outline_width, outline_width + 1):
                                if dx != 0 or dy != 0:
                                    draw.text((text_x + dx, text_y + dy), line, 
                                            font=font, fill=self.outline_color)
                        
                        # Draw main text
                        draw.text((text_x, text_y), line, font=font, fill=self.text_color)
            
            # Convert back to numpy array
            result = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            self._log(f"✅ ENHANCED text rendering complete - rendered {region_count} regions", "info")
            return result
    
    def _draw_text_background(self, draw: ImageDraw, x: int, y: int, w: int, h: int,
                            lines: List[str], font: ImageFont, font_size: int, 
                            start_y: int):
        """Draw background behind text with selected style"""
        # Early return if opacity is 0 (fully transparent)
        if self.text_bg_opacity == 0:
            return
        
        # Calculate actual text bounds
        line_height = font_size * 1.2
        max_width = 0
        
        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=font)
            line_width = bbox[2] - bbox[0]
            max_width = max(max_width, line_width)
        
        # Apply size reduction
        padding = int(font_size * 0.3)
        bg_width = int((max_width + padding * 2) * self.text_bg_reduction)
        bg_height = int((len(lines) * line_height + padding * 2) * self.text_bg_reduction)
        
        # Center background
        bg_x = x + (w - bg_width) // 2
        bg_y = int(start_y - padding)
        
        # Create semi-transparent color
        bg_color = (255, 255, 255, self.text_bg_opacity)
        
        if self.text_bg_style == 'box':
            # Rounded rectangle
            radius = min(20, bg_width // 10, bg_height // 10)
            self._draw_rounded_rectangle(draw, bg_x, bg_y, bg_x + bg_width, 
                                       bg_y + bg_height, radius, bg_color)
            
        elif self.text_bg_style == 'circle':
            # Ellipse that encompasses the text
            center_x = bg_x + bg_width // 2
            center_y = bg_y + bg_height // 2
            # Make it slightly wider to look more natural
            ellipse_width = int(bg_width * 1.2)
            ellipse_height = bg_height
            
            draw.ellipse([center_x - ellipse_width // 2, center_y - ellipse_height // 2,
                         center_x + ellipse_width // 2, center_y + ellipse_height // 2],
                        fill=bg_color)
            
        elif self.text_bg_style == 'wrap':
            # Individual background for each line
            for i, line in enumerate(lines):
                bbox = draw.textbbox((0, 0), line, font=font)
                line_width = bbox[2] - bbox[0]
                
                line_bg_width = int((line_width + padding) * self.text_bg_reduction)
                line_bg_x = x + (w - line_bg_width) // 2
                line_bg_y = int(start_y + i * line_height - padding // 2)
                line_bg_height = int(line_height + padding // 2)
                
                # Draw rounded rectangle for each line
                radius = min(10, line_bg_width // 10, line_bg_height // 10)
                self._draw_rounded_rectangle(draw, line_bg_x, line_bg_y, 
                                           line_bg_x + line_bg_width,
                                           line_bg_y + line_bg_height, radius, bg_color)
    
    def _draw_text_shadow(self, draw: ImageDraw, x: int, y: int, text: str, font: ImageFont):
        """Draw text shadow with optional blur effect"""
        if self.shadow_blur == 0:
            # Simple sharp shadow
            shadow_x = x + self.shadow_offset_x
            shadow_y = y + self.shadow_offset_y
            draw.text((shadow_x, shadow_y), text, font=font, fill=self.shadow_color)
        else:
            # Blurred shadow (simulated with multiple layers)
            blur_range = self.shadow_blur
            opacity_step = 80 // (blur_range + 1)  # Distribute opacity across blur layers
            
            for blur_offset in range(blur_range, 0, -1):
                layer_opacity = opacity_step * (blur_range - blur_offset + 1)
                shadow_color_with_opacity = self.shadow_color + (layer_opacity,)
                
                # Draw shadow at multiple positions for blur effect
                for dx in range(-blur_offset, blur_offset + 1):
                    for dy in range(-blur_offset, blur_offset + 1):
                        if dx*dx + dy*dy <= blur_offset*blur_offset:  # Circular blur
                            shadow_x = x + self.shadow_offset_x + dx
                            shadow_y = y + self.shadow_offset_y + dy
                            draw.text((shadow_x, shadow_y), text, font=font, 
                                    fill=shadow_color_with_opacity)
    
    def _draw_rounded_rectangle(self, draw: ImageDraw, x1: int, y1: int, 
                               x2: int, y2: int, radius: int, fill):
        """Draw a rounded rectangle"""
        # Draw the main rectangle
        draw.rectangle([x1 + radius, y1, x2 - radius, y2], fill=fill)
        draw.rectangle([x1, y1 + radius, x2, y2 - radius], fill=fill)
        
        # Draw the corners
        draw.pieslice([x1, y1, x1 + 2 * radius, y1 + 2 * radius], 180, 270, fill=fill)
        draw.pieslice([x2 - 2 * radius, y1, x2, y1 + 2 * radius], 270, 360, fill=fill)
        draw.pieslice([x1, y2 - 2 * radius, x1 + 2 * radius, y2], 90, 180, fill=fill)
        draw.pieslice([x2 - 2 * radius, y2 - 2 * radius, x2, y2], 0, 90, fill=fill)
    
    def _get_font(self, font_size: int) -> ImageFont:
        """Get font with specified size, using selected style if available"""
        font_path = self.selected_font_style or self.font_path
        
        if font_path:
            try:
                return ImageFont.truetype(font_path, font_size)
            except:
                pass
        
        return ImageFont.load_default()
    
    def _fit_text_to_region(self, text: str, max_width: int, max_height: int, draw: ImageDraw) -> Tuple[int, List[str]]:
            """Find optimal font size and text wrapping"""
            # Use 80% of the region for text (leave margins)
            usable_width = int(max_width * 0.8)
            usable_height = int(max_height * 0.8)
            
            # Standard font size range
            min_font_size = self.min_font_size
            max_font_size = self.max_font_size
            
            # First, find the best base font size (without multiplier)
            best_base_size = min_font_size
            best_lines = []
            
            for font_size in range(max_font_size, min_font_size - 1, -1):
                font = self._get_font(font_size)
                
                # Wrap text
                lines = self._wrap_text(text, font, usable_width, draw)
                
                # Check if it fits vertically
                line_height = font_size * 1.2
                total_height = len(lines) * line_height
                
                if total_height <= usable_height:
                    best_base_size = font_size
                    best_lines = lines
                    break
            
            # Now apply multiplier if in multiplier mode
            if self.font_size_mode == 'multiplier':
                final_size = int(best_base_size * self.font_size_multiplier)
                # Clamp to reasonable bounds
                final_size = max(self.min_font_size, min(final_size, self.max_font_size * 3))
                
                # Re-calculate lines with the multiplied size
                font = self._get_font(final_size)
                best_lines = self._wrap_text(text, font, usable_width, draw)
                
                # Check if we need to truncate with new size
                line_height = final_size * 1.2
                max_lines = int(usable_height // line_height)
                if len(best_lines) > max_lines and max_lines > 0:
                    best_lines = best_lines[:max_lines-1] + [best_lines[max_lines-1][:10] + '...']
            else:
                final_size = best_base_size
                
            return final_size, best_lines
    
    def _wrap_text(self, text: str, font: ImageFont, max_width: int, draw: ImageDraw) -> List[str]:
        """Wrap text to fit within max_width"""
        words = text.split()
        lines = []
        current_line = []
        
        for word in words:
            test_line = ' '.join(current_line + [word])
            text_bbox = draw.textbbox((0, 0), test_line, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            
            if text_width <= max_width:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                    current_line = [word]
                else:
                    # Word is too long, split it
                    lines.append(word)
                    current_line = []
        
        if current_line:
            lines.append(' '.join(current_line))
        
        return lines
    
    def _merge_nearby_regions(self, regions: List[TextRegion]) -> List[TextRegion]:
        """Merge text regions that are likely part of the same speech bubble"""
        if len(regions) <= 1:
            return regions
        
        merged = []
        used = set()
        
        for i, region1 in enumerate(regions):
            if i in used:
                continue
            
            # Start with this region
            merged_text = region1.text
            merged_vertices = list(region1.vertices)
            
            # Check for nearby regions
            for j, region2 in enumerate(regions[i+1:], i+1):
                if j in used:
                    continue
                
                if self._regions_are_nearby(region1, region2):
                    # Merge the regions
                    merged_text += " " + region2.text
                    merged_vertices.extend(region2.vertices)
                    used.add(j)
            
            # Calculate new bounding box
            xs = [v[0] for v in merged_vertices]
            ys = [v[1] for v in merged_vertices]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            merged_bbox = (min_x, min_y, max_x - min_x, max_y - min_y)
            
            merged_region = TextRegion(
                text=merged_text,
                vertices=merged_vertices,
                bounding_box=merged_bbox,
                confidence=region1.confidence,
                region_type='merged_text_block'
            )
            
            merged.append(merged_region)
            used.add(i)
        
        return merged
    
    def _regions_are_nearby(self, region1: TextRegion, region2: TextRegion, threshold: int = 50) -> bool:
        """Check if two regions are close enough to be in the same bubble"""
        x1, y1, w1, h1 = region1.bounding_box
        x2, y2, w2, h2 = region2.bounding_box
        
        # Check horizontal distance between closest edges
        horizontal_gap = 0
        if x1 + w1 < x2:  # region1 is to the left
            horizontal_gap = x2 - (x1 + w1)
        elif x2 + w2 < x1:  # region2 is to the left
            horizontal_gap = x1 - (x2 + w2)
        
        # Check vertical distance between closest edges
        vertical_gap = 0
        if y1 + h1 < y2:  # region1 is above
            vertical_gap = y2 - (y1 + h1)
        elif y2 + h2 < y1:  # region2 is above
            vertical_gap = y1 - (y2 + h2)
        
        # Regions are nearby if they're close horizontally OR vertically
        # This handles both horizontal text and vertical text layouts
        return (horizontal_gap < threshold and vertical_gap < threshold * 2) or \
               (vertical_gap < threshold and horizontal_gap < threshold * 2)
    
    def _find_font(self) -> str:
        """Find a suitable font for text rendering"""
        font_candidates = [
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/calibri.ttf", 
            "C:/Windows/Fonts/tahoma.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        ]
        
        for font_path in font_candidates:
            if os.path.exists(font_path):
                return font_path
        
        return None  # Will use default font
    
    def translate_regions(self, regions: List[TextRegion], image_path: str) -> List[TextRegion]:
        """Translate all text regions with API delay"""
        self._log(f"\n📝 Translating {len(regions)} text regions...")
        
        # Check stop before even starting
        if self._check_stop():
            self._log(f"\n⏹️ Translation stopped before processing any regions", "warning")
            return regions
        
        for i, region in enumerate(regions):
            if self._check_stop():
                self._log(f"\n⏹️ Translation stopped by user after {i}/{len(regions)} regions", "warning")
                break            
            if region.text.strip():
                self._log(f"\n[{i+1}/{len(regions)}] Original: {region.text}")
                
                # Get context for translation
                context = self.translation_context[-5:] if self.contextual_enabled else None
                
                # Translate with image context
                translated = self.translate_text(
                    region.text, 
                    context,
                    image_path=image_path,
                    region=region
                )
                region.translated_text = translated
                
                self._log(f"Translated: {translated}")
                
                # Apply API delay
                if i < len(regions) - 1:  # Don't delay after last translation
                    self._log(f"⏳ Waiting {self.api_delay}s before next translation...")
                    # Check stop flag every 0.1 seconds during delay
                    for _ in range(int(self.api_delay * 10)):
                        if self._check_stop():
                            self._log(f"\n⏹️ Translation stopped during delay", "warning")
                            return regions
                        time.sleep(0.1)
        
        return regions

    def process_image(self, image_path: str, output_path: Optional[str] = None) -> Dict[str, Any]:
        """Process a single manga image through the full pipeline"""
        
        self._log(f"\n{'='*60}")
        self._log(f"🖼️ STARTING MANGA TRANSLATION PIPELINE")
        self._log(f"📁 Input: {image_path}")
        self._log(f"📁 Output: {output_path or 'Auto-generated'}")
        self._log(f"{'='*60}\n")
        
        result = {
            'success': False,
            'input_path': image_path,
            'output_path': output_path,
            'regions': [],
            'errors': [],
            'interrupted': False
        }
        
        try:
            # Check for stop signal
            if self._check_stop():
                result['interrupted'] = True
                self._log("⏹️ Translation stopped before processing", "warning")
                return result
            
            # Step 1: Detect text regions using Google Cloud Vision
            self._log(f"📍 [STEP 1] Text Detection Phase")
            regions = self.detect_text_regions(image_path)
            
            if not regions:
                error_msg = "No text regions detected by Cloud Vision"
                self._log(f"⚠️ {error_msg}", "warning")
                result['errors'].append(error_msg)
                # Still save the original image as "translated" if no text found
                if output_path:
                    import shutil
                    shutil.copy2(image_path, output_path)
                    result['output_path'] = output_path
                result['success'] = True  # Consider it a success if no text to translate
                return result
            
            self._log(f"\n✅ Detection complete: {len(regions)} regions found")
            
            # Step 2: Translate regions (full page context or individual based on settings)
            self._log(f"\n📍 [STEP 2] Translation Phase")
            
            if self.full_page_context_enabled:
                # Full page context translation mode
                self._log(f"\n📄 Using FULL PAGE CONTEXT mode")
                self._log("   This mode sends all text together for more consistent translations", "info")
                self._log(f"   ⚠️ Note: Full page context uses a single API call - stopping may take a moment")
                
                # Check for stop signal
                if self._check_stop():
                    result['interrupted'] = True
                    self._log("\n⏹️ Translation stopped before processing", "warning")
                    return result

                
                # Check for stop signal
                if self._check_stop():
                    result['interrupted'] = True
                    self._log("\n⏹️ Translation stopped before processing", "warning")
                    return result
                
                # Translate all regions at once with full context
                translations = self.translate_full_page_context(regions, image_path)
                
                if translations:
                    # Apply translations to regions
                    translated_count = 0
                    for region in regions:
                        if region.text in translations:
                            region.translated_text = translations[region.text]
                            translated_count += 1
                            self._log(f"   ✅ Applied translation for: '{region.text[:30]}...'")
                        else:
                            self._log(f"   ⚠️ No translation found for: '{region.text[:30]}...'", "warning")
                    
                    self._log(f"\n📊 Full page context translation complete: {translated_count}/{len(regions)} regions translated")
                else:
                    self._log("❌ Full page context translation failed - no translations returned", "error")
                    result['errors'].append("Full page context translation failed")
                    
            else:
                # Individual translation mode (original behavior)
                self._log(f"\n📝 Using INDIVIDUAL translation mode")
                
                regions = self.translate_regions(regions, image_path)

            # Check if we should continue after translation
            if self._check_stop():
                result['interrupted'] = True
                self._log("⏹️ Translation cancelled before image processing", "warning")
                result['regions'] = [r.to_dict() for r in regions]
                return result

            # Also check if any regions were actually translated
            if not any(region.translated_text for region in regions):
                result['interrupted'] = True
                self._log("⏹️ No regions were translated - translation was interrupted", "warning")
                result['regions'] = [r.to_dict() for r in regions]
                return result
            
            # Step 3: Render translated text
            self._log(f"\n📍 [STEP 3] Image Processing Phase")
            
            import cv2
            self._log(f"🖼️ Loading image with OpenCV...")
            image = cv2.imread(image_path)
            self._log(f"   Image dimensions: {image.shape[1]}x{image.shape[0]}")
            
            # Check if we should skip inpainting based on user preference
            if self.skip_inpainting:
                # User wants to preserve original art
                self._log(f"🎨 Skipping inpainting (preserving original art)", "info")
                self._log(f"   Background opacity: {int(self.text_bg_opacity/255*100)}%", "info")
                inpainted = image.copy()
            else:
                # Normal inpainting flow
                self._log(f"🎭 Creating text mask...")
                mask = self.create_text_mask(image, regions)
                
                self._log(f"🎨 Inpainting to remove original text")
                inpainted = self.inpaint_regions(image, mask)
            
            # Render translated text
            self._log(f"✍️ Rendering translated text...")
            self._log(f"   Using enhanced renderer with custom settings", "info")
            final_image = self.render_translated_text(inpainted, regions)
            
            # Save output
            if output_path:
                cv2.imwrite(output_path, final_image)
                result['output_path'] = output_path
            else:
                # Generate output path
                base, ext = os.path.splitext(image_path)
                output_path = f"{base}_translated{ext}"
                cv2.imwrite(output_path, final_image)
                result['output_path'] = output_path
            
            self._log(f"\n💾 Saved output to: {output_path}")

            # Update result
            result['regions'] = [r.to_dict() for r in regions]
            # Only mark as success if we completed everything
            if not result.get('interrupted', False):
                result['success'] = True
                self._log(f"\n✅ TRANSLATION PIPELINE COMPLETE", "success")
            else:
                self._log(f"\n⚠️ TRANSLATION INTERRUPTED - Partial output saved", "warning")

            self._log(f"{'='*60}\n")
            
        except Exception as e:
            error_msg = f"Error processing image: {str(e)}\n{traceback.format_exc()}"
            self._log(f"\n❌ PIPELINE ERROR:", "error")
            self._log(f"   {str(e)}", "error")
            self._log(f"   Type: {type(e).__name__}", "error")
            self._log(traceback.format_exc(), "error")
            result['errors'].append(error_msg)
        
        return result
