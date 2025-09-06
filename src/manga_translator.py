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
import cv2
from PIL import ImageEnhance, ImageFilter
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from bubble_detector import BubbleDetector

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
    
    def __init__(self, ocr_config: dict, unified_client, main_gui, log_callback=None):
        """Initialize with OCR configuration and API client from main GUI
        
        Args:
            ocr_config: Dictionary with OCR provider settings:
                {
                    'provider': 'google' or 'azure',
                    'google_credentials_path': str (if google),
                    'azure_key': str (if azure),
                    'azure_endpoint': str (if azure)
                }
        """
        self.main_gui = main_gui
        self.config = main_gui.config
        self.manga_settings = self.config.get('manga_settings', {})
        
        # Initialize attributes
        self.current_image = None
        self.current_mask = None
        self.text_regions = []
        self.translated_regions = []
        self.final_image = None
        
        # Initialize inpainter attributes (ADD THESE)
        self.local_inpainter = None
        self.hybrid_inpainter = None
        self.inpainter = None
        
        # Initialize bubble detector
        self.bubble_detector = None
        
        # Processing flags
        self.is_processing = False
        self.cancel_requested = False
        
        # Cache for processed images
        self.cache = {}
        # Determine OCR provider
        self.ocr_provider = ocr_config.get('provider', 'google')
        self.bubble_detector = None

        if self.ocr_provider == 'google':
            if not GOOGLE_CLOUD_VISION_AVAILABLE:
                raise ImportError("Google Cloud Vision required. Install with: pip install google-cloud-vision")
            
            google_path = ocr_config.get('google_credentials_path')
            if not google_path:
                raise ValueError("Google credentials path required")
                
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = google_path
            self.vision_client = vision.ImageAnnotatorClient()
            
        elif self.ocr_provider == 'azure':
            # Import Azure libraries
            try:
                from azure.cognitiveservices.vision.computervision import ComputerVisionClient
                from msrest.authentication import CognitiveServicesCredentials
                self.azure_cv = ComputerVisionClient
                self.azure_creds = CognitiveServicesCredentials
            except ImportError:
                raise ImportError("Azure Computer Vision required. Install with: pip install azure-cognitiveservices-vision-computervision")
            
            azure_key = ocr_config.get('azure_key')
            azure_endpoint = ocr_config.get('azure_endpoint')
            
            if not azure_key or not azure_endpoint:
                raise ValueError("Azure key and endpoint required")
                
            self.vision_client = self.azure_cv(
                azure_endpoint,
                self.azure_creds(azure_key)
            )
        else:
            # New OCR providers handled by OCR manager
            from ocr_manager import OCRManager
            self.ocr_manager = OCRManager(log_callback=log_callback)
            print(f"Initialized OCR Manager for {self.ocr_provider}")
        
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
        
        # Initialize HistoryManager placeholder
        self.history_manager = None
        self.history_manager_initialized = False
        self.history_output_dir = None
        
        # Full page context translation settings
        self.full_page_context_enabled = True
        
        # Default prompt for full page context mode
        self.full_page_context_prompt = (
            "You will receive multiple text segments from a manga page. "
            "Translate each segment considering the context of all segments together. "
            "Maintain consistency in character names, tone, and style across all translations.\n\n"
            "IMPORTANT: Return your response as a valid JSON object where each key is the EXACT original text "
            "(without the [0], [1] index prefixes) and each value is the translation.\n"
            "Make sure to properly escape any special characters in the JSON:\n"
            "- Use \\n for newlines\n"
            "- Use \\\" for quotes\n"
            "- Use \\\\ for backslashes\n\n"
            "Example:\n"
            '{\n'
            '  こんにちは: Hello,\n'
            '  ありがとう: Thank you\n'
            '}\n\n'
            'Do NOT include the [0], [1], etc. prefixes in the JSON keys.'
        )

        # Visual context setting (for non-vision model support)
        self.visual_context_enabled = main_gui.config.get('manga_visual_context_enabled', True)
        
        # Store context for contextual translation (backwards compatibility)
        self.translation_context = []
        
        # Font settings for text rendering
        self.font_path = self._find_font()
        self.min_font_size = 10
        self.max_font_size = 60
        self.min_readable_size = main_gui.config.get('manga_min_readable_size', 16)
        self.max_font_size_limit = main_gui.config.get('manga_max_font_size', 24)
        self.strict_text_wrapping = main_gui.config.get('manga_strict_text_wrapping', False)
        
        # Enhanced text rendering settings - Load from config if available
        config = main_gui.config if hasattr(main_gui, 'config') else {}
        
        self.text_bg_opacity = config.get('manga_bg_opacity', 255)  # 0-255, default fully opaque
        self.text_bg_style = config.get('manga_bg_style', 'box')  # 'box', 'circle', 'wrap'
        self.text_bg_reduction = config.get('manga_bg_reduction', 1.0)  # Size reduction factor (0.5-1.0)
        self.constrain_to_bubble = config.get('manga_constrain_to_bubble', True) 
        
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
        
        #inpainting quality
        self.inpaint_quality = config.get('manga_inpaint_quality', 'high')  # 'high' or 'fast'        
        
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
        
        self.manga_settings = config.get('manga_settings', {})

        # Initialize local inpainter if configured
        if self.manga_settings.get('inpainting', {}).get('method') == 'local':
            self._initialize_local_inpainter()
            
        # advanced settings
        self.debug_mode = self.manga_settings.get('advanced', {}).get('debug_mode', False)
        self.save_intermediate = self.manga_settings.get('advanced', {}).get('save_intermediate', False)
        self.parallel_processing = self.manga_settings.get('advanced', {}).get('parallel_processing', False)
        self.max_workers = self.manga_settings.get('advanced', {}).get('max_workers', 4)
        
            
    def set_stop_flag(self, stop_flag):
        """Set the stop flag for checking interruptions"""
        self.stop_flag = stop_flag

    def _check_stop(self):
        """Check if stop has been requested"""
        if self.stop_flag and self.stop_flag.is_set():
            return True
        return False

    def _merge_with_bubble_detection(self, regions: List[TextRegion], image_path: str) -> List[TextRegion]:
        """Merge text regions by bubble and filter based on RT-DETR class settings"""
        try:
            # Get detector settings from config
            ocr_settings = self.main_gui.config.get('manga_settings', {}).get('ocr', {})
            detector_type = ocr_settings.get('detector_type', 'yolo')
            
            # Check if bubble detection is enabled
            if not ocr_settings.get('bubble_detection_enabled', False):
                self._log("📦 Bubble detection is disabled in settings", "info")
                return self._merge_nearby_regions(regions)
            
            # Initialize detector if needed
            if self.bubble_detector is None:
                from bubble_detector import BubbleDetector
                self.bubble_detector = BubbleDetector()
            
            bubbles = None
            rtdetr_detections = None  # Store full RT-DETR results for filtering
            
            if detector_type == 'rtdetr':
                # Use RT-DETR
                self._log("🤖 Using RT-DETR for bubble detection", "info")
                
                # Load RT-DETR if needed
                if not self.bubble_detector.rtdetr_loaded:
                    self._log("📥 Loading RT-DETR model...", "info")
                    if not self.bubble_detector.load_rtdetr_model():
                        self._log("⚠️ Failed to load RT-DETR, falling back to traditional merging", "warning")
                        return self._merge_nearby_regions(regions)
                
                # Get RT-DETR settings including class filters
                rtdetr_confidence = ocr_settings.get('rtdetr_confidence', 0.3)
                detect_empty = ocr_settings.get('detect_empty_bubbles', True)
                detect_text_bubbles = ocr_settings.get('detect_text_bubbles', True)
                detect_free_text = ocr_settings.get('detect_free_text', True)
                
                self._log(f"📋 RT-DETR class filters:", "info")
                self._log(f"   Empty bubbles: {'✓' if detect_empty else '✗'}", "info")
                self._log(f"   Text bubbles: {'✓' if detect_text_bubbles else '✗'}", "info")
                self._log(f"   Free text: {'✓' if detect_free_text else '✗'}", "info")
                self._log(f"🎯 RT-DETR confidence threshold: {rtdetr_confidence:.2f}", "info")

                # Get FULL RT-DETR detections (not just bubbles)
                rtdetr_detections = self.bubble_detector.detect_with_rtdetr(
                    image_path=image_path,
                    confidence=rtdetr_confidence,
                    return_all_bubbles=False  # Get dict with all classes
                )
                
                # Combine enabled bubble types for merging
                bubbles = []
                if detect_empty and 'bubbles' in rtdetr_detections:
                    bubbles.extend(rtdetr_detections['bubbles'])
                if detect_text_bubbles and 'text_bubbles' in rtdetr_detections:
                    bubbles.extend(rtdetr_detections['text_bubbles'])
                
                # Store free text locations for filtering later
                free_text_regions = rtdetr_detections.get('text_free', []) if detect_free_text else []
                
                self._log(f"✅ RT-DETR detected:", "success")
                self._log(f"   {len(rtdetr_detections.get('bubbles', []))} empty bubbles", "info")
                self._log(f"   {len(rtdetr_detections.get('text_bubbles', []))} text bubbles", "info")
                self._log(f"   {len(rtdetr_detections.get('text_free', []))} free text regions", "info")
                
            elif detector_type == 'yolo':
                # Use YOLOv8 (existing code)
                self._log("🤖 Using YOLOv8 for bubble detection", "info")
                
                model_path = ocr_settings.get('bubble_model_path')
                if not model_path:
                    self._log("⚠️ No YOLO model configured, falling back to traditional merging", "warning")
                    return self._merge_nearby_regions(regions)
                
                if not self.bubble_detector.model_loaded:
                    self._log(f"📥 Loading YOLO model: {os.path.basename(model_path)}")
                    if not self.bubble_detector.load_model(model_path):
                        self._log("⚠️ Failed to load YOLO model, falling back to traditional merging", "warning")
                        return self._merge_nearby_regions(regions)
                
                confidence = ocr_settings.get('bubble_confidence', 0.5)
                self._log(f"🎯 Detecting bubbles with YOLO (confidence >= {confidence:.2f})")
                bubbles = self.bubble_detector.detect_bubbles(image_path, confidence=confidence, use_rtdetr=False)
                
            else:  # auto mode
                self._log("🤖 Auto mode: using best available detector", "info")
                
                if not self.bubble_detector.rtdetr_loaded:
                    self.bubble_detector.load_rtdetr_model()
                
                confidence = ocr_settings.get('bubble_confidence', 0.5)
                bubbles = self.bubble_detector.detect_bubbles(
                    image_path, 
                    confidence=confidence,
                    use_rtdetr=None
                )
            
            if not bubbles:
                self._log("⚠️ No bubbles detected, using traditional merging", "warning")
                return self._merge_nearby_regions(regions)
            
            self._log(f"✅ Found {len(bubbles)} bubbles for grouping", "success")
            
            # Merge regions within bubbles
            merged_regions = []
            used_indices = set()
            
            for bubble_idx, (bx, by, bw, bh) in enumerate(bubbles):
                bubble_regions = []
                
                for idx, region in enumerate(regions):
                    if idx in used_indices:
                        continue
                        
                    rx, ry, rw, rh = region.bounding_box
                    region_center_x = rx + rw / 2
                    region_center_y = ry + rh / 2
                    
                    if (bx <= region_center_x <= bx + bw and 
                        by <= region_center_y <= by + bh):
                        bubble_regions.append(region)
                        used_indices.add(idx)
                
                if bubble_regions:
                    merged_text = " ".join(r.text for r in bubble_regions)
                    
                    min_x = min(r.bounding_box[0] for r in bubble_regions)
                    min_y = min(r.bounding_box[1] for r in bubble_regions)
                    max_x = max(r.bounding_box[0] + r.bounding_box[2] for r in bubble_regions)
                    max_y = max(r.bounding_box[1] + r.bounding_box[3] for r in bubble_regions)
                    
                    all_vertices = []
                    for r in bubble_regions:
                        if hasattr(r, 'vertices') and r.vertices:
                            all_vertices.extend(r.vertices)
                    
                    if not all_vertices:
                        all_vertices = [
                            (min_x, min_y),
                            (max_x, min_y),
                            (max_x, max_y),
                            (min_x, max_y)
                        ]
                    
                    merged_region = TextRegion(
                        text=merged_text,
                        vertices=all_vertices,
                        bounding_box=(min_x, min_y, max_x - min_x, max_y - min_y),
                        confidence=0.95,
                        region_type='bubble_detected'
                    )
                    
                    # Store original regions for masking
                    merged_region.original_regions = bubble_regions
                    merged_region.bubble_bounds = (bx, by, bw, bh)
                    # Mark that this should be inpainted
                    merged_region.should_inpaint = True
                    
                    merged_regions.append(merged_region)
                    self._log(f"   Bubble {bubble_idx + 1}: Merged {len(bubble_regions)} text regions", "info")
            
            # Handle text outside bubbles based on RT-DETR settings
            for idx, region in enumerate(regions):
                if idx not in used_indices:
                    # This text is outside any bubble
                    
                    # For RT-DETR mode, check if we should include free text
                    if detector_type == 'rtdetr':
                        # If "Free Text" checkbox is checked, include ALL text outside bubbles
                        # Don't require RT-DETR to specifically detect it as free text
                        if ocr_settings.get('detect_free_text', True):
                            region.should_inpaint = True
                            self._log(f"   Text outside bubbles INCLUDED: '{region.text[:30]}...'", "debug")
                        else:
                            region.should_inpaint = False
                            self._log(f"   Text outside bubbles EXCLUDED (Free Text unchecked): '{region.text[:30]}...'", "info")
                    else:
                        # For YOLO/auto, include all text by default
                        region.should_inpaint = True
                    
                    merged_regions.append(region)

            # Log summary
            regions_to_inpaint = sum(1 for r in merged_regions if getattr(r, 'should_inpaint', True))
            regions_to_skip = len(merged_regions) - regions_to_inpaint

            self._log(f"📊 Bubble detection complete: {len(regions)} → {len(merged_regions)} regions", "success")
            if detector_type == 'rtdetr':
                self._log(f"   {regions_to_inpaint} regions will be inpainted", "info")
                if regions_to_skip > 0:
                    self._log(f"   {regions_to_skip} regions will be preserved (Free Text unchecked)", "info")

            return merged_regions
            
        except Exception as e:
            self._log(f"❌ Bubble detection error: {str(e)}", "error")
            self._log("   Falling back to traditional merging", "warning")
            return self._merge_nearby_regions(regions)
        
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

    def _is_primarily_english(self, text: str) -> bool:
        """Check if text is primarily English/ASCII characters"""
        if not text:
            return False
        
        # Check for CJK characters FIRST - if found, it's NOT English
        has_cjk = any(
            '\u4e00' <= char <= '\u9fff' or  # Chinese
            '\u3040' <= char <= '\u309f' or  # Hiragana  
            '\u30a0' <= char <= '\u30ff' or  # Katakana
            '\uac00' <= char <= '\ud7af' or  # Korean
            '\uff00' <= char <= '\uffef'     # Full-width characters
            for char in text
        )
        
        if has_cjk:
            return False  # Has Asian characters, NOT English
        
        # Only NOW check for English patterns
        text_stripped = text.strip()
        
        # Single ASCII letters
        if len(text_stripped) == 1 and text_stripped.isalpha() and ord(text_stripped) < 128:
            self._log(f"   Excluding single English letter: '{text_stripped}'", "debug")
            return True
        
        # Short English text
        if len(text_stripped) <= 3:
            ascii_letters = sum(1 for char in text_stripped if char.isalpha() and ord(char) < 128)
            if ascii_letters >= len(text_stripped) * 0.5:
                self._log(f"   Excluding short English text: '{text_stripped}'", "debug")
                return True
        
        # Count ASCII (excluding spaces which manga-ocr adds)
        ascii_chars = sum(1 for char in text if 33 <= ord(char) <= 126)
        total_chars = sum(1 for char in text if not char.isspace())
        
        if total_chars == 0:
            return False
        
        ratio = ascii_chars / total_chars
        
        if ratio > 0.7:
            self._log(f"   Excluding English text ({ratio:.0%} ASCII): '{text[:30]}...'", "debug")
            return True
        
        return False
            
    def detect_text_regions(self, image_path: str) -> List[TextRegion]:
        """Detect text regions using configured OCR provider"""
        self._log(f"🔍 Detecting text regions in: {os.path.basename(image_path)}")
        self._log(f"   Using OCR provider: {self.ocr_provider.upper()}")
        
        try:
            # CLEAR ANY CACHED STATE FROM PREVIOUS IMAGE
            if hasattr(self, 'ocr_manager') and self.ocr_manager:
                # Clear any cached results in OCR manager
                if hasattr(self.ocr_manager, 'last_results'):
                    self.ocr_manager.last_results = None
                if hasattr(self.ocr_manager, 'cache'):
                    self.ocr_manager.cache = {}
            
            # Clear bubble detector cache if it exists
            if hasattr(self, 'bubble_detector') and self.bubble_detector:
                if hasattr(self.bubble_detector, 'last_detections'):
                    self.bubble_detector.last_detections = None
            
            # Get manga settings from main_gui config
            manga_settings = self.main_gui.config.get('manga_settings', {})
            preprocessing = manga_settings.get('preprocessing', {})
            ocr_settings = manga_settings.get('ocr', {})
            
            # Get text filtering settings
            min_text_length = ocr_settings.get('min_text_length', 2)
            exclude_english = ocr_settings.get('exclude_english_text', True)
            confidence_threshold = ocr_settings.get('confidence_threshold', 0.1)
            
            # Load and preprocess image if enabled
            if preprocessing.get('enabled', True):
                self._log("📐 Preprocessing enabled - enhancing image quality")
                processed_image_data = self._preprocess_image(image_path, preprocessing)
            else:
                # Read image file without preprocessing
                with open(image_path, 'rb') as image_file:
                    processed_image_data = image_file.read()
            
            regions = []
            
            # Route to appropriate provider
            if self.ocr_provider == 'google':
                # === GOOGLE CLOUD VISION (unchanged) ===
                # Create Vision API image object
                image = vision.Image(content=processed_image_data)
                
                # Build image context with all parameters
                image_context = vision.ImageContext(
                    language_hints=ocr_settings.get('language_hints', ['ja', 'ko', 'zh'])
                )
                
                # Add text detection params if available in your API version
                if hasattr(vision, 'TextDetectionParams'):
                    image_context.text_detection_params = vision.TextDetectionParams(
                        enable_text_detection_confidence_score=True
                    )
                
                # Configure text detection based on settings
                detection_mode = ocr_settings.get('text_detection_mode', 'document')
                
                if detection_mode == 'document':
                    response = self.vision_client.document_text_detection(
                        image=image,
                        image_context=image_context
                    )
                else:
                    response = self.vision_client.text_detection(
                        image=image,
                        image_context=image_context
                    )
                
                if response.error.message:
                    raise Exception(f"Cloud Vision API error: {response.error.message}")
                
                # Process each page (usually just one for manga)
                for page in response.full_text_annotation.pages:
                    for block in page.blocks:
                        # Extract text first to check if it's worth processing
                        block_text = ""
                        total_confidence = 0.0
                        word_count = 0
                        
                        for paragraph in block.paragraphs:
                            for word in paragraph.words:
                                # Get word-level confidence (more reliable than block level)
                                word_confidence = getattr(word, 'confidence', 0.0)  # Default to 0 if not available
                                word_text = ''.join([symbol.text for symbol in word.symbols])
                                
                                # Only include words above threshold
                                if word_confidence >= confidence_threshold:
                                    block_text += word_text + " "
                                    total_confidence += word_confidence
                                    word_count += 1
                                else:
                                    self._log(f"   Skipping low confidence word ({word_confidence:.2f}): {word_text}")
                        
                        block_text = block_text.strip()
                        
                        # TEXT FILTERING SECTION
                        # Skip if text is too short
                        if len(block_text) < min_text_length:
                            self._log(f"   Skipping short text ({len(block_text)} chars): {block_text}")
                            continue
                        
                        # Skip if primarily English and exclude_english is enabled
                        if exclude_english and self._is_primarily_english(block_text):
                            self._log(f"   Skipping English text: {block_text[:50]}...")
                            continue
                        
                        # Skip if no confident words found
                        if word_count == 0 or not block_text:
                            self._log(f"   Skipping block - no words above threshold {confidence_threshold}")
                            continue
                        
                        # Calculate average confidence for the block
                        avg_confidence = total_confidence / word_count if word_count > 0 else 0.0
                        
                        # Extract vertices and create region
                        vertices = [(v.x, v.y) for v in block.bounding_box.vertices]
                        
                        # Calculate bounding box
                        xs = [v[0] for v in vertices]
                        ys = [v[1] for v in vertices]
                        x_min, x_max = min(xs), max(xs)
                        y_min, y_max = min(ys), max(ys)
                        
                        region = TextRegion(
                            text=block_text,
                            vertices=vertices,
                            bounding_box=(x_min, y_min, x_max - x_min, y_max - y_min),
                            confidence=avg_confidence,  # Use average confidence
                            region_type='text_block'
                        )
                        regions.append(region)
                        self._log(f"   Found text region ({avg_confidence:.2f}): {block_text[:50]}...")
                        
            elif self.ocr_provider == 'azure':
                # === AZURE COMPUTER VISION (unchanged) ===
                import io
                import time
                from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
                
                # Check if image needs format conversion for Azure
                file_ext = os.path.splitext(image_path)[1].lower()
                azure_supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.pdf', '.tiff']
                
                if file_ext == '.webp' or file_ext not in azure_supported_formats:
                    self._log(f"⚠️ Converting {file_ext} to PNG for Azure compatibility")
                    from PIL import Image
                    img = Image.open(io.BytesIO(processed_image_data))
                    buffer = io.BytesIO()
                    img.save(buffer, format='PNG')
                    processed_image_data = buffer.getvalue()
                
                # Create stream from image data
                image_stream = io.BytesIO(processed_image_data)
                
                # Get Azure-specific settings
                reading_order = ocr_settings.get('azure_reading_order', 'natural')
                model_version = ocr_settings.get('azure_model_version', 'latest')
                max_wait = ocr_settings.get('azure_max_wait', 60)
                poll_interval = ocr_settings.get('azure_poll_interval', 0.5)
                
                # Map language hints to Azure language codes
                language_hints = ocr_settings.get('language_hints', ['ja', 'ko', 'zh'])
                
                # Build parameters dictionary
                read_params = {
                    'raw': True,
                    'readingOrder': reading_order
                }
                
                # Add model version if not using latest
                if model_version != 'latest':
                    read_params['model-version'] = model_version
                
                # Use language parameter only if single language is selected
                if len(language_hints) == 1:
                    azure_lang = language_hints[0]
                    # Map to Azure language codes
                    lang_mapping = {
                        'zh': 'zh-Hans',
                        'zh-TW': 'zh-Hant',
                        'zh-CN': 'zh-Hans',
                        'ja': 'ja',
                        'ko': 'ko',
                        'en': 'en'
                    }
                    azure_lang = lang_mapping.get(azure_lang, azure_lang)
                    read_params['language'] = azure_lang
                    self._log(f"   Using Azure Read API with language: {azure_lang}, order: {reading_order}")
                else:
                    self._log(f"   Using Azure Read API (auto-detect for {len(language_hints)} languages, order: {reading_order})")
                
                # Start Read operation with error handling
                try:
                    read_response = self.vision_client.read_in_stream(
                        image_stream,
                        **read_params
                    )
                except Exception as e:
                    error_msg = str(e)
                    if 'Bad Request' in error_msg:
                        self._log("❌ Azure Read API Bad Request - retrying without language parameter", "error")
                        # Retry without language parameter
                        image_stream.seek(0)
                        read_params.pop('language', None)
                        read_response = self.vision_client.read_in_stream(
                            image_stream,
                            **read_params
                        )
                    else:
                        raise
                
                # Get operation ID
                operation_location = read_response.headers["Operation-Location"]
                operation_id = operation_location.split("/")[-1]
                
                # Poll for results with configurable timeout
                self._log(f"   Waiting for Azure OCR to complete (max {max_wait}s)...")
                wait_time = 0
                last_status = None
                
                while wait_time < max_wait:
                    result = self.vision_client.get_read_result(operation_id)
                    
                    # Log status changes
                    if result.status != last_status:
                        self._log(f"   Status: {result.status}")
                        last_status = result.status
                    
                    if result.status not in [OperationStatusCodes.running, OperationStatusCodes.not_started]:
                        break
                    
                    time.sleep(poll_interval)
                    wait_time += poll_interval
                
                if result.status == OperationStatusCodes.succeeded:
                    # Track statistics
                    total_lines = 0
                    handwritten_lines = 0
                    
                    for page_num, page in enumerate(result.analyze_result.read_results):
                        if len(result.analyze_result.read_results) > 1:
                            self._log(f"   Processing page {page_num + 1}/{len(result.analyze_result.read_results)}")
                        
                        for line in page.lines:
                            # TEXT FILTERING FOR AZURE
                            # Skip if text is too short
                            if len(line.text) < min_text_length:
                                self._log(f"   Skipping short text ({len(line.text)} chars): {line.text}")
                                continue
                            
                            # Skip if primarily English and exclude_english is enabled
                            if exclude_english and self._is_primarily_english(line.text):
                                self._log(f"   Skipping English text: {line.text[:50]}...")
                                continue
                            
                            # Azure provides 8-point bounding box
                            bbox = line.bounding_box
                            vertices = [
                                (bbox[0], bbox[1]),
                                (bbox[2], bbox[3]),
                                (bbox[4], bbox[5]),
                                (bbox[6], bbox[7])
                            ]
                            
                            # Calculate rectangular bounding box
                            xs = [v[0] for v in vertices]
                            ys = [v[1] for v in vertices]
                            x_min, x_max = min(xs), max(xs)
                            y_min, y_max = min(ys), max(ys)
                            
                            # Calculate confidence from word-level data
                            confidence = 0.95  # Default high confidence
                            
                            if hasattr(line, 'words') and line.words:
                                # Calculate average confidence from words
                                confidences = []
                                for word in line.words:
                                    if hasattr(word, 'confidence'):
                                        confidences.append(word.confidence)
                                
                                if confidences:
                                    confidence = sum(confidences) / len(confidences)
                                    self._log(f"   Line has {len(line.words)} words, avg confidence: {confidence:.3f}")
                            
                            # Check for handwriting style (if available)
                            style = 'print'  # Default
                            style_confidence = None
                            
                            if hasattr(line, 'appearance') and line.appearance:
                                if hasattr(line.appearance, 'style'):
                                    style_info = line.appearance.style
                                    if hasattr(style_info, 'name'):
                                        style = style_info.name
                                        if style == 'handwriting':
                                            handwritten_lines += 1
                                    if hasattr(style_info, 'confidence'):
                                        style_confidence = style_info.confidence
                                        self._log(f"   Style: {style} (confidence: {style_confidence:.2f})")
                            
                            # Apply confidence threshold filtering
                            if confidence >= confidence_threshold:
                                region = TextRegion(
                                    text=line.text,
                                    vertices=vertices,
                                    bounding_box=(x_min, y_min, x_max - x_min, y_max - y_min),
                                    confidence=confidence,
                                    region_type='text_line'
                                )
                                
                                # Add extra attributes for Azure-specific info
                                region.style = style
                                region.style_confidence = style_confidence
                                
                                regions.append(region)
                                total_lines += 1
                                
                                # More detailed logging
                                if style == 'handwriting':
                                    self._log(f"   Found handwritten text ({confidence:.2f}): {line.text[:50]}...")
                                else:
                                    self._log(f"   Found text region ({confidence:.2f}): {line.text[:50]}...")
                            else:
                                self._log(f"   Skipping low confidence text ({confidence:.2f}): {line.text[:30]}...")
                    
                    # Log summary statistics
                    if total_lines > 0:
                        self._log(f"   Total lines detected: {total_lines}")
                        if handwritten_lines > 0:
                            self._log(f"   Handwritten lines: {handwritten_lines} ({handwritten_lines/total_lines*100:.1f}%)")
                    
                elif result.status == OperationStatusCodes.failed:
                    # More detailed error handling
                    error_msg = "Azure OCR failed"
                    if hasattr(result, 'message'):
                        error_msg += f": {result.message}"
                    if hasattr(result.analyze_result, 'errors') and result.analyze_result.errors:
                        for error in result.analyze_result.errors:
                            self._log(f"   Error: {error}", "error")
                    raise Exception(error_msg)
                else:
                    # Timeout or other status
                    raise Exception(f"Azure OCR ended with status: {result.status} after {wait_time}s")
                    
            else:
                # === NEW OCR PROVIDERS ===
                import cv2
                import numpy as np
                from ocr_manager import OCRManager
                
                # Load image as numpy array
                if isinstance(processed_image_data, bytes):
                    # Convert bytes to numpy array
                    nparr = np.frombuffer(processed_image_data, np.uint8)
                    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                else:
                    # Load from file path
                    image = cv2.imread(image_path)
                    if image is None:
                        # Try with PIL for Unicode paths
                        from PIL import Image as PILImage
                        pil_image = PILImage.open(image_path)
                        image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                
                # Create OCR manager if not exists
                if not hasattr(self, 'ocr_manager'):
                    self.ocr_manager = OCRManager(log_callback=self._log)
                
                # Check provider status and load if needed
                provider_status = self.ocr_manager.check_provider_status(self.ocr_provider)
                
                if not provider_status['installed']:
                    self._log(f"❌ {self.ocr_provider} is not installed", "error")
                    self._log(f"   Please install it from the GUI settings", "error")
                    raise Exception(f"{self.ocr_provider} OCR provider is not installed")
                
                if not provider_status['loaded']:
                    self._log(f"🔥 Loading {self.ocr_provider} model...")
                    if not self.ocr_manager.load_provider(self.ocr_provider):
                        raise Exception(f"Failed to load {self.ocr_provider} model")
                
                # Initialize ocr_results here before any provider-specific code
                ocr_results = []
                
                # Special handling for manga-ocr (needs region detection first)
                if self.ocr_provider == 'manga-ocr':
                    # IMPORTANT: Initialize fresh results list
                    ocr_results = []
                    
                    # Check if we should use bubble detection for regions
                    if ocr_settings.get('bubble_detection_enabled', False):
                        self._log("📝 Using bubble detection regions for manga-ocr...")
                        
                        # Run bubble detection to get regions
                        if self.bubble_detector is None:
                            from bubble_detector import BubbleDetector
                            self.bubble_detector = BubbleDetector()
                        
                        # Get regions from bubble detector - ensure fresh detection
                        if self.bubble_detector.load_rtdetr_model():
                            # IMPORTANT: Get fresh detections for this specific image
                            rtdetr_detections = self.bubble_detector.detect_with_rtdetr(
                                image_path=image_path,
                                confidence=ocr_settings.get('rtdetr_confidence', 0.3),
                                return_all_bubbles=False
                            )
                            
                            # Process detections immediately and don't store
                            all_regions = []
                            
                            # ONLY ADD TEXT-CONTAINING REGIONS
                            # Skip empty bubbles since they shouldn't have text
                            if 'text_bubbles' in rtdetr_detections:
                                all_regions.extend(rtdetr_detections.get('text_bubbles', []))
                            if 'text_free' in rtdetr_detections:
                                all_regions.extend(rtdetr_detections.get('text_free', []))
                            
                            # DO NOT ADD empty bubbles - they're duplicates of text_bubbles
                            # if 'bubbles' in rtdetr_detections:  # <-- REMOVE THIS
                            #     all_regions.extend(rtdetr_detections.get('bubbles', []))
                            
                            self._log(f"📊 Processing {len(all_regions)} text-containing regions (skipping empty bubbles)")
                            
                            # Clear detection results after extracting regions
                            rtdetr_detections = None
                            
                            # Process each region with manga-ocr
                            for i, (x, y, w, h) in enumerate(all_regions):
                                cropped = image[y:y+h, x:x+w]
                                result = self.ocr_manager.detect_text(cropped, 'manga-ocr', confidence=confidence_threshold)
                                if result and len(result) > 0 and result[0].text.strip():
                                    result[0].bbox = (x, y, w, h)
                                    result[0].vertices = [(x, y), (x+w, y), (x+w, y+h), (x, y+h)]
                                    ocr_results.append(result[0])
                                    self._log(f"🔍 Processing region {i+1}/{len(all_regions)} with manga-ocr...")
                                    self._log(f"✅ Detected text: {result[0].text[:50]}...")
                            
                            # Clear regions list after processing
                            all_regions = None
                    else:
                        # NO bubble detection - just process full image
                        self._log("📝 Processing full image with manga-ocr (no bubble detection)")
                        ocr_results = self.ocr_manager.detect_text(image, self.ocr_provider, confidence=confidence_threshold)  
                    
                elif self.ocr_provider == 'pororo':
                    # Initialize results list
                    ocr_results = []
                    
                    # Configure Pororo for appropriate language
                    language_hints = ocr_settings.get('language_hints', ['ko'])
                    self._log("🇰🇷 Pororo OCR optimized for Korean text")
                    
                    # Check if we should use bubble detection for regions
                    if ocr_settings.get('bubble_detection_enabled', False):
                        self._log("📝 Using bubble detection regions for Pororo...")
                        
                        # Run bubble detection to get regions
                        if self.bubble_detector is None:
                            from bubble_detector import BubbleDetector
                            self.bubble_detector = BubbleDetector()
                        
                        # Get regions from bubble detector
                        if self.bubble_detector.load_rtdetr_model():
                            rtdetr_detections = self.bubble_detector.detect_with_rtdetr(
                                image_path=image_path,
                                confidence=ocr_settings.get('rtdetr_confidence', 0.3),
                                return_all_bubbles=False
                            )
                            
                            # Process only text-containing regions
                            all_regions = []
                            if 'text_bubbles' in rtdetr_detections:
                                all_regions.extend(rtdetr_detections.get('text_bubbles', []))
                            if 'text_free' in rtdetr_detections:
                                all_regions.extend(rtdetr_detections.get('text_free', []))
                            
                            self._log(f"📊 Processing {len(all_regions)} text regions with Pororo")
                            
                            # Process each region with Pororo
                            for i, (x, y, w, h) in enumerate(all_regions):
                                cropped = image[y:y+h, x:x+w]
                                result = self.ocr_manager.detect_text(cropped, 'pororo', confidence=confidence_threshold)
                                if result and len(result) > 0 and result[0].text.strip():
                                    result[0].bbox = (x, y, w, h)
                                    result[0].vertices = [(x, y), (x+w, y), (x+w, y+h), (x, y+h)]
                                    ocr_results.append(result[0])
                                    self._log(f"✅ Region {i+1}: {result[0].text[:50]}...")
                    else:
                        # Process full image without bubble detection
                        self._log("📝 Processing full image with Pororo")
                        ocr_results = self.ocr_manager.detect_text(image, self.ocr_provider)

                elif self.ocr_provider == 'easyocr':
                    # Initialize results list
                    ocr_results = []
                    
                    # Configure EasyOCR languages
                    language_hints = ocr_settings.get('language_hints', ['ja', 'en'])
                    validated_languages = self._validate_easyocr_languages(language_hints)
                    
                    easyocr_provider = self.ocr_manager.get_provider('easyocr')
                    if easyocr_provider:
                        if easyocr_provider.languages != validated_languages:
                            easyocr_provider.languages = validated_languages
                            easyocr_provider.is_loaded = False
                            self._log(f"🔥 Reloading EasyOCR with languages: {validated_languages}")
                            self.ocr_manager.load_provider('easyocr')
                    
                    # Check if we should use bubble detection
                    if ocr_settings.get('bubble_detection_enabled', False):
                        self._log("📝 Using bubble detection regions for EasyOCR...")
                        
                        # Run bubble detection to get regions
                        if self.bubble_detector is None:
                            from bubble_detector import BubbleDetector
                            self.bubble_detector = BubbleDetector()
                        
                        # Get regions from bubble detector
                        if self.bubble_detector.load_rtdetr_model():
                            rtdetr_detections = self.bubble_detector.detect_with_rtdetr(
                                image_path=image_path,
                                confidence=ocr_settings.get('rtdetr_confidence', 0.3),
                                return_all_bubbles=False
                            )
                            
                            # Process only text-containing regions
                            all_regions = []
                            if 'text_bubbles' in rtdetr_detections:
                                all_regions.extend(rtdetr_detections.get('text_bubbles', []))
                            if 'text_free' in rtdetr_detections:
                                all_regions.extend(rtdetr_detections.get('text_free', []))
                            
                            self._log(f"📊 Processing {len(all_regions)} text regions with EasyOCR")
                            
                            # Process each region with EasyOCR
                            for i, (x, y, w, h) in enumerate(all_regions):
                                cropped = image[y:y+h, x:x+w]
                                result = self.ocr_manager.detect_text(cropped, 'easyocr', confidence=confidence_threshold)
                                if result and len(result) > 0 and result[0].text.strip():
                                    result[0].bbox = (x, y, w, h)
                                    result[0].vertices = [(x, y), (x+w, y), (x+w, y+h), (x, y+h)]
                                    ocr_results.append(result[0])
                                    self._log(f"✅ Region {i+1}: {result[0].text[:50]}...")
                    else:
                        # Process full image without bubble detection
                        self._log("📝 Processing full image with EasyOCR")
                        ocr_results = self.ocr_manager.detect_text(image, self.ocr_provider)

                elif self.ocr_provider == 'paddleocr':
                    # Initialize results list
                    ocr_results = []
                    
                    # Configure PaddleOCR language
                    language_hints = ocr_settings.get('language_hints', ['ja'])
                    lang_map = {'ja': 'japan', 'ko': 'korean', 'zh': 'ch', 'en': 'en'}
                    paddle_lang = lang_map.get(language_hints[0] if language_hints else 'ja', 'japan')
                    
                    # Reload if language changed
                    paddle_provider = self.ocr_manager.get_provider('paddleocr')
                    if paddle_provider and paddle_provider.is_loaded:
                        if hasattr(paddle_provider.model, 'lang') and paddle_provider.model.lang != paddle_lang:
                            from paddleocr import PaddleOCR
                            paddle_provider.model = PaddleOCR(
                                use_angle_cls=True,
                                lang=paddle_lang,
                                use_gpu=True,
                                show_log=False
                            )
                            self._log(f"🔥 Reloaded PaddleOCR with language: {paddle_lang}")
                    
                    # Check if we should use bubble detection
                    if ocr_settings.get('bubble_detection_enabled', False):
                        self._log("📝 Using bubble detection regions for PaddleOCR...")
                        
                        # Run bubble detection to get regions
                        if self.bubble_detector is None:
                            from bubble_detector import BubbleDetector
                            self.bubble_detector = BubbleDetector()
                        
                        # Get regions from bubble detector
                        if self.bubble_detector.load_rtdetr_model():
                            rtdetr_detections = self.bubble_detector.detect_with_rtdetr(
                                image_path=image_path,
                                confidence=ocr_settings.get('rtdetr_confidence', 0.3),
                                return_all_bubbles=False
                            )
                            
                            # Process only text-containing regions
                            all_regions = []
                            if 'text_bubbles' in rtdetr_detections:
                                all_regions.extend(rtdetr_detections.get('text_bubbles', []))
                            if 'text_free' in rtdetr_detections:
                                all_regions.extend(rtdetr_detections.get('text_free', []))
                            
                            self._log(f"📊 Processing {len(all_regions)} text regions with PaddleOCR")
                            
                            # Process each region with PaddleOCR
                            for i, (x, y, w, h) in enumerate(all_regions):
                                cropped = image[y:y+h, x:x+w]
                                result = self.ocr_manager.detect_text(cropped, 'paddleocr', confidence=confidence_threshold)
                                if result and len(result) > 0 and result[0].text.strip():
                                    result[0].bbox = (x, y, w, h)
                                    result[0].vertices = [(x, y), (x+w, y), (x+w, y+h), (x, y+h)]
                                    ocr_results.append(result[0])
                                    self._log(f"✅ Region {i+1}: {result[0].text[:50]}...")
                    else:
                        # Process full image without bubble detection
                        self._log("📝 Processing full image with PaddleOCR")
                        ocr_results = self.ocr_manager.detect_text(image, self.ocr_provider)

                elif self.ocr_provider == 'doctr':
                    # Initialize results list
                    ocr_results = []
                    
                    self._log("📄 DocTR OCR for document text recognition")
                    
                    # Check if we should use bubble detection
                    if ocr_settings.get('bubble_detection_enabled', False):
                        self._log("📝 Using bubble detection regions for DocTR...")
                        
                        # Run bubble detection to get regions
                        if self.bubble_detector is None:
                            from bubble_detector import BubbleDetector
                            self.bubble_detector = BubbleDetector()
                        
                        # Get regions from bubble detector
                        if self.bubble_detector.load_rtdetr_model():
                            rtdetr_detections = self.bubble_detector.detect_with_rtdetr(
                                image_path=image_path,
                                confidence=ocr_settings.get('rtdetr_confidence', 0.3),
                                return_all_bubbles=False
                            )
                            
                            # Process only text-containing regions
                            all_regions = []
                            if 'text_bubbles' in rtdetr_detections:
                                all_regions.extend(rtdetr_detections.get('text_bubbles', []))
                            if 'text_free' in rtdetr_detections:
                                all_regions.extend(rtdetr_detections.get('text_free', []))
                            
                            self._log(f"📊 Processing {len(all_regions)} text regions with DocTR")
                            
                            # Process each region with DocTR
                            for i, (x, y, w, h) in enumerate(all_regions):
                                cropped = image[y:y+h, x:x+w]
                                result = self.ocr_manager.detect_text(cropped, 'doctr', confidence=confidence_threshold)
                                if result and len(result) > 0 and result[0].text.strip():
                                    result[0].bbox = (x, y, w, h)
                                    result[0].vertices = [(x, y), (x+w, y), (x+w, y+h), (x, y+h)]
                                    ocr_results.append(result[0])
                                    self._log(f"✅ Region {i+1}: {result[0].text[:50]}...")
                    else:
                        # Process full image without bubble detection
                        self._log("📝 Processing full image with DocTR")
                        ocr_results = self.ocr_manager.detect_text(image, self.ocr_provider)

                else:
                    # Default processing for any other providers
                    ocr_results = self.ocr_manager.detect_text(image, self.ocr_provider)
                
                # Convert OCR results to TextRegion format
                for result in ocr_results:
                    # Apply filtering
                    if len(result.text) < min_text_length:
                        self._log(f"   Skipping short text ({len(result.text)} chars): {result.text}")
                        continue
                    
                    if exclude_english and self._is_primarily_english(result.text):
                        self._log(f"   Skipping English text: {result.text[:50]}...")
                        continue
                    
                    if result.confidence < confidence_threshold:
                        self._log(f"   Skipping low confidence ({result.confidence:.2f}): {result.text[:30]}...")
                        continue
                    
                    # Create TextRegion
                    region = TextRegion(
                        text=result.text,
                        vertices=result.vertices if result.vertices else [
                            (result.bbox[0], result.bbox[1]),
                            (result.bbox[0] + result.bbox[2], result.bbox[1]),
                            (result.bbox[0] + result.bbox[2], result.bbox[1] + result.bbox[3]),
                            (result.bbox[0], result.bbox[1] + result.bbox[3])
                        ],
                        bounding_box=result.bbox,
                        confidence=result.confidence,
                        region_type='text_block'
                    )
                    regions.append(region)
                    self._log(f"   Found text ({result.confidence:.2f}): {result.text[:50]}...")
            
            # MERGING SECTION (applies to all providers)
            # Check if bubble detection is enabled
            if ocr_settings.get('bubble_detection_enabled', False):
                self._log("🤖 Using AI bubble detection for merging")
                regions = self._merge_with_bubble_detection(regions, image_path)
            else:
                # Traditional merging
                merge_threshold = ocr_settings.get('merge_nearby_threshold', 20)
                
                # Apply provider-specific adjustments
                if self.ocr_provider == 'azure':
                    azure_multiplier = ocr_settings.get('azure_merge_multiplier', 2.0)
                    merge_threshold = int(merge_threshold * azure_multiplier)
                    self._log(f"📋 Using Azure-adjusted merge threshold: {merge_threshold}px")
                    
                    # Pre-group Azure lines if the method exists
                    if hasattr(self, '_pregroup_azure_lines'):
                        regions = self._pregroup_azure_lines(regions, merge_threshold)
                
                elif self.ocr_provider in ['paddleocr', 'easyocr', 'doctr']:
                    # These providers often return smaller text segments
                    line_multiplier = ocr_settings.get('line_ocr_merge_multiplier', 1.5)
                    merge_threshold = int(merge_threshold * line_multiplier)
                    self._log(f"📋 Using line-based OCR adjusted threshold: {merge_threshold}px")
                
                # Apply standard merging
                regions = self._merge_nearby_regions(regions, threshold=merge_threshold)
            
            self._log(f"✅ Detected {len(regions)} text regions after merging")
            
            # Save debug images if enabled
            advanced_settings = manga_settings.get('advanced', {})
            if advanced_settings.get('debug_mode', False) or advanced_settings.get('save_intermediate', False):
                self._save_debug_image(image_path, regions)
            
            return regions
            
        except Exception as e:
            self._log(f"❌ Error detecting text: {str(e)}", "error")
            import traceback
            self._log(traceback.format_exc(), "error")
            raise

    def _validate_easyocr_languages(self, languages):
        """Validate EasyOCR language combinations"""
        # EasyOCR compatibility rules
        incompatible_sets = [
            {'ja', 'ko'},  # Japanese + Korean
            {'ja', 'zh'},  # Japanese + Chinese  
            {'ko', 'zh'}   # Korean + Chinese
        ]
        
        lang_set = set(languages)
        
        for incompatible in incompatible_sets:
            if incompatible.issubset(lang_set):
                # Conflict detected - keep first language + English
                primary_lang = languages[0] if languages else 'en'
                result = [primary_lang, 'en'] if primary_lang != 'en' else ['en']
                
                self._log(f"⚠️ EasyOCR: {' + '.join(incompatible)} not compatible", "warning")
                self._log(f"🔧 Auto-adjusted from {languages} to {result}", "info")
                return result
        
        return languages
    
    def _pregroup_azure_lines(self, lines: List[TextRegion], base_threshold: int) -> List[TextRegion]:
        """Pre-group Azure lines that are obviously part of the same text block
        This makes them more like Google's blocks before the main merge logic"""
        
        if len(lines) <= 1:
            return lines
        
        # Sort by vertical position first, then horizontal
        lines.sort(key=lambda r: (r.bounding_box[1], r.bounding_box[0]))
        
        pregrouped = []
        i = 0
        
        while i < len(lines):
            current_group = [lines[i]]
            current_bbox = list(lines[i].bounding_box)
            
            # Look ahead for lines that should obviously be grouped
            j = i + 1
            while j < len(lines):
                x1, y1, w1, h1 = current_bbox
                x2, y2, w2, h2 = lines[j].bounding_box
                
                # Calculate gaps
                vertical_gap = y2 - (y1 + h1) if y2 > y1 + h1 else 0
                
                # Check horizontal alignment
                center_x1 = x1 + w1 / 2
                center_x2 = x2 + w2 / 2
                horizontal_offset = abs(center_x1 - center_x2)
                avg_width = (w1 + w2) / 2
                
                # Group if:
                # 1. Lines are vertically adjacent (small gap)
                # 2. Lines are well-aligned horizontally (likely same bubble)
                if (vertical_gap < h1 * 0.5 and  # Less than half line height gap
                    horizontal_offset < avg_width * 0.5):  # Well centered
                    
                    # Add to group
                    current_group.append(lines[j])
                    
                    # Update bounding box to include new line
                    min_x = min(x1, x2)
                    min_y = min(y1, y2)
                    max_x = max(x1 + w1, x2 + w2)
                    max_y = max(y1 + h1, y2 + h2)
                    current_bbox = [min_x, min_y, max_x - min_x, max_y - min_y]
                    
                    j += 1
                else:
                    break
            
            # Create merged region from group
            if len(current_group) > 1:
                merged_text = " ".join([line.text for line in current_group])
                all_vertices = []
                for line in current_group:
                    all_vertices.extend(line.vertices)
                
                merged_region = TextRegion(
                    text=merged_text,
                    vertices=all_vertices,
                    bounding_box=tuple(current_bbox),
                    confidence=0.95,
                    region_type='pregrouped_lines'
                )
                pregrouped.append(merged_region)
                
                self._log(f"   Pre-grouped {len(current_group)} Azure lines into block")
            else:
                # Single line, keep as is
                pregrouped.append(lines[i])
            
            i = j if j > i + 1 else i + 1
        
        self._log(f"   Azure pre-grouping: {len(lines)} lines → {len(pregrouped)} blocks")
        return pregrouped

    def _detect_text_azure(self, image_data: bytes, ocr_settings: dict) -> List[TextRegion]:
        """Detect text using Azure Computer Vision"""
        import io
        from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
        
        stream = io.BytesIO(image_data)
        
        # Use Read API for better manga text detection
        read_result = self.vision_client.read_in_stream(
            stream,
            raw=True,
            language='ja'  # or from ocr_settings
        )
        
        # Get operation ID from headers
        operation_location = read_result.headers["Operation-Location"]
        operation_id = operation_location.split("/")[-1]
        
        # Wait for completion
        import time
        while True:
            result = self.vision_client.get_read_result(operation_id)
            if result.status not in [OperationStatusCodes.running, OperationStatusCodes.not_started]:
                break
            time.sleep(0.5)
        
        regions = []
        confidence_threshold = ocr_settings.get('confidence_threshold', 0.8)
        
        if result.status == OperationStatusCodes.succeeded:
            for page in result.analyze_result.read_results:
                for line in page.lines:
                    # Azure returns bounding box as 8 coordinates
                    bbox = line.bounding_box
                    vertices = [
                        (bbox[0], bbox[1]),
                        (bbox[2], bbox[3]),
                        (bbox[4], bbox[5]),
                        (bbox[6], bbox[7])
                    ]
                    
                    xs = [v[0] for v in vertices]
                    ys = [v[1] for v in vertices]
                    x_min, x_max = min(xs), max(xs)
                    y_min, y_max = min(ys), max(ys)
                    
                    # Azure doesn't provide per-line confidence in Read API
                    confidence = 0.95  # Default high confidence
                    
                    if confidence >= confidence_threshold:
                        region = TextRegion(
                            text=line.text,
                            vertices=vertices,
                            bounding_box=(x_min, y_min, x_max - x_min, y_max - y_min),
                            confidence=confidence,
                            region_type='text_line'
                        )
                        regions.append(region)
        
        return regions

    def _preprocess_image(self, image_path: str, preprocessing_settings: Dict) -> bytes:
        """Preprocess image for better OCR results"""
        try:
            # Open image with PIL
            pil_image = Image.open(image_path)
            
            # Convert to RGB if necessary
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # Auto-detect quality issues if enabled
            if preprocessing_settings.get('auto_detect_quality', True):
                needs_enhancement = self._detect_quality_issues(pil_image, preprocessing_settings)
                if needs_enhancement:
                    self._log("   Auto-detected quality issues - applying enhancements")
            else:
                needs_enhancement = True
            
            if needs_enhancement:
                # Apply contrast enhancement
                contrast_threshold = preprocessing_settings.get('contrast_threshold', 0.4)
                enhancer = ImageEnhance.Contrast(pil_image)
                pil_image = enhancer.enhance(1 + contrast_threshold)
                
                # Apply sharpness enhancement
                sharpness_threshold = preprocessing_settings.get('sharpness_threshold', 0.3)
                enhancer = ImageEnhance.Sharpness(pil_image)
                pil_image = enhancer.enhance(1 + sharpness_threshold)
                
                # Apply general enhancement strength
                enhancement_strength = preprocessing_settings.get('enhancement_strength', 1.5)
                if enhancement_strength != 1.0:
                    # Brightness adjustment
                    enhancer = ImageEnhance.Brightness(pil_image)
                    pil_image = enhancer.enhance(enhancement_strength)
            
            # Resize if too large
            max_dimension = preprocessing_settings.get('max_image_dimension', 2000)
            if pil_image.width > max_dimension or pil_image.height > max_dimension:
                ratio = min(max_dimension / pil_image.width, max_dimension / pil_image.height)
                new_size = (int(pil_image.width * ratio), int(pil_image.height * ratio))
                pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)
                self._log(f"   Resized image to {new_size[0]}x{new_size[1]}")
            
            # Convert back to bytes
            from io import BytesIO
            buffered = BytesIO()
            pil_image.save(buffered, format="PNG", optimize=True)
            return buffered.getvalue()
            
        except Exception as e:
            self._log(f"⚠️ Preprocessing failed: {str(e)}, using original image", "warning")
            with open(image_path, 'rb') as f:
                return f.read()

    def _detect_quality_issues(self, image: Image.Image, settings: Dict) -> bool:
        """Auto-detect if image needs quality enhancement"""
        # Convert to grayscale for analysis
        gray = image.convert('L')
        
        # Get histogram
        hist = gray.histogram()
        
        # Calculate contrast (simplified)
        pixels = sum(hist)
        mean = sum(i * hist[i] for i in range(256)) / pixels
        variance = sum(hist[i] * (i - mean) ** 2 for i in range(256)) / pixels
        std_dev = variance ** 0.5
        
        # Low contrast if std deviation is low
        contrast_threshold = settings.get('contrast_threshold', 0.4) * 100
        if std_dev < contrast_threshold:
            self._log("   Low contrast detected")
            return True
        
        # Check for blur using Laplacian variance
        import numpy as np
        gray_array = np.array(gray)
        laplacian = cv2.Laplacian(gray_array, cv2.CV_64F)
        variance = laplacian.var()
        
        sharpness_threshold = settings.get('sharpness_threshold', 0.3) * 100
        if variance < sharpness_threshold:
            self._log("   Blur detected")
            return True
        
        return False

    def _save_debug_image(self, image_path: str, regions: List[TextRegion]):
        """Save debug image with detected regions highlighted"""
        # Check if debug mode is enabled
        if not self.manga_settings.get('advanced', {}).get('debug_mode', False):
            return
        
        try:
            import cv2
            import numpy as np
            from PIL import Image as PILImage
            
            # Handle Unicode paths
            try:
                img = cv2.imread(image_path)
                if img is None:
                    # Fallback to PIL for Unicode paths
                    pil_image = PILImage.open(image_path)
                    img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            except Exception as e:
                self._log(f"   Failed to load image for debug: {str(e)}", "warning")
                return
            
            # Create debug directory if save_intermediate is enabled
            if self.manga_settings.get('advanced', {}).get('save_intermediate', False):
                debug_dir = os.path.join(os.path.dirname(image_path), 'debug')
                os.makedirs(debug_dir, exist_ok=True)
                base_name = os.path.splitext(os.path.basename(image_path))[0]
            
            # Draw rectangles around detected text regions
            overlay = img.copy()
            
            # Calculate statistics
            total_chars = sum(len(r.text) for r in regions)
            avg_confidence = np.mean([r.confidence for r in regions]) if regions else 0
            
            for i, region in enumerate(regions):
                x, y, w, h = region.bounding_box
                
                # Color based on confidence
                if region.confidence > 0.95:
                    color = (0, 255, 0)  # Green - high confidence
                elif region.confidence > 0.8:
                    color = (0, 165, 255)  # Orange - medium confidence
                else:
                    color = (0, 0, 255)  # Red - low confidence
                
                # Draw rectangle
                cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)
                
                # Add region info
                info_text = f"#{i} ({region.confidence:.2f})"
                cv2.putText(overlay, info_text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, color, 1, cv2.LINE_AA)
                
                # Add character count
                char_count = len(region.text.strip())
                cv2.putText(overlay, f"{char_count} chars", (x, y + h + 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
                
                # Add detected text preview if in verbose debug mode
                if self.manga_settings.get('advanced', {}).get('save_intermediate', False):
                    text_preview = region.text[:20] + "..." if len(region.text) > 20 else region.text
                    cv2.putText(overlay, text_preview, (x, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.4, color, 1, cv2.LINE_AA)
            
            # Add overall statistics to the image
            stats_bg = overlay.copy()
            cv2.rectangle(stats_bg, (10, 10), (300, 90), (0, 0, 0), -1)
            cv2.addWeighted(stats_bg, 0.7, overlay, 0.3, 0, overlay)
            
            stats_text = [
                f"Regions: {len(regions)}",
                f"Total chars: {total_chars}",
                f"Avg confidence: {avg_confidence:.2f}"
            ]
            
            for i, text in enumerate(stats_text):
                cv2.putText(overlay, text, (20, 35 + i*20), cv2.FONT_HERSHEY_SIMPLEX,
                           0.5, (255, 255, 255), 1, cv2.LINE_AA)
            
            # Save main debug image
            if self.manga_settings.get('advanced', {}).get('save_intermediate', False):
                debug_path = os.path.join(debug_dir, f"{base_name}_debug_regions.png")
            else:
                debug_path = image_path.replace('.', '_debug.')
            
            cv2.imwrite(debug_path, overlay)
            self._log(f"   📸 Saved debug image: {debug_path}")
            
            # Save text mask
            mask = self.create_text_mask(img, regions)
            mask_debug_path = debug_path.replace('_debug', '_mask')
            cv2.imwrite(mask_debug_path, mask)
            mask_percentage = ((mask > 0).sum() / mask.size) * 100
            self._log(f"   🎭 Saved mask image: {mask_debug_path}", "info")
            self._log(f"   📊 Mask coverage: {mask_percentage:.1f}% of image", "info")
                        
            # If save_intermediate is enabled, save additional debug images
            if self.manga_settings.get('advanced', {}).get('save_intermediate', False):
                # Save confidence heatmap
                heatmap = self._create_confidence_heatmap(img, regions)
                heatmap_path = os.path.join(debug_dir, f"{base_name}_confidence_heatmap.png")
                cv2.imwrite(heatmap_path, heatmap)
                self._log(f"   🌡️ Saved confidence heatmap: {heatmap_path}")
                
                # Save polygon visualization with safe text areas
                if any(hasattr(r, 'vertices') and r.vertices for r in regions):
                    polygon_img = img.copy()
                    for region in regions:
                        if hasattr(region, 'vertices') and region.vertices:
                            # Draw polygon
                            pts = np.array(region.vertices, np.int32)
                            pts = pts.reshape((-1, 1, 2))
                            
                            # Fill with transparency
                            overlay_poly = polygon_img.copy()
                            cv2.fillPoly(overlay_poly, [pts], (0, 255, 255))
                            cv2.addWeighted(overlay_poly, 0.2, polygon_img, 0.8, 0, polygon_img)
                            
                            # Draw outline
                            cv2.polylines(polygon_img, [pts], True, (255, 0, 0), 2)
                            
                            # Draw safe text area
                            try:
                                safe_x, safe_y, safe_w, safe_h = self.get_safe_text_area(region)
                                cv2.rectangle(polygon_img, (safe_x, safe_y), 
                                            (safe_x + safe_w, safe_y + safe_h), 
                                            (0, 255, 0), 1)
                            except:
                                pass  # Skip if get_safe_text_area fails
                    
                    polygon_path = os.path.join(debug_dir, f"{base_name}_polygons.png")
                    cv2.imwrite(polygon_path, polygon_img)
                    self._log(f"   🔷 Saved polygon visualization: {polygon_path}")
                
                # Save individual region crops with more info
                regions_dir = os.path.join(debug_dir, 'regions')
                os.makedirs(regions_dir, exist_ok=True)
                
                for i, region in enumerate(regions[:10]):  # Limit to first 10 regions
                    x, y, w, h = region.bounding_box
                    # Add padding
                    pad = 10
                    x1 = max(0, x - pad)
                    y1 = max(0, y - pad)
                    x2 = min(img.shape[1], x + w + pad)
                    y2 = min(img.shape[0], y + h + pad)
                    
                    region_crop = img[y1:y2, x1:x2].copy()
                    
                    # Draw bounding box on crop
                    cv2.rectangle(region_crop, (pad, pad), 
                                (pad + w, pad + h), (0, 255, 0), 2)
                    
                    # Add text info on the crop
                    info = f"Conf: {region.confidence:.2f} | Chars: {len(region.text)}"
                    cv2.putText(region_crop, info, (5, 15), cv2.FONT_HERSHEY_SIMPLEX,
                               0.4, (255, 255, 255), 1, cv2.LINE_AA)
                    
                    # Save with meaningful filename
                    safe_text = region.text[:20].replace('/', '_').replace('\\', '_').strip()
                    region_path = os.path.join(regions_dir, f"region_{i:03d}_{safe_text}.png")
                    cv2.imwrite(region_path, region_crop)
                
                self._log(f"   📁 Saved individual region crops to: {regions_dir}")
            
        except Exception as e:
            self._log(f"   ❌ Failed to save debug image: {str(e)}", "warning")
            if self.manga_settings.get('advanced', {}).get('debug_mode', False):
                # If debug mode is on, log the full traceback
                import traceback
                self._log(traceback.format_exc(), "warning")

    def _create_confidence_heatmap(self, img, regions):
        """Create a heatmap showing OCR confidence levels"""
        heatmap = np.zeros_like(img[:, :, 0], dtype=np.float32)
        
        for region in regions:
            x, y, w, h = region.bounding_box
            confidence = region.confidence
            heatmap[y:y+h, x:x+w] = confidence
        
        # Convert to color heatmap
        heatmap_normalized = (heatmap * 255).astype(np.uint8)
        heatmap_colored = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)
        
        # Blend with original image
        result = cv2.addWeighted(img, 0.7, heatmap_colored, 0.3, 0)
        return result

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

            # Get the prompt from prompt_profiles dictionary
            system_prompt = ''
            if hasattr(self.main_gui, 'prompt_profiles') and profile_name in self.main_gui.prompt_profiles:
                system_prompt = self.main_gui.prompt_profiles[profile_name]
                self._log(f"📋 Using profile: {profile_name}")
            else:
                self._log(f"⚠️ Profile '{profile_name}' not found in prompt_profiles", "warning")

            self._log(f"📝 System prompt: {system_prompt[:100]}..." if system_prompt else "📝 No system prompt configured")

            if system_prompt:
                messages = [{"role": "system", "content": system_prompt}]
            else:
                messages = []
            
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
            
            # Add full image context if available AND visual context is enabled
            if image_path and self.visual_context_enabled:
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
            elif image_path and not self.visual_context_enabled:
                # Visual context disabled - text-only mode
                self._log(f"📝 Text-only mode (visual context disabled)")
                messages.append({"role": "user", "content": text})
            else:
                # No image path provided - text-only translation
                messages.append({"role": "user", "content": text})
            
            # Check input token limit
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
                            # Only count image tokens if visual context is enabled
                            if self.visual_context_enabled:
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
            if '\\\\' in translated or '\\n' in translated or "\\'" in translated or '\\"' in translated:
                self._log(f"⚠️ Detected escaped content, unescaping...", "warning")
                try:
                    # DON'T use unicode_escape for Korean text - it corrupts it
                    # Instead, just replace the escape sequences manually
                    before = translated
                    
                    # Handle quotes and apostrophes
                    translated = translated.replace("\\'", "'")  # Escaped apostrophe
                    translated = translated.replace('\\"', '"')  # Escaped quote
                    translated = translated.replace("\\`", "`")  # Escaped backtick
                    translated = translated.replace("\\u2019", "'")  # Unicode right single quote
                    translated = translated.replace("\\u2018", "'")  # Unicode left single quote
                    translated = translated.replace("\\u201c", '"')  # Unicode left double quote
                    translated = translated.replace("\\u201d", '"')  # Unicode right double quote
                    
                    # Handle newlines and other escapes
                    translated = translated.replace('\\n', '\n')
                    translated = translated.replace('\\\\', '\\')
                    translated = translated.replace('\\/', '/')
                    translated = translated.replace('\\t', '\t')
                    translated = translated.replace('\\r', '\r')
                    
                    # Clean up smart quotes using Unicode escape codes
                    translated = translated.replace('\u2018', "'")  # Left single quotation mark
                    translated = translated.replace('\u2019', "'")  # Right single quotation mark
                    translated = translated.replace('\u201c', '"')  # Left double quotation mark
                    translated = translated.replace('\u201d', '"')  # Right double quotation mark
                    
                    self._log(f"📦 Unescaped safely: '{before[:50]}...' -> '{translated[:50]}...'")
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
            
            # Ensure visual_context_enabled exists (temporary fix)
            if not hasattr(self, 'visual_context_enabled'):
                self.visual_context_enabled = self.main_gui.config.get('manga_visual_context_enabled', True)
            
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
            
            # Create the full context message text
            context_text = "\n".join(text_list)
            
            # Log text content info
            total_chars = sum(len(region.text) for region in regions)
            self._log(f"📝 Text content: {len(regions)} regions, {total_chars} total characters")
            
            # Process image if visual context is enabled
            if self.visual_context_enabled:
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
                    
                    # Create message with both text and image
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
                    self._log(f"   Falling back to text-only translation", "warning")
                    
                    # Fall back to text-only translation
                    messages.append({"role": "user", "content": context_text})
            else:
                # Visual context disabled - send text only
                self._log(f"📝 Text-only mode (visual context disabled for non-vision models)")
                messages.append({"role": "user", "content": context_text})
            
            # CHECK 5: Before API call
            if self._check_stop():
                self._log("⏹️ Translation stopped before API call", "warning")
                return {}
            
            # Check input token limit
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
                            # Only count image tokens if visual context is enabled
                            if self.visual_context_enabled:
                                image_tokens += 258

            estimated_tokens = text_tokens + image_tokens

            # Check token limit only if it's enabled
            if self.input_token_limit is None:
                self._log(f"📊 Token estimate - Text: {text_tokens}, Images: {image_tokens} (Total: {estimated_tokens} / unlimited)")
            else:
                self._log(f"📊 Token estimate - Text: {text_tokens}, Images: {image_tokens} (Total: {estimated_tokens} / {self.input_token_limit})")
                
                if estimated_tokens > self.input_token_limit:
                    self._log(f"⚠️ Token limit exceeded, trimming context", "warning")
                    # Keep system prompt and current message only
                    messages = [messages[0], messages[-1]]  
                    # Recalculate tokens
                    text_tokens = len(messages[0]["content"]) // 4
                    if isinstance(messages[-1]["content"], str):
                        text_tokens += len(messages[-1]["content"]) // 4
                    else:
                        for content_part in messages[-1]["content"]:
                            if content_part.get("type") == "text":
                                text_tokens += len(content_part.get("text", "")) // 4
                    estimated_tokens = text_tokens + image_tokens
                    self._log(f"📊 Trimmed token estimate: {estimated_tokens}")
            
            # [Rest of the method remains the same - API call, response handling, etc.]
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
            if '\\\\' in response_text or '\\n' in response_text or "\\'" in response_text or '\\"' in response_text:
                self._log(f"⚠️ Detected escaped content, unescaping...", "warning")
                try:
                    # DON'T use unicode_escape for Korean text - it corrupts it
                    # Instead, just replace the escape sequences manually
                    
                    # Handle quotes and apostrophes
                    response_text = response_text.replace("\\'", "'")  # Escaped apostrophe
                    response_text = response_text.replace('\\"', '"')  # Escaped quote
                    response_text = response_text.replace("\\`", "`")  # Escaped backtick
                    response_text = response_text.replace("\\u2019", "'")  # Unicode right single quote
                    response_text = response_text.replace("\\u2018", "'")  # Unicode left single quote
                    response_text = response_text.replace("\\u201c", '"')  # Unicode left double quote
                    response_text = response_text.replace("\\u201d", '"')  # Unicode right double quote
                    
                    # Handle newlines and other escapes
                    response_text = response_text.replace('\\n', '\n')
                    response_text = response_text.replace('\\\\', '\\')
                    response_text = response_text.replace('\\/', '/')
                    response_text = response_text.replace('\\t', '\t')
                    response_text = response_text.replace('\\r', '\r')
                    
                    # Clean up smart quotes using Unicode escape codes
                    response_text = response_text.replace('\u2018', "'")  # Left single quotation mark
                    response_text = response_text.replace('\u2019', "'")  # Right single quotation mark
                    response_text = response_text.replace('\u201c', '"')  # Left double quotation mark
                    response_text = response_text.replace('\u201d', '"')  # Right double quotation mark
                    
                    self._log(f"📦 Unescaped content safely")
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
                
                # Fix encoding issues first
                response_text = self._fix_encoding_issues(response_text)
                
                # Try parsing JSON again after fixing encoding
                try:
                    translations = json.loads(response_text)
                    self._log(f"✅ Successfully parsed {len(translations)} translations after encoding fix", "success")
                except json.JSONDecodeError:
                    # If it still fails, continue with the existing fallback
                    # Fallback: try to fix common JSON issues
                    try:
                        self._log("🔧 Attempting to fix JSON by escaping control characters...", "info")
                        
                        # Method 1: Use json.dumps to properly escape the string, then parse it
                        import re
                        
                        # First, try to extract key-value pairs manually
                        translations = {}
                        
                        # Pattern to match "key": "value" pairs, handling quotes and newlines
                        pattern = r'"([^"]+)"\s*:\s*"([^"]*(?:\\.[^"]*)*)"'
                        matches = re.findall(pattern, response_text)
                        
                        for key, value in matches:
                            # Unescape the value
                            try:
                                # Replace literal \n with actual newlines
                                value = value.replace('\\n', '\n')
                                value = value.replace('\\"', '"')
                                value = value.replace('\\\\', '\\')
                                translations[key] = value
                                self._log(f"  ✅ Extracted: '{key[:30]}...' → '{value[:30]}...'", "info")
                            except Exception as ex:
                                self._log(f"  ⚠️ Failed to process pair: {ex}", "warning")
                        
                        if translations:
                            self._log(f"✅ Recovered {len(translations)} translations using regex", "success")
                        else:
                            # Method 2: Try to clean and re-parse
                            cleaned = response_text
                            # Remove any actual newlines within string values
                            cleaned = re.sub(r'(?<="[^"]*)\n(?=[^"]*")', '\\n', cleaned)
                            cleaned = re.sub(r'(?<="[^"]*)\r(?=[^"]*")', '\\r', cleaned)
                            cleaned = re.sub(r'(?<="[^"]*)\t(?=[^"]*")', '\\t', cleaned)
                            
                            translations = json.loads(cleaned)
                            self._log(f"✅ Successfully parsed after cleaning: {len(translations)} translations", "success")
                            
                    except Exception as e2:
                        self._log(f"❌ Failed to recover JSON: {str(e2)}", "error")
                        self._log(f"   Returning empty translations", "error")
                        return {}
            
            # Map translations back to regions
            result = {}
            all_originals = []
            all_translations = []
            
            # Extract translation values in order
            translation_values = list(translations.values()) if translations else []
            
            # Use position-based mapping if counts match
            if len(translation_values) >= len(regions) - 1:  # Allow 1 missing
                self._log(f"📊 Using position-based mapping ({len(translation_values)} translations for {len(regions)} regions)")
                
                for i, region in enumerate(regions):
                    # CHECK 9: During mapping
                    if i % 10 == 0 and self._check_stop():
                        self._log(f"⏹️ Translation stopped during mapping (processed {i}/{len(regions)} regions)", "warning")
                        return result
                    
                    # Use position-based translation
                    if i < len(translation_values):
                        translated = translation_values[i]
                    else:
                        translated = region.text  # Fallback
                    
                    # Apply glossary
                    if translated != region.text and hasattr(self.main_gui, 'manual_glossary') and self.main_gui.manual_glossary:
                        for entry in self.main_gui.manual_glossary:
                            if 'source' in entry and 'target' in entry:
                                if entry['source'] in translated:
                                    translated = translated.replace(entry['source'], entry['target'])
                    
                    result[region.text] = translated
                    region.translated_text = translated
                    self._log(f"  ✅ Applied translation for: '{region.text[:40]}...'", "info")
                    
                    if translated != region.text:
                        all_originals.append(f"[{i+1}] {region.text}")
                        all_translations.append(f"[{i+1}] {translated}")
            else:
                # Fallback to key-based matching
                for i, region in enumerate(regions):
                    if i % 10 == 0 and self._check_stop():
                        self._log(f"⏹️ Translation stopped during mapping (processed {i}/{len(regions)} regions)", "warning")
                        return result
                    
                    key = f"[{i}] {region.text}"
                    
                    if key in translations:
                        translated = translations[key]
                    elif region.text in translations:
                        translated = translations[region.text]
                    else:
                        translated = region.text
                    
                    # Apply glossary
                    if translated != region.text and hasattr(self.main_gui, 'manual_glossary') and self.main_gui.manual_glossary:
                        for entry in self.main_gui.manual_glossary:
                            if 'source' in entry and 'target' in entry:
                                if entry['source'] in translated:
                                    translated = translated.replace(entry['source'], entry['target'])
                    
                    result[region.text] = translated
                    region.translated_text = translated
                    
                    if translated != region.text:
                        self._log(f"  ✅ Mapped translation: '{region.text[:30]}...' → '{translated[:30]}...'")
                        all_originals.append(f"[{i+1}] {region.text}")
                        all_translations.append(f"[{i+1}] {translated}")
            
            # Save as ONE combined history entry for the entire page
            if self.history_manager and self.contextual_enabled and all_originals:
                try:
                    # Combine all text from this page into a single exchange
                    combined_original = "\n".join(all_originals)
                    combined_translation = "\n".join(all_translations)
                    
                    # Save as a single history entry
                    self.history_manager.append_to_history(
                        user_content=combined_original,
                        assistant_content=combined_translation,
                        hist_limit=self.translation_history_limit,
                        reset_on_limit=not self.rolling_history_enabled,
                        rolling_window=self.rolling_history_enabled
                    )
                    
                    self._log(f"📚 Saved {len(all_originals)} translations as 1 combined history entry", "success")
                    
                    # Check current history status
                    current_history = self.history_manager.load_history()
                    current_exchanges = len(current_history) // 2
                    self._log(f"📚 History now contains {current_exchanges} exchanges (pages)", "info")
                    
                    if self.history_manager.will_reset_on_next_append(
                        self.translation_history_limit, 
                        self.rolling_history_enabled
                    ):
                        mode = "roll over" if self.rolling_history_enabled else "reset"
                        self._log(f"📚 History will {mode} on next page (at limit: {self.translation_history_limit})", "info")
                    
                except Exception as e:
                    self._log(f"⚠️ Failed to save page to history: {str(e)}", "warning")
            
            return result
            
        except Exception as e:
            
            # CHECK 10: In exception handler
            if self._check_stop():
                self._log("⏹️ Translation stopped due to user request", "warning")
                return {}  
                
            self._log(f"❌ Full page context translation error: {str(e)}", "error")
            self._log(traceback.format_exc(), "error")
            return {}

    def _fix_encoding_issues(self, text: str) -> str:
        """Fix common encoding issues in text, especially for Korean"""
        if not text:
            return text
        
        # Check for mojibake indicators (UTF-8 misinterpreted as Latin-1)
        mojibake_indicators = ['ë', 'ì', 'ê°', 'ã', 'Ã', 'â', 'ä', 'ð', 'í', 'ë­', 'ì´']
        
        if any(indicator in text for indicator in mojibake_indicators):
            self._log("🔧 Detected mojibake encoding issue, attempting fixes...", "debug")
            
            # Try multiple encoding fixes
            encodings_to_try = [
                ('latin-1', 'utf-8'),
                ('windows-1252', 'utf-8'),
                ('iso-8859-1', 'utf-8'),
                ('cp1252', 'utf-8')
            ]
            
            for from_enc, to_enc in encodings_to_try:
                try:
                    fixed = text.encode(from_enc, errors='ignore').decode(to_enc, errors='ignore')
                    
                    # Check if the fix actually improved things
                    # Should have Korean characters (Hangul range) or be cleaner
                    if any('\uAC00' <= c <= '\uD7AF' for c in fixed) or fixed.count('�') < text.count('�'):
                        self._log(f"✅ Fixed encoding using {from_enc} -> {to_enc}", "debug")
                        return fixed
                except:
                    continue
        
        # Clean up any remaining control characters
        import re
        text = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]', '', text)
        
        return text
                
    def create_text_mask(self, image: np.ndarray, regions: List[TextRegion]) -> np.ndarray:
        """Create mask with comprehensive per-text-type dilation settings"""
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        regions_masked = 0
        regions_skipped = 0
        
        self._log(f"🎭 Creating text mask for {len(regions)} regions", "info")
        
        # Get manga settings
        manga_settings = self.main_gui.config.get('manga_settings', {})
        
        # Get dilation settings
        base_dilation_size = manga_settings.get('mask_dilation', 15)
        
        # Check if using uniform iterations for all text types
        use_all_iterations = manga_settings.get('use_all_iterations', False)
        
        if use_all_iterations:
            # Use the same iteration count for all text types
            all_iterations = manga_settings.get('all_iterations', 2)
            text_bubble_iterations = all_iterations
            empty_bubble_iterations = all_iterations
            free_text_iterations = all_iterations
            self._log(f"📏 Using uniform iterations: {all_iterations} for all text types", "info")
        else:
            # Use individual iteration settings
            text_bubble_iterations = manga_settings.get('text_bubble_dilation_iterations',
                                                       manga_settings.get('bubble_dilation_iterations', 2))
            empty_bubble_iterations = manga_settings.get('empty_bubble_dilation_iterations', 3)
            free_text_iterations = manga_settings.get('free_text_dilation_iterations', 0)
            self._log(f"📏 Using individual iterations - Text bubbles: {text_bubble_iterations}, "
                     f"Empty bubbles: {empty_bubble_iterations}, Free text: {free_text_iterations}", "info")
        
        # Create separate masks for different text types
        text_bubble_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        empty_bubble_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        free_text_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        text_bubble_count = 0
        empty_bubble_count = 0
        free_text_count = 0
        
        for i, region in enumerate(regions):
            # CHECK: Should this region be inpainted?
            if not getattr(region, 'should_inpaint', True):
                # Skip this region - it shouldn't be inpainted
                regions_skipped += 1
                self._log(f"   Region {i+1}: SKIPPED (filtered by settings)", "debug")
                continue
            
            regions_masked += 1
            
            # Determine text type
            text_type = 'free_text'  # default
            
            # Check if region has bubble_type attribute (from bubble detection)
            if hasattr(region, 'bubble_type'):
                # RT-DETR classifications
                if region.bubble_type == 'empty_bubble':
                    text_type = 'empty_bubble'
                elif region.bubble_type == 'text_bubble':
                    text_type = 'text_bubble'
                else:  # 'free_text' or others
                    text_type = 'free_text'
            else:
                # Fallback: use simple heuristics if no bubble detection
                x, y, w, h = region.bounding_box
                x, y, w, h = int(x), int(y), int(w), int(h)
                aspect_ratio = w / h if h > 0 else 1
                
                # Check if region has text
                has_text = hasattr(region, 'text') and region.text and len(region.text.strip()) > 0
                
                # Heuristic: bubbles tend to be more square-ish or tall
                # Free text tends to be wide and short
                if aspect_ratio < 2.5 and w > 50 and h > 50:
                    if has_text:
                        text_type = 'text_bubble'
                    else:
                        # Could be empty bubble if it's round/oval shaped
                        text_type = 'empty_bubble'
                else:
                    text_type = 'free_text'
            
            # Select appropriate mask and increment counter
            if text_type == 'text_bubble':
                target_mask = text_bubble_mask
                text_bubble_count += 1
                mask_type = "TEXT BUBBLE"
            elif text_type == 'empty_bubble':
                target_mask = empty_bubble_mask
                empty_bubble_count += 1
                mask_type = "EMPTY BUBBLE"
            else:
                target_mask = free_text_mask
                free_text_count += 1
                mask_type = "FREE TEXT"
            
            # Check if this is a merged region with original regions
            if hasattr(region, 'original_regions') and region.original_regions:
                # Use original regions for precise masking
                self._log(f"   Region {i+1} ({mask_type}): Using {len(region.original_regions)} original regions", "debug")
                
                for orig_region in region.original_regions:
                    if hasattr(orig_region, 'vertices') and orig_region.vertices:
                        pts = np.array(orig_region.vertices, np.int32)
                        pts = pts.reshape((-1, 1, 2))
                        cv2.fillPoly(target_mask, [pts], 255)
                    else:
                        x, y, w, h = orig_region.bounding_box
                        x, y, w, h = int(x), int(y), int(w), int(h)
                        cv2.rectangle(target_mask, (x, y), (x + w, y + h), 255, -1)
            else:
                # Normal region
                if hasattr(region, 'vertices') and region.vertices and len(region.vertices) <= 8:
                    pts = np.array(region.vertices, np.int32)
                    pts = pts.reshape((-1, 1, 2))
                    cv2.fillPoly(target_mask, [pts], 255)
                    self._log(f"   Region {i+1} ({mask_type}): Using polygon", "debug")
                else:
                    x, y, w, h = region.bounding_box
                    x, y, w, h = int(x), int(y), int(w), int(h)
                    cv2.rectangle(target_mask, (x, y), (x + w, y + h), 255, -1)
                    self._log(f"   Region {i+1} ({mask_type}): Using bounding box", "debug")
        
        self._log(f"📊 Mask breakdown: {text_bubble_count} text bubbles, {empty_bubble_count} empty bubbles, "
                 f"{free_text_count} free text regions, {regions_skipped} skipped", "info")
        
        # Apply different dilation settings to each mask type
        if base_dilation_size > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (base_dilation_size, base_dilation_size))
            
            # Apply dilation to text bubble mask
            if text_bubble_count > 0 and text_bubble_iterations > 0:
                self._log(f"📏 Applying text bubble dilation: {base_dilation_size}px, {text_bubble_iterations} iterations", "info")
                text_bubble_mask = cv2.dilate(text_bubble_mask, kernel, iterations=text_bubble_iterations)
            
            # Apply dilation to empty bubble mask
            if empty_bubble_count > 0 and empty_bubble_iterations > 0:
                self._log(f"📏 Applying empty bubble dilation: {base_dilation_size}px, {empty_bubble_iterations} iterations", "info")
                empty_bubble_mask = cv2.dilate(empty_bubble_mask, kernel, iterations=empty_bubble_iterations)
            
            # Apply dilation to free text mask
            if free_text_count > 0 and free_text_iterations > 0:
                self._log(f"📏 Applying free text dilation: {base_dilation_size}px, {free_text_iterations} iterations", "info")
                free_text_mask = cv2.dilate(free_text_mask, kernel, iterations=free_text_iterations)
            elif free_text_count > 0 and free_text_iterations == 0:
                self._log(f"📏 No dilation for free text (iterations=0, perfect for B&W panels)", "info")
        
        # Combine all masks
        mask = cv2.bitwise_or(text_bubble_mask, empty_bubble_mask)
        mask = cv2.bitwise_or(mask, free_text_mask)
        
        coverage_percent = (np.sum(mask > 0) / mask.size) * 100
        self._log(f"📊 Final mask coverage: {coverage_percent:.1f}% of image", "info")
        
        return mask
    
    def _initialize_local_inpainter(self):
        """Initialize local inpainting if configured"""
        try:
            from local_inpainter import LocalInpainter, HybridInpainter, AnimeMangaInpaintModel
            
            inpaint_method = self.manga_settings.get('inpainting', {}).get('method', 'cloud')
            
            if inpaint_method == 'local':
                # Get current settings
                local_method = self.manga_settings.get('inpainting', {}).get('local_method', 'anime')
                model_path = self.manga_settings.get('inpainting', {}).get(f'{local_method}_model_path', '')
                
                # Check if we need to reinitialize due to changes
                need_reload = False
                
                # Initialize tracking attributes if they don't exist
                if not hasattr(self, '_last_local_method'):
                    self._last_local_method = None
                    self._last_local_model_path = None
                
                # Check for changes
                if self._last_local_method != local_method:
                    self._log(f"🔄 Local method changed from {self._last_local_method} to {local_method}", "info")
                    need_reload = True
                
                if self._last_local_model_path != model_path:
                    self._log(f"🔄 Model path changed", "info")
                    if self._last_local_model_path:
                        self._log(f"   Old: {os.path.basename(self._last_local_model_path)}", "debug")
                    if model_path:
                        self._log(f"   New: {os.path.basename(model_path)}", "debug")
                    need_reload = True
                
                # Store current settings
                self._last_local_method = local_method
                self._last_local_model_path = model_path
                
                # Initialize inpainter if needed
                if self.local_inpainter is None:
                    self.local_inpainter = LocalInpainter()
                    need_reload = True  # First time, definitely need to load
                
                # If no model path or doesn't exist, try to find or download one
                if not model_path or not os.path.exists(model_path):
                    self._log(f"⚠️ Model path not found: {model_path}", "warning")
                    
                    # Try to download JIT model automatically
                    self._log("📥 Attempting to download JIT model...", "info")
                    downloaded_path = self.local_inpainter.download_jit_model(local_method)
                    if downloaded_path:
                        model_path = downloaded_path
                        self._log(f"✅ Downloaded JIT model to: {model_path}")
                        need_reload = True  # Downloaded new model
                
                # Load or reload the model if needed
                if model_path and os.path.exists(model_path):
                    if need_reload or not self.local_inpainter.model_loaded:
                        self._log(f"📥 Loading {local_method} model...", "info")
                        if self.local_inpainter.load_model(local_method, model_path, force_reload=need_reload):
                            self._log(f"✅ Local inpainter loaded with {local_method.upper()}")
                        else:
                            self._log(f"⚠️ Failed to load model, but inpainter is ready", "warning")
                    else:
                        self._log(f"✅ Using already loaded {local_method.upper()} model", "info")
                else:
                    self._log(f"⚠️ No model available, but inpainter is initialized", "warning")
                
                # Always return True so local_inpainter exists
                return True
            
            elif inpaint_method == 'hybrid':
                # Track hybrid settings changes
                if not hasattr(self, '_last_hybrid_config'):
                    self._last_hybrid_config = None
                
                current_hybrid_config = self.manga_settings.get('inpainting', {}).get('hybrid_methods', [])
                
                # Check if hybrid config changed
                need_reload = self._last_hybrid_config != current_hybrid_config
                if need_reload:
                    self._log("🔄 Hybrid configuration changed, reloading...", "info")
                    self.hybrid_inpainter = None  # Clear old instance
                
                self._last_hybrid_config = current_hybrid_config.copy() if current_hybrid_config else []
                
                if self.hybrid_inpainter is None:
                    self.hybrid_inpainter = HybridInpainter()
                
                # Load multiple methods
                methods = self.manga_settings.get('inpainting', {}).get('hybrid_methods', [])
                loaded = 0
                
                for method_config in methods:
                    method = method_config.get('method')
                    model_path = method_config.get('model_path')
                    
                    if method and model_path:
                        if self.hybrid_inpainter.add_method(method, method, model_path):
                            loaded += 1
                            self._log(f"✅ Added {method.upper()} to hybrid inpainter")
                
                if loaded > 0:
                    self._log(f"✅ Hybrid inpainter ready with {loaded} methods")
                else:
                    self._log("⚠️ Hybrid inpainter initialized but no methods loaded", "warning")
                
                return True
            
            return False
            
        except ImportError:
            self._log("❌ Local inpainter module not available", "error")
            return False
        except Exception as e:
            self._log(f"❌ Error initializing inpainter: {e}", "error")
            return False


    def inpaint_regions(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Inpaint using configured method (cloud, local, or hybrid)"""
        if self.skip_inpainting:
            self._log("   ⏭️ Skipping inpainting (preserving original art)", "info")
            return image.copy()
        
        inpaint_method = self.manga_settings.get('inpainting', {}).get('method', 'cloud')
        
        if inpaint_method == 'local':
            # Check if local_inpainter exists
            if not hasattr(self, 'local_inpainter'):
                self._log("   ⚠️ Local inpainter not initialized, attempting now...", "warning")
                self._initialize_local_inpainter()
            
            if hasattr(self, 'local_inpainter') and self.local_inpainter:
                # Check if model is loaded
                if not self.local_inpainter.model_loaded:
                    self._log("   ⚠️ No model loaded, attempting to load...", "warning")
                    local_method = self.manga_settings.get('inpainting', {}).get('local_method', 'anime')
                    
                    # Try to download JIT model
                    model_path = self.local_inpainter.download_jit_model(local_method)
                    if model_path:
                        self.local_inpainter.load_model(local_method, model_path)
                
                if self.local_inpainter.model_loaded:
                    self._log("   🖥️ Using local inpainting", "info")
                    return self.local_inpainter.inpaint(image, mask)
                else:
                    self._log("   ❌ No model loaded, returning original", "error")
                    return image.copy()
            else:
                self._log("   ❌ Local inpainter not available", "error")
                return image.copy()
        
        elif inpaint_method == 'hybrid' and hasattr(self, 'hybrid_inpainter'):
            self._log("   🔄 Using hybrid ensemble inpainting", "info")
            return self.hybrid_inpainter.inpaint_ensemble(image, mask)
        
        elif inpaint_method == 'cloud' and hasattr(self, 'use_cloud_inpainting') and self.use_cloud_inpainting:
            return self._cloud_inpaint(image, mask)
        
        else:
            self._log(f"   ⚠️ No valid inpainting method '{inpaint_method}' available, returning original", "error")
            return image.copy()
            
    def _cloud_inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
            """Use Replicate API for inpainting"""
            try:
                import requests
                import base64
                from io import BytesIO
                from PIL import Image as PILImage
                import cv2
                
                self._log("   ☁️ Cloud inpainting via Replicate API", "info")
                
                # Convert to PIL
                image_pil = PILImage.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                mask_pil = PILImage.fromarray(mask).convert('L')
                
                # Convert to base64
                img_buffer = BytesIO()
                image_pil.save(img_buffer, format='PNG')
                img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
                
                mask_buffer = BytesIO()
                mask_pil.save(mask_buffer, format='PNG')
                mask_base64 = base64.b64encode(mask_buffer.getvalue()).decode()
                
                # Get cloud settings
                cloud_settings = self.main_gui.config.get('manga_settings', {})
                model_type = cloud_settings.get('cloud_inpaint_model', 'ideogram-v2')
                timeout = cloud_settings.get('cloud_timeout', 60)
                
                # Determine model identifier based on model type
                if model_type == 'ideogram-v2':
                    model = 'ideogram-ai/ideogram-v2'
                    self._log(f"   Using Ideogram V2 inpainting model", "info")
                elif model_type == 'sd-inpainting':
                    model = 'stability-ai/stable-diffusion-inpainting'
                    self._log(f"   Using Stable Diffusion inpainting model", "info")
                elif model_type == 'flux-inpainting':
                    model = 'zsxkib/flux-dev-inpainting'
                    self._log(f"   Using FLUX inpainting model", "info")
                elif model_type == 'custom':
                    model = cloud_settings.get('cloud_custom_version', '')
                    if not model:
                        raise Exception("No custom model identifier specified")
                    self._log(f"   Using custom model: {model}", "info")
                else:
                    # Default to Ideogram V2
                    model = 'ideogram-ai/ideogram-v2'
                    self._log(f"   Using default Ideogram V2 model", "info")
                
                # Build input data based on model type
                input_data = {
                    'image': f'data:image/png;base64,{img_base64}',
                    'mask': f'data:image/png;base64,{mask_base64}'
                }
                
                # Add prompt settings for models that support them
                if model_type in ['ideogram-v2', 'sd-inpainting', 'flux-inpainting', 'custom']:
                    prompt = cloud_settings.get('cloud_inpaint_prompt', 'clean background, smooth surface')
                    input_data['prompt'] = prompt
                    self._log(f"   Prompt: {prompt}", "info")
                    
                    # SD-specific parameters
                    if model_type == 'sd-inpainting':
                        negative_prompt = cloud_settings.get('cloud_negative_prompt', 'text, writing, letters')
                        input_data['negative_prompt'] = negative_prompt
                        input_data['num_inference_steps'] = cloud_settings.get('cloud_inference_steps', 20)
                        self._log(f"   Negative prompt: {negative_prompt}", "info")
                
                # Get the latest version of the model
                headers = {
                    'Authorization': f'Token {self.replicate_api_key}',
                    'Content-Type': 'application/json'
                }
                
                # First, get the latest version of the model
                model_response = requests.get(
                    f'https://api.replicate.com/v1/models/{model}',
                    headers=headers
                )
                
                if model_response.status_code != 200:
                    # If model lookup fails, try direct prediction with model identifier
                    self._log(f"   Model lookup returned {model_response.status_code}, trying direct prediction", "warning")
                    version = None
                else:
                    model_info = model_response.json()
                    version = model_info.get('latest_version', {}).get('id')
                    if not version:
                        raise Exception(f"Could not get version for model {model}")
                
                # Create prediction
                prediction_data = {
                    'input': input_data
                }
                
                if version:
                    prediction_data['version'] = version
                else:
                    # For custom models, try extracting version from model string
                    if ':' in model:
                        # Format: owner/model:version
                        model_name, version_id = model.split(':', 1)
                        prediction_data['version'] = version_id
                    else:
                        raise Exception(f"Could not determine version for model {model}. Try using format: owner/model:version")
                
                response = requests.post(
                    'https://api.replicate.com/v1/predictions',
                    headers=headers,
                    json=prediction_data
                )
                
                if response.status_code != 201:
                    raise Exception(f"API error: {response.text}")
                    
                # Get prediction URL
                prediction = response.json()
                prediction_url = prediction.get('urls', {}).get('get') or prediction.get('id')
                
                if not prediction_url:
                    raise Exception("No prediction URL returned")
                
                # If we only got an ID, construct the URL
                if not prediction_url.startswith('http'):
                    prediction_url = f'https://api.replicate.com/v1/predictions/{prediction_url}'
                
                # Poll for result with configured timeout
                import time
                for i in range(timeout):
                    response = requests.get(prediction_url, headers=headers)
                    result = response.json()
                    
                    # Log progress every 5 seconds
                    if i % 5 == 0 and i > 0:
                        self._log(f"   ⏳ Still processing... ({i}s elapsed)", "info")
                    
                    if result['status'] == 'succeeded':
                        # Download result image (handle both single URL and list)
                        output = result.get('output')
                        if not output:
                            raise Exception("No output returned from model")
                        
                        if isinstance(output, list):
                            output_url = output[0] if output else None
                        else:
                            output_url = output
                        
                        if not output_url:
                            raise Exception("No output URL in result")
                            
                        img_response = requests.get(output_url)
                        
                        # Convert back to numpy
                        result_pil = PILImage.open(BytesIO(img_response.content))
                        result_bgr = cv2.cvtColor(np.array(result_pil), cv2.COLOR_RGB2BGR)
                        
                        self._log("   ✅ Cloud inpainting completed", "success")
                        return result_bgr
                        
                    elif result['status'] == 'failed':
                        error_msg = result.get('error', 'Unknown error')
                        # Check for common errors
                        if 'version' in error_msg.lower():
                            error_msg += f" (Try using the model identifier '{model}' in the custom field)"
                        raise Exception(f"Inpainting failed: {error_msg}")
                        
                    time.sleep(1)
                    
                raise Exception(f"Timeout waiting for inpainting (>{timeout}s)")
                
            except Exception as e:
                self._log(f"   ❌ Cloud inpainting failed: {str(e)}", "error")
                return image.copy()         
            

    def _regions_overlap(self, region1: TextRegion, region2: TextRegion) -> bool:
        """Check if two regions overlap"""
        x1, y1, w1, h1 = region1.bounding_box
        x2, y2, w2, h2 = region2.bounding_box
        
        # Check if rectangles overlap
        if (x1 + w1 < x2 or x2 + w2 < x1 or 
            y1 + h1 < y2 or y2 + h2 < y1):
            return False
        
        return True

    def _calculate_overlap_area(self, region1: TextRegion, region2: TextRegion) -> float:
        """Calculate the area of overlap between two regions"""
        x1, y1, w1, h1 = region1.bounding_box
        x2, y2, w2, h2 = region2.bounding_box
        
        # Calculate intersection
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        return (x_right - x_left) * (y_bottom - y_top)

    def _adjust_overlapping_regions(self, regions: List[TextRegion], image_width: int, image_height: int) -> List[TextRegion]:
        """Adjust positions of overlapping regions to prevent overlap"""
        if len(regions) <= 1:
            return regions
        
        # Create a copy of regions to modify
        adjusted_regions = []
        for region in regions:
            # Create a new TextRegion with copied values
            adjusted_region = TextRegion(
                text=region.text,
                vertices=list(region.vertices),
                bounding_box=list(region.bounding_box),  # Make it mutable
                confidence=region.confidence,
                region_type=region.region_type
            )
            if hasattr(region, 'translated_text'):
                adjusted_region.translated_text = region.translated_text
            adjusted_regions.append(adjusted_region)
        
        # Sort by y-coordinate (top to bottom) then x-coordinate (left to right)
        adjusted_regions.sort(key=lambda r: (r.bounding_box[1], r.bounding_box[0]))
        
        # Adjust overlapping regions
        for i in range(len(adjusted_regions)):
            for j in range(i + 1, len(adjusted_regions)):
                region1 = adjusted_regions[i]
                region2 = adjusted_regions[j]
                
                if self._regions_overlap(region1, region2):
                    # Calculate overlap
                    overlap_area = self._calculate_overlap_area(region1, region2)
                    x1, y1, w1, h1 = region1.bounding_box
                    x2, y2, w2, h2 = region2.bounding_box
                    
                    # Determine adjustment direction based on relative positions
                    center1_x = x1 + w1 / 2
                    center1_y = y1 + h1 / 2
                    center2_x = x2 + w2 / 2
                    center2_y = y2 + h2 / 2
                    
                    # Calculate minimum separation needed
                    min_gap = 10  # Minimum pixels between regions
                    
                    # Prefer vertical adjustment for manga (usually vertical text flow)
                    vertical_overlap = min(y1 + h1, y2 + h2) - max(y1, y2)
                    horizontal_overlap = min(x1 + w1, x2 + w2) - max(x1, x2)
                    
                    if vertical_overlap < horizontal_overlap:
                        # Adjust vertically
                        if center2_y > center1_y:
                            # Move region2 down
                            new_y2 = y1 + h1 + min_gap
                            if new_y2 + h2 <= image_height:
                                region2.bounding_box = (x2, new_y2, w2, h2)
                                self._log(f"  📍 Moved region down to prevent overlap", "info")
                            else:
                                # Move region1 up if region2 can't go down
                                new_y1 = y2 - h1 - min_gap
                                if new_y1 >= 0:
                                    region1.bounding_box = (x1, new_y1, w1, h1)
                                    self._log(f"  📍 Moved region up to prevent overlap", "info")
                        else:
                            # Move region2 up
                            new_y2 = y1 - h2 - min_gap
                            if new_y2 >= 0:
                                region2.bounding_box = (x2, new_y2, w2, h2)
                                self._log(f"  📍 Moved region up to prevent overlap", "info")
                            else:
                                # Move region1 down if region2 can't go up
                                new_y1 = y2 + h2 + min_gap
                                if new_y1 + h1 <= image_height:
                                    region1.bounding_box = (x1, new_y1, w1, h1)
                                    self._log(f"  📍 Moved region down to prevent overlap", "info")
                    else:
                        # Adjust horizontally
                        if center2_x > center1_x:
                            # Move region2 right
                            new_x2 = x1 + w1 + min_gap
                            if new_x2 + w2 <= image_width:
                                region2.bounding_box = (new_x2, y2, w2, h2)
                                self._log(f"  📍 Moved region right to prevent overlap", "info")
                            else:
                                # Move region1 left if region2 can't go right
                                new_x1 = x2 - w1 - min_gap
                                if new_x1 >= 0:
                                    region1.bounding_box = (new_x1, y1, w1, h1)
                                    self._log(f"  📍 Moved region left to prevent overlap", "info")
                        else:
                            # Move region2 left
                            new_x2 = x1 - w2 - min_gap
                            if new_x2 >= 0:
                                region2.bounding_box = (new_x2, y2, w2, h2)
                                self._log(f"  📍 Moved region left to prevent overlap", "info")
                            else:
                                # Move region1 right if region2 can't go left
                                new_x1 = x2 + w2 + min_gap
                                if new_x1 + w1 <= image_width:
                                    region1.bounding_box = (new_x1, y1, w1, h1)
                                    self._log(f"  📍 Moved region right to prevent overlap", "info")
        
        return adjusted_regions

    
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
        
        # Get image dimensions for boundary checking
        image_height, image_width = image.shape[:2]
        
        # Adjust overlapping regions before rendering
        adjusted_regions = self._adjust_overlapping_regions(regions, image_width, image_height)
        
        # Check if any regions still overlap after adjustment (shouldn't happen, but let's verify)
        has_overlaps = False
        for i, region1 in enumerate(adjusted_regions):
            for region2 in adjusted_regions[i+1:]:
                if self._regions_overlap(region1, region2):
                    has_overlaps = True
                    self._log("  ⚠️ Regions still overlap after adjustment", "warning")
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
            
            for region in adjusted_regions:
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
                    # Pass the region to use vertices
                    if hasattr(region, 'vertices') and region.vertices:
                        _, _, safe_w, safe_h = self.get_safe_text_area(region)
                        lines = self._wrap_text(region.translated_text, 
                                              self._get_font(font_size), 
                                              safe_w, region_draw)
                    else:
                        lines = self._wrap_text(region.translated_text, 
                                              self._get_font(font_size), 
                                              int(w * 0.8), region_draw)
                elif self.font_size_mode == 'multiplier':
                    # Use dynamic sizing with multiplier - pass region for vertices
                    font_size, lines = self._fit_text_to_region(
                        region.translated_text, w, h, region_draw, region
                    )
                else:
                    # Auto mode - use standard fitting - pass region for vertices
                    font_size, lines = self._fit_text_to_region(
                        region.translated_text, w, h, region_draw, region
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
            
            for region in adjusted_regions:
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
     
    def get_safe_text_area(self, region: TextRegion) -> Tuple[int, int, int, int]:
        """Get safe text area with less conservative margins for readability"""
        if not hasattr(region, 'vertices') or not region.vertices:
            x, y, w, h = region.bounding_box
            margin_factor = 0.85  # Less conservative default
            safe_width = int(w * margin_factor)
            safe_height = int(h * margin_factor)
            safe_x = x + (w - safe_width) // 2
            safe_y = y + (h - safe_height) // 2
            return safe_x, safe_y, safe_width, safe_height
        
        try:
            # Convert vertices to numpy array with correct dtype
            vertices = np.array(region.vertices, dtype=np.int32)
            hull = cv2.convexHull(vertices)
            hull_area = cv2.contourArea(hull)
            poly_area = cv2.contourArea(vertices)
            
            if poly_area > 0:
                convexity = hull_area / poly_area
            else:
                convexity = 1.0
            
            # LESS CONSERVATIVE margins for better readability
            if convexity < 0.85:  # Speech bubble with tail
                margin_factor = 0.75
                self._log(f"  Speech bubble detected, using 75% of area", "info")
            elif convexity > 0.98:  # Rectangular
                margin_factor = 0.9
                self._log(f"  Rectangular bubble, using 90% of area", "info")
            else:  # Regular bubble
                margin_factor = 0.8
                self._log(f"  Regular bubble, using 80% of area", "info")
        except:
            margin_factor = 0.8  # Safe default
        
        # Convert vertices to numpy array for boundingRect
        vertices_np = np.array(region.vertices, dtype=np.int32)
        x, y, w, h = cv2.boundingRect(vertices_np)
        
        safe_width = int(w * margin_factor)
        safe_height = int(h * margin_factor)
        safe_x = x + (w - safe_width) // 2
        safe_y = y + (h - safe_height) // 2
        
        return safe_x, safe_y, safe_width, safe_height
    
    def _fit_text_to_region(self, text: str, max_width: int, max_height: int, draw: ImageDraw, region: TextRegion = None) -> Tuple[int, List[str]]:
        """Find optimal font size with smart strategy selection based on context"""
        
        # MINIMUM READABLE SIZE (Option 3)
        MIN_READABLE_SIZE = self.min_readable_size
        
        # Check if this region might overlap with others (Option 1 trigger)
        has_potential_overlap = self._check_potential_overlap(region) if region else False
        
        # Get usable area
        if region and hasattr(region, 'vertices') and region.vertices:
            safe_x, safe_y, safe_width, safe_height = self.get_safe_text_area(region)
            usable_width = safe_width
            usable_height = safe_height
            self._log(f"  Using vertex-based safe area: {safe_width}x{safe_height}", "info")
        else:
            # LESS CONSERVATIVE for better readability (Option 2)
            margin = 0.75 if has_potential_overlap else 0.85  # Use more space when no overlap
            usable_width = int(max_width * margin)
            usable_height = int(max_height * margin)
            self._log(f"  Using {int(margin*100)}% margins (overlap: {has_potential_overlap})", "info")
        
        # Font size range with maximum limit applied
        min_font_size = max(self.min_font_size, MIN_READABLE_SIZE)  # Enforce minimum
        max_font_size = min(self.max_font_size, self.max_font_size_limit)  # Apply user-configured maximum
        
        # Calculate text metrics
        text_length = len(text.strip())
        area = usable_width * usable_height
        
        # OPTION 1: Simple top-down approach for overlapping regions
        if has_potential_overlap:
            self._log("  Using conservative sizing due to potential overlap", "info")
            return self._fit_text_simple_topdown(text, usable_width, usable_height, draw, min_font_size, max_font_size)
        
        # OPTION 2: Less conservative approach for non-overlapping regions
        # Start with HIGHER estimates
        if text_length > 0:
            pixels_per_char = area / text_length
            
            # MORE AGGRESSIVE sizing
            if pixels_per_char > 800:
                initial_estimate = int(max_font_size * 0.9)  # Was 0.7
            elif pixels_per_char > 400:
                initial_estimate = int(max_font_size * 0.7)  # Was 0.5
            elif pixels_per_char > 200:
                initial_estimate = int(max_font_size * 0.5)  # Was 0.35
            elif pixels_per_char > 100:
                initial_estimate = int(max_font_size * 0.4)  # Was 0.25
            else:
                initial_estimate = int(max_font_size * 0.3)  # Was 0.2
        else:
            initial_estimate = int(max_font_size * 0.6)  # Was 0.4
        
        # Ensure we start high enough
        initial_estimate = max(initial_estimate, min_font_size + 10)
        initial_estimate = min(initial_estimate, max_font_size)
        
        self._log(f"  Text length: {text_length}, Initial estimate: {initial_estimate}", "info")
        
        # Handle multiplier modes
        if self.font_size_mode == 'multiplier' and not self.constrain_to_bubble:
            # Unconstrained multiplier
            base_size = initial_estimate
            target_size = int(base_size * self.font_size_multiplier)
            target_size = max(min_font_size, min(target_size, self.max_font_size_limit * 3))  # Apply limit even in unconstrained mode
            
            font = self._get_font(target_size)
            lines = self._wrap_text(text, font, usable_width, draw)
            
            return target_size, lines
        
        # Binary search with preference for LARGER sizes
        low = min_font_size
        high = initial_estimate
        best_size = min_font_size
        best_lines = []
        
        # First, try to find the largest size that fits
        while low <= high:
            mid = (low + high) // 2
            font = self._get_font(mid)
            lines = self._wrap_text(text, font, usable_width, draw)
            
            # Less strict height check - use more vertical space
            line_height = mid * 1.25  # Reduced from 1.3 for tighter spacing
            total_height = len(lines) * line_height
            
            if total_height <= usable_height:  # No padding, use full height
                best_size = mid
                best_lines = lines
                low = mid + 1  # Try even larger
            else:
                high = mid - 1
        
        # Apply multiplier if needed
        if self.font_size_mode == 'multiplier' and self.constrain_to_bubble:
            multiplied_size = int(best_size * self.font_size_multiplier)
            multiplied_size = max(min_font_size, min(multiplied_size, self.max_font_size_limit))  # Apply maximum limit
            
            font = self._get_font(multiplied_size)
            lines = self._wrap_text(text, font, usable_width, draw)
            line_height = multiplied_size * 1.25
            total_height = len(lines) * line_height
            
            if total_height <= usable_height:
                best_size = multiplied_size
                best_lines = lines
        
        # OPTION 3: Enforce minimum readable size
        if best_size < MIN_READABLE_SIZE:
            self._log(f"  Size {best_size} below minimum, using {MIN_READABLE_SIZE}", "warning")
            best_size = MIN_READABLE_SIZE
            font = self._get_font(best_size)
            best_lines = self._wrap_text(text, font, usable_width, draw)
            
            # If it doesn't fit at minimum size, we still use it (readability > fit)
            line_height = best_size * 1.25
            max_lines = int(usable_height / line_height)
            if len(best_lines) > max_lines > 0:
                # Only truncate if really necessary
                best_lines = best_lines[:max_lines]
                if len(best_lines) < len(self._wrap_text(text, font, usable_width, draw)):
                    best_lines[-1] = best_lines[-1][:-3] + "..."
        
        self._log(f"  Final size: {best_size}, Lines: {len(best_lines)}", "info")
        
        return best_size, best_lines

    def _fit_text_simple_topdown(self, text: str, usable_width: int, usable_height: int, 
                                 draw: ImageDraw, min_size: int, max_size: int) -> Tuple[int, List[str]]:
        """Simple top-down approach - start large and shrink only if needed"""
        # Start from a reasonable large size
        start_size = int(max_size * 0.8)
        
        for font_size in range(start_size, min_size - 1, -2):  # Step by 2 for speed
            font = self._get_font(font_size)
            lines = self._wrap_text(text, font, usable_width, draw)
            
            line_height = font_size * 1.2  # Tighter for overlaps
            total_height = len(lines) * line_height
            
            if total_height <= usable_height:
                return font_size, lines
        
        # If nothing fits, use minimum
        font = self._get_font(min_size)
        lines = self._wrap_text(text, font, usable_width, draw)
        return min_size, lines

    def _check_potential_overlap(self, region: TextRegion) -> bool:
        """Check if this region might overlap with others based on position"""
        if not region or not hasattr(region, 'bounding_box'):
            return False
        
        x, y, w, h = region.bounding_box
        
        # Simple heuristic: small regions or regions at edges might overlap
        # You can make this smarter based on your needs
        if w < 100 or h < 50:  # Small bubbles often overlap
            return True
        
        # Add more overlap detection logic here if needed
        # For now, default to no overlap for larger bubbles
        return False
    
    def _wrap_text(self, text: str, font: ImageFont, max_width: int, draw: ImageDraw) -> List[str]:
        """Wrap text to fit within max_width with optional strict wrapping"""
        # Handle empty text
        if not text.strip():
            return []
        
        words = text.split()
        lines = []
        current_line = []
        
        for word in words:
            # Check if word alone is too long
            word_bbox = draw.textbbox((0, 0), word, font=font)
            word_width = word_bbox[2] - word_bbox[0]
            
            if word_width > max_width and len(word) > 1:
                # Word is too long for the bubble
                if current_line:
                    # Save current line first
                    lines.append(' '.join(current_line))
                    current_line = []
                
                if self.strict_text_wrapping:
                    # STRICT MODE: Force break the word to fit within bubble
                    # This is the original behavior that ensures text stays within bounds
                    broken_parts = self._force_break_word(word, font, max_width, draw)
                    lines.extend(broken_parts)
                else:
                    # RELAXED MODE: Keep word whole (may exceed bubble)
                    lines.append(word)
                    # self._log(f"  ⚠️ Word '{word}' exceeds bubble width, keeping whole", "warning")
            else:
                # Normal word processing
                if current_line:
                    test_line = ' '.join(current_line + [word])
                else:
                    test_line = word
                
                text_bbox = draw.textbbox((0, 0), test_line, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                
                if text_width <= max_width:
                    current_line.append(word)
                else:
                    if current_line:
                        lines.append(' '.join(current_line))
                        current_line = [word]
                    else:
                        # Single word that fits
                        lines.append(word)
        
        if current_line:
            lines.append(' '.join(current_line))
        
        return lines

    # Keep the existing _force_break_word method as is (the complete version from earlier):
    def _force_break_word(self, word: str, font: ImageFont, max_width: int, draw: ImageDraw) -> List[str]:
        """Force break a word that's too long to fit"""
        lines = []
        
        # Binary search to find how many characters fit
        low = 1
        high = len(word)
        chars_that_fit = 1
        
        while low <= high:
            mid = (low + high) // 2
            test_text = word[:mid]
            bbox = draw.textbbox((0, 0), test_text, font=font)
            width = bbox[2] - bbox[0]
            
            if width <= max_width:
                chars_that_fit = mid
                low = mid + 1
            else:
                high = mid - 1
        
        # Break the word into pieces
        remaining = word
        while remaining:
            if len(remaining) <= chars_that_fit:
                # Last piece
                lines.append(remaining)
                break
            else:
                # Find the best break point
                break_at = chars_that_fit
                
                # Try to break at a more natural point if possible
                # Look for vowel-consonant boundaries for better hyphenation
                for i in range(min(chars_that_fit, len(remaining) - 1), max(1, chars_that_fit - 5), -1):
                    if i < len(remaining) - 1:
                        current_char = remaining[i].lower()
                        next_char = remaining[i + 1].lower()
                        
                        # Good hyphenation points:
                        # - Between consonant and vowel
                        # - After prefix (un-, re-, pre-, etc.)
                        # - Before suffix (-ing, -ed, -er, etc.)
                        if (current_char in 'bcdfghjklmnpqrstvwxyz' and next_char in 'aeiou') or \
                           (current_char in 'aeiou' and next_char in 'bcdfghjklmnpqrstvwxyz'):
                            break_at = i + 1
                            break
                
                # Add hyphen if we're breaking in the middle of a word
                if break_at < len(remaining):
                    # Check if adding hyphen still fits
                    test_with_hyphen = remaining[:break_at] + '-'
                    bbox = draw.textbbox((0, 0), test_with_hyphen, font=font)
                    width = bbox[2] - bbox[0]
                    
                    if width <= max_width:
                        lines.append(remaining[:break_at] + '-')
                    else:
                        # Hyphen doesn't fit, break without it
                        lines.append(remaining[:break_at])
                else:
                    lines.append(remaining[:break_at])
                
                remaining = remaining[break_at:]
        
        return lines
    
    def _estimate_font_size_for_region(self, region: TextRegion) -> int:
        """Estimate the likely font size for a text region based on its dimensions and text content"""
        x, y, w, h = region.bounding_box
        text_length = len(region.text.strip())
        
        if text_length == 0:
            return self.max_font_size // 2  # Default middle size
        
        # Calculate area per character
        area = w * h
        area_per_char = area / text_length
        
        # Estimate font size based on area per character
        # These ratios are approximate and based on typical manga text
        if area_per_char > 800:
            estimated_size = int(self.max_font_size * 0.8)
        elif area_per_char > 400:
            estimated_size = int(self.max_font_size * 0.6)
        elif area_per_char > 200:
            estimated_size = int(self.max_font_size * 0.4)
        elif area_per_char > 100:
            estimated_size = int(self.max_font_size * 0.3)
        else:
            estimated_size = int(self.max_font_size * 0.2)
        
        # Clamp to reasonable bounds
        return max(self.min_font_size, min(estimated_size, self.max_font_size))


    def _likely_different_bubbles(self, region1: TextRegion, region2: TextRegion) -> bool:
        """Detect if regions are likely in different speech bubbles based on spatial patterns"""
        x1, y1, w1, h1 = region1.bounding_box
        x2, y2, w2, h2 = region2.bounding_box
        
        # Calculate gaps and positions
        horizontal_gap = 0
        if x1 + w1 < x2:
            horizontal_gap = x2 - (x1 + w1)
        elif x2 + w2 < x1:
            horizontal_gap = x1 - (x2 + w2)
        
        vertical_gap = 0
        if y1 + h1 < y2:
            vertical_gap = y2 - (y1 + h1)
        elif y2 + h2 < y1:
            vertical_gap = y1 - (y2 + h2)
        
        # Calculate relative positions
        center_x1 = x1 + w1 / 2
        center_x2 = x2 + w2 / 2
        center_y1 = y1 + h1 / 2
        center_y2 = y2 + h2 / 2
        
        horizontal_center_diff = abs(center_x1 - center_x2)
        avg_width = (w1 + w2) / 2
        
        # FIRST CHECK: Very small gaps always indicate same bubble
        if horizontal_gap < 15 and vertical_gap < 15:
            return False  # Definitely same bubble
        
        # STRICTER CHECK: For regions that are horizontally far apart
        # Even if they pass the gap threshold, check if they're likely different bubbles
        if horizontal_gap > 40:  # Significant horizontal gap
            # Unless they're VERY well aligned vertically, they're different bubbles
            vertical_overlap = min(y1 + h1, y2 + h2) - max(y1, y2)
            min_height = min(h1, h2)
            
            if vertical_overlap < min_height * 0.8:  # Need 80% overlap to be same bubble
                return True
        
        # SPECIFIC FIX: Check for multi-line text pattern
        # If regions are well-aligned horizontally, they're likely in the same bubble
        if horizontal_center_diff < avg_width * 0.35:  # Relaxed from 0.2 to 0.35
            # Additional checks for multi-line text:
            # 1. Similar widths (common in speech bubbles)
            width_ratio = max(w1, w2) / min(w1, w2) if min(w1, w2) > 0 else 999
            
            # 2. Reasonable vertical spacing (not too far apart)
            avg_height = (h1 + h2) / 2
            
            if width_ratio < 2.0 and vertical_gap < avg_height * 1.5:
                # This is very likely multi-line text in the same bubble
                return False
        
        # Pattern 1: Side-by-side bubbles (common in manga)
        # Characteristics: Significant horizontal gap, similar vertical position
        if horizontal_gap > 50:  # Increased from 25 to avoid false positives
            vertical_overlap = min(y1 + h1, y2 + h2) - max(y1, y2)
            min_height = min(h1, h2)
            
            # If they have good vertical overlap, they're likely side-by-side bubbles
            if vertical_overlap > min_height * 0.5:
                return True
        
        # Pattern 2: Stacked bubbles
        # Characteristics: Significant vertical gap, similar horizontal position
        if vertical_gap > 25:  # Back to original threshold
            horizontal_overlap = min(x1 + w1, x2 + w2) - max(x1, x2)
            min_width = min(w1, w2)
            
            # If they have good horizontal overlap, they're likely stacked bubbles
            if horizontal_overlap > min_width * 0.5:
                return True
        
        # Pattern 3: Diagonal arrangement (different speakers)
        # If regions are separated both horizontally and vertically
        if horizontal_gap > 20 and vertical_gap > 20:
            return True
        
        # Pattern 4: Large gap relative to region size
        avg_height = (h1 + h2) / 2
        
        if horizontal_gap > avg_width * 0.6 or vertical_gap > avg_height * 0.6:
            return True
        
        return False

    def _regions_should_merge(self, region1: TextRegion, region2: TextRegion, threshold: int = 50) -> bool:
        """Determine if two regions should be merged - with bubble detection"""
        
        # First check if they're close enough spatially
        if not self._regions_are_nearby(region1, region2, threshold):
            return False
        
        x1, y1, w1, h1 = region1.bounding_box
        x2, y2, w2, h2 = region2.bounding_box
        
        # ONLY apply special handling if regions are from Azure
        if hasattr(region1, 'from_azure') and region1.from_azure:
            # Azure lines are typically small - be more lenient
            avg_height = (h1 + h2) / 2
            if avg_height < 50:  # Likely single lines
                self._log(f"   Azure lines detected, using lenient merge criteria", "info")
                
                center_x1 = x1 + w1 / 2
                center_x2 = x2 + w2 / 2
                horizontal_center_diff = abs(center_x1 - center_x2)
                avg_width = (w1 + w2) / 2
                
                # If horizontally aligned and nearby, merge them
                if horizontal_center_diff < avg_width * 0.7:
                    return True
        
        # GOOGLE LOGIC - unchanged from your original
        # SPECIAL CASE: If one region is very small, bypass strict checks
        area1 = w1 * h1
        area2 = w2 * h2
        if area1 < 500 or area2 < 500:
            self._log(f"   Small text region (area: {min(area1, area2)}), bypassing strict alignment checks", "info")
            return True
        
        # Calculate actual gaps between regions
        horizontal_gap = 0
        if x1 + w1 < x2:
            horizontal_gap = x2 - (x1 + w1)
        elif x2 + w2 < x1:
            horizontal_gap = x1 - (x2 + w2)
        
        vertical_gap = 0
        if y1 + h1 < y2:
            vertical_gap = y2 - (y1 + h1)
        elif y2 + h2 < y1:
            vertical_gap = y1 - (y2 + h2)
        
        # Calculate centers for alignment checks
        center_x1 = x1 + w1 / 2
        center_x2 = x2 + w2 / 2
        center_y1 = y1 + h1 / 2
        center_y2 = y2 + h2 / 2
        
        horizontal_center_diff = abs(center_x1 - center_x2)
        vertical_center_diff = abs(center_y1 - center_y2)
        
        avg_width = (w1 + w2) / 2
        avg_height = (h1 + h2) / 2
        
        # Determine text orientation and layout
        is_horizontal_text = horizontal_gap > vertical_gap or (horizontal_center_diff < avg_width * 0.5)
        is_vertical_text = vertical_gap > horizontal_gap or (vertical_center_diff < avg_height * 0.5)
        
        # PRELIMINARY CHECK: If regions overlap or are extremely close, merge them
        # This handles text that's clearly in the same bubble
        
        # Check for overlap
        overlap_x = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
        overlap_y = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
        has_overlap = overlap_x > 0 and overlap_y > 0
        
        if has_overlap:
            self._log(f"   Regions overlap - definitely same bubble, merging", "info")
            return True
        
        # If gaps are tiny (< 10 pixels), merge regardless of other factors
        if horizontal_gap < 10 and vertical_gap < 10:
            self._log(f"   Very small gaps ({horizontal_gap}, {vertical_gap}) - merging", "info")
            return True
        
        # BUBBLE BOUNDARY CHECK: Use spatial patterns to detect different bubbles
        # But be less aggressive if gaps are small
        if horizontal_gap < 20 and vertical_gap < 20:
            # Very close regions are almost certainly in the same bubble
            self._log(f"   Regions very close, skipping bubble boundary check", "info")
        elif self._likely_different_bubbles(region1, region2):
            self._log(f"   Regions likely in different speech bubbles", "info")
            return False
        
        # CHECK 1: For well-aligned text with small gaps, merge immediately
        # This catches multi-line text in the same bubble
        if is_horizontal_text and vertical_center_diff < avg_height * 0.4:
            # Horizontal text that's well-aligned vertically
            if horizontal_gap <= threshold and vertical_gap <= threshold * 0.5:
                self._log(f"   Well-aligned horizontal text with acceptable gaps, merging", "info")
                return True
        
        if is_vertical_text and horizontal_center_diff < avg_width * 0.4:
            # Vertical text that's well-aligned horizontally
            if vertical_gap <= threshold and horizontal_gap <= threshold * 0.5:
                self._log(f"   Well-aligned vertical text with acceptable gaps, merging", "info")
                return True
        
        # ADDITIONAL CHECK: Multi-line text in speech bubbles
        # Even if not perfectly aligned, check for typical multi-line patterns
        if horizontal_center_diff < avg_width * 0.5 and vertical_gap <= threshold:
            # Lines that are reasonably centered and within threshold should merge
            self._log(f"   Multi-line text pattern detected, merging", "info")
            return True
        
        # CHECK 2: Check alignment quality
        # Poor alignment often indicates different bubbles
        if is_horizontal_text:
            # For horizontal text, check vertical alignment
            if vertical_center_diff > avg_height * 0.6:
                self._log(f"   Poor vertical alignment for horizontal text", "info")
                return False
        elif is_vertical_text:
            # For vertical text, check horizontal alignment
            if horizontal_center_diff > avg_width * 0.6:
                self._log(f"   Poor horizontal alignment for vertical text", "info")
                return False
        
        # CHECK 3: Font size check (but be reasonable)
        font_size1 = self._estimate_font_size_for_region(region1)
        font_size2 = self._estimate_font_size_for_region(region2)
        size_ratio = max(font_size1, font_size2) / max(min(font_size1, font_size2), 1)
        
        # Allow some variation for emphasis or stylistic choices
        if size_ratio > 2.0:
            self._log(f"   Font sizes too different ({font_size1} vs {font_size2})", "info")
            return False
        
        # CHECK 4: Final sanity check on merged area
        merged_width = max(x1 + w1, x2 + w2) - min(x1, x2)
        merged_height = max(y1 + h1, y2 + h2) - min(y1, y2)
        merged_area = merged_width * merged_height
        combined_area = (w1 * h1) + (w2 * h2)
        
        # If merged area is way larger than combined areas, they're probably far apart
        if merged_area > combined_area * 2.5:
            self._log(f"   Merged area indicates regions are too far apart", "info")
            return False
        
        # If we get here, apply standard threshold checks
        if horizontal_gap <= threshold and vertical_gap <= threshold:
            self._log(f"   Standard threshold check passed, merging", "info")
            return True
        
        self._log(f"   No merge conditions met", "info")
        return False

    def _merge_nearby_regions(self, regions: List[TextRegion], threshold: int = 50) -> List[TextRegion]:
        """Merge text regions that are likely part of the same speech bubble - with debug logging"""
        if len(regions) <= 1:
            return regions
        
        self._log(f"\n=== MERGE DEBUG: Starting merge analysis ===", "info")
        self._log(f"  Total regions: {len(regions)}", "info")
        self._log(f"  Threshold: {threshold}px", "info")
        
        # First, let's log what regions we have
        for i, region in enumerate(regions):
            x, y, w, h = region.bounding_box
            self._log(f"  Region {i}: pos({x},{y}) size({w}x{h}) text='{region.text[:20]}...'", "info")
        
        # Sort regions by area (largest first) to handle contained regions properly
        sorted_indices = sorted(range(len(regions)), 
                              key=lambda i: regions[i].bounding_box[2] * regions[i].bounding_box[3], 
                              reverse=True)
        
        merged = []
        used = set()
        
        # Process each region in order of size (largest first)
        for idx in sorted_indices:
            i = idx
            if i in used:
                continue
            
            region1 = regions[i]
            
            # Start with this region
            merged_text = region1.text
            merged_vertices = list(region1.vertices) if hasattr(region1, 'vertices') else []
            regions_merged = [i]  # Track which regions were merged
            
            self._log(f"\n  Checking region {i} for merges:", "info")
            
            # Check against all other unused regions
            for j in range(len(regions)):
                if j == i or j in used:
                    continue
                
                region2 = regions[j]
                self._log(f"    Testing merge with region {j}:", "info")
                
                # Check if region2 is contained within region1
                x1, y1, w1, h1 = region1.bounding_box
                x2, y2, w2, h2 = region2.bounding_box
                
                # Check if region2 is fully contained within region1
                if (x2 >= x1 and y2 >= y1 and 
                    x2 + w2 <= x1 + w1 and y2 + h2 <= y1 + h1):
                    self._log(f"      ✓ Region {j} is INSIDE region {i} - merging!", "success")
                    merged_text += " " + region2.text
                    if hasattr(region2, 'vertices'):
                        merged_vertices.extend(region2.vertices)
                    used.add(j)
                    regions_merged.append(j)
                    continue
                
                # Check if region1 is contained within region2 (shouldn't happen due to sorting, but be safe)
                if (x1 >= x2 and y1 >= y2 and 
                    x1 + w1 <= x2 + w2 and y1 + h1 <= y2 + h2):
                    self._log(f"      ✓ Region {i} is INSIDE region {j} - merging!", "success")
                    merged_text += " " + region2.text
                    if hasattr(region2, 'vertices'):
                        merged_vertices.extend(region2.vertices)
                    used.add(j)
                    regions_merged.append(j)
                    # Update region1's bounding box to the larger region
                    region1 = TextRegion(
                        text=merged_text,
                        vertices=merged_vertices,
                        bounding_box=region2.bounding_box,
                        confidence=region1.confidence,
                        region_type='temp_merge'
                    )
                    continue
                
                # FIX: Always check proximity against ORIGINAL regions, not the expanded one
                # This prevents cascade merging across bubble boundaries
                if self._regions_are_nearby(regions[i], region2, threshold):  # Use regions[i] not region1
                    #self._log(f"      ✓ Regions are nearby", "info")
                    
                    # Then check if they should merge (also use original region)
                    if self._regions_should_merge(regions[i], region2, threshold):  # Use regions[i] not region1
                        #self._log(f"      ✓ Regions should merge!", "success")
                        
                        # Actually perform the merge
                        merged_text += " " + region2.text
                        if hasattr(region2, 'vertices'):
                            merged_vertices.extend(region2.vertices)
                        used.add(j)
                        regions_merged.append(j)
                        
                        # DON'T update region1 for proximity checks - keep using original regions
                    else:
                        self._log(f"      ✗ Regions should not merge", "warning")
                else:
                    self._log(f"      ✗ Regions not nearby", "warning")
            
            # Log if we merged multiple regions
            if len(regions_merged) > 1:
                self._log(f"  ✅ MERGED regions {regions_merged} into one bubble", "success")
            else:
                self._log(f"  ℹ️ Region {i} not merged with any other", "info")
            
            # Create final merged region with all the merged vertices
            if merged_vertices:
                xs = [v[0] for v in merged_vertices]
                ys = [v[1] for v in merged_vertices]
            else:
                # Fallback: calculate from all merged regions
                all_xs = []
                all_ys = []
                for idx in regions_merged:
                    x, y, w, h = regions[idx].bounding_box
                    all_xs.extend([x, x + w])
                    all_ys.extend([y, y + h])
                xs = all_xs
                ys = all_ys
            
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            merged_bbox = (min_x, min_y, max_x - min_x, max_y - min_y)
            
            merged_region = TextRegion(
                text=merged_text,
                vertices=merged_vertices,
                bounding_box=merged_bbox,
                confidence=regions[i].confidence,
                region_type='merged_text_block' if len(regions_merged) > 1 else regions[i].region_type
            )
            
            # Copy over any additional attributes
            if hasattr(regions[i], 'translated_text'):
                merged_region.translated_text = regions[i].translated_text
            
            merged.append(merged_region)
            used.add(i)
        
        self._log(f"\n=== MERGE DEBUG: Complete ===", "info")
        self._log(f"  Final region count: {len(merged)} (was {len(regions)})", "info")
        
        # Verify the merge worked
        if len(merged) == len(regions):
            self._log(f"  ⚠️ WARNING: No regions were actually merged!", "warning")
        
        return merged
    
    def _regions_are_nearby(self, region1: TextRegion, region2: TextRegion, threshold: int = 50) -> bool:
        """Check if two regions are close enough to be in the same bubble - WITH DEBUG"""
        x1, y1, w1, h1 = region1.bounding_box
        x2, y2, w2, h2 = region2.bounding_box
        
        #self._log(f"\n    === NEARBY CHECK DEBUG ===", "info")
        #self._log(f"    Region 1: pos({x1},{y1}) size({w1}x{h1})", "info")
        #self._log(f"    Region 2: pos({x2},{y2}) size({w2}x{h2})", "info")
        #self._log(f"    Threshold: {threshold}", "info")
        
        # Calculate gaps between closest edges
        horizontal_gap = 0
        if x1 + w1 < x2:  # region1 is to the left
            horizontal_gap = x2 - (x1 + w1)
        elif x2 + w2 < x1:  # region2 is to the left
            horizontal_gap = x1 - (x2 + w2)
        
        vertical_gap = 0
        if y1 + h1 < y2:  # region1 is above
            vertical_gap = y2 - (y1 + h1)
        elif y2 + h2 < y1:  # region2 is above
            vertical_gap = y1 - (y2 + h2)
        
        #self._log(f"    Horizontal gap: {horizontal_gap}", "info")
        #self._log(f"    Vertical gap: {vertical_gap}", "info")
        
        # Detect if regions are likely vertical text based on aspect ratio
        aspect1 = w1 / max(h1, 1)
        aspect2 = w2 / max(h2, 1)
        
        # More permissive vertical text detection
        # Vertical text typically has aspect ratio < 1.0 (taller than wide)
        is_vertical_text = (aspect1 < 1.0 and aspect2 < 1.0) or (aspect1 < 0.5 or aspect2 < 0.5)
        
        # Also check if text is arranged vertically (one above the other with minimal horizontal offset)
        center_x1 = x1 + w1 / 2
        center_x2 = x2 + w2 / 2
        horizontal_center_diff = abs(center_x1 - center_x2)
        avg_width = (w1 + w2) / 2
        
        # If regions are vertically stacked with aligned centers, treat as vertical text
        is_vertically_stacked = (horizontal_center_diff < avg_width * 1.5) and (vertical_gap >= 0)
        
        #self._log(f"    Is vertical text: {is_vertical_text}", "info")
        #self._log(f"    Is vertically stacked: {is_vertically_stacked}", "info")
        #self._log(f"    Horizontal center diff: {horizontal_center_diff:.1f}", "info")
        
        # SIMPLE APPROACH: Just check if gaps are within threshold
        # Don't overthink it
        if horizontal_gap <= threshold and vertical_gap <= threshold:
            #self._log(f"    ✅ NEARBY: Both gaps within threshold", "success")
            return True
        
        # SPECIAL CASE: Vertically stacked text with good alignment
        # This is specifically for multi-line text in bubbles
        if horizontal_center_diff < avg_width * 0.8 and vertical_gap <= threshold * 1.5:
            #self._log(f"    ✅ NEARBY: Vertically aligned text in same bubble", "success")
            return True
        
        # If one gap is small and the other is slightly over, still consider nearby
        if (horizontal_gap <= threshold * 0.5 and vertical_gap <= threshold * 1.5) or \
           (vertical_gap <= threshold * 0.5 and horizontal_gap <= threshold * 1.5):
            #self._log(f"    ✅ NEARBY: One small gap, other slightly over", "success")
            return True
        
        # Special case: Wide bubbles with text on sides
        # If regions are at nearly the same vertical position, they might be in a wide bubble
        if abs(y1 - y2) < 10:  # Nearly same vertical position
            # Check if this could be a wide bubble spanning both regions
            if horizontal_gap <= threshold * 3:  # Allow up to 3x threshold for wide bubbles
                #self._log(f"    ✅ NEARBY: Same vertical level, possibly wide bubble", "success")
                return True
        
        #self._log(f"    ❌ NOT NEARBY: Gaps exceed threshold", "warning")
        return False
    
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
        
        # Check if parallel processing is enabled
        parallel_enabled = self.manga_settings.get('advanced', {}).get('parallel_processing', False)
        max_workers = self.manga_settings.get('advanced', {}).get('max_workers', 4)
        
        if parallel_enabled and len(regions) > 1:
            self._log(f"🚀 Using PARALLEL processing with {max_workers} workers")
            return self._translate_regions_parallel(regions, image_path, max_workers)
        else:
            # SEQUENTIAL CODE
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
                    
                    # SAVE TO HISTORY HERE
                    if self.history_manager and self.contextual_enabled and translated:
                        try:
                            self.history_manager.append_to_history(
                                user_content=region.text,
                                assistant_content=translated,
                                hist_limit=self.translation_history_limit,
                                reset_on_limit=not self.rolling_history_enabled,
                                rolling_window=self.rolling_history_enabled
                            )
                            self._log(f"📚 Saved to history (exchange {i+1})")
                        except Exception as e:
                            self._log(f"⚠️ Failed to save history: {e}", "warning")
                    
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

    #  parallel processing:
    def _translate_regions_parallel(self, regions: List[TextRegion], image_path: str, max_workers: int) -> List[TextRegion]:
        """Translate regions using parallel processing"""
        # Thread-safe storage for results
        results_lock = threading.Lock()
        translated_regions = {}
        failed_indices = []
        
        # Filter out empty regions
        valid_regions = [(i, region) for i, region in enumerate(regions) if region.text.strip()]
        
        if not valid_regions:
            return regions
        
        # Create a thread pool
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all translation tasks
            future_to_data = {}
            
            for i, region in valid_regions:
                # Check for stop signal before submitting
                if self._check_stop():
                    self._log(f"\n⏹️ Translation stopped before submitting region {i+1}", "warning")
                    break
                
                # Submit translation task
                future = executor.submit(
                    self._translate_single_region_parallel, 
                    region, 
                    i, 
                    len(valid_regions),
                    image_path
                )
                future_to_data[future] = (i, region)
            
            # Process completed translations
            completed = 0
            for future in as_completed(future_to_data):
                i, region = future_to_data[future]
                
                # Check for stop signal
                if self._check_stop():
                    self._log(f"\n⏹️ Translation stopped at {completed}/{len(valid_regions)} completed", "warning")
                    # Cancel remaining futures
                    for f in future_to_data:
                        f.cancel()
                    break
                
                try:
                    translated_text = future.result()
                    if translated_text:
                        with results_lock:
                            translated_regions[i] = translated_text
                        completed += 1
                        self._log(f"✅ [{completed}/{len(valid_regions)}] Completed region {i+1}")
                    else:
                        with results_lock:
                            failed_indices.append(i)
                        self._log(f"❌ [{completed}/{len(valid_regions)}] Failed region {i+1}", "error")
                
                except Exception as e:
                    with results_lock:
                        failed_indices.append(i)
                    self._log(f"❌ Error in region {i+1}: {str(e)}", "error")
        
        # Apply translations back to regions
        for i, region in enumerate(regions):
            if i in translated_regions:
                region.translated_text = translated_regions[i]
        
        # Report summary
        success_count = len(translated_regions)
        fail_count = len(failed_indices)
        self._log(f"\n📊 Parallel translation complete: {success_count} succeeded, {fail_count} failed")
        
        return regions

    def reset_for_new_image(self):
        """Reset internal state for processing a new image"""
        # Clear any cached detection results
        if hasattr(self, 'last_detection_results'):
            del self.last_detection_results
        
        # Clear OCR manager cache if it exists
        if hasattr(self, 'ocr_manager') and self.ocr_manager:
            if hasattr(self.ocr_manager, 'last_results'):
                self.ocr_manager.last_results = None
            if hasattr(self.ocr_manager, 'cache'):
                self.ocr_manager.cache = {}
        
        # Don't clear translation context if using rolling history
        if not self.rolling_history_enabled:
            self.translation_context = []
        
        # Clear any cached regions
        if hasattr(self, '_cached_regions'):
            del self._cached_regions
        
        self._log("🔄 Reset translator state for new image", "debug")

    def _translate_single_region_parallel(self, region: TextRegion, index: int, total: int, image_path: str) -> Optional[str]:
        """Translate a single region for parallel processing"""
        try:
            thread_name = threading.current_thread().name
            self._log(f"\n[{thread_name}] [{index+1}/{total}] Original: {region.text}")
            
            # Note: Context is not used in parallel mode to avoid race conditions
            # Pass None for context to maintain compatibility with your translate_text method
            translated = self.translate_text(
                region.text,
                None,  # No context in parallel mode
                image_path=image_path,
                region=region
            )
            
            if translated:
                self._log(f"[{thread_name}] Translated: {translated}")
                
                # Add random delay to prevent API rate limiting
                import random
                delay = self.api_delay + random.uniform(0, 0.5)
                time.sleep(delay)
                
                return translated
            else:
                self._log(f"[{thread_name}] Translation failed", "error")
                return None
                
        except Exception as e:
            self._log(f"[{thread_name}] Error: {str(e)}", "error")
            return None


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
            'interrupted': False,
            'format_info': {}
        }
        
        try:
            # Determine the output directory from output_path
            if output_path:
                output_dir = os.path.dirname(output_path)
            else:
                # If no output path specified, use default
                output_dir = os.path.join(os.path.dirname(image_path), "translated_images")
            
            # Ensure output directory exists
            os.makedirs(output_dir, exist_ok=True)
            
            # Initialize HistoryManager with the output directory
            if self.contextual_enabled and not self.history_manager_initialized:
                # Only initialize if we're in a new output directory
                if output_dir != getattr(self, 'history_output_dir', None):
                    try:
                        self.history_manager = HistoryManager(output_dir)
                        self.history_manager_initialized = True
                        self.history_output_dir = output_dir
                        self._log(f"📚 Initialized HistoryManager in output directory: {output_dir}")
                    except Exception as e:
                        self._log(f"⚠️ Failed to initialize history manager: {str(e)}", "warning")
                        self.history_manager = None
            
            # Check for stop signal
            if self._check_stop():
                result['interrupted'] = True
                self._log("⏹️ Translation stopped before processing", "warning")
                return result
            
            # Format detection if enabled
            if self.manga_settings.get('advanced', {}).get('format_detection', False):
                self._log("🔍 Analyzing image format...")
                img = Image.open(image_path)
                width, height = img.size
                aspect_ratio = height / width
                
                # Detect format type
                format_info = {
                    'width': width,
                    'height': height,
                    'aspect_ratio': aspect_ratio,
                    'is_webtoon': aspect_ratio > 3.0,
                    'is_spread': width > height * 1.3,
                    'format': 'unknown'
                }
                
                if format_info['is_webtoon']:
                    format_info['format'] = 'webtoon'
                    self._log("📱 Detected WEBTOON format - vertical scroll manga")
                elif format_info['is_spread']:
                    format_info['format'] = 'spread'
                    self._log("📖 Detected SPREAD format - two-page layout")
                else:
                    format_info['format'] = 'single_page'
                    self._log("📄 Detected SINGLE PAGE format")
                
                result['format_info'] = format_info
                
                # Handle webtoon mode if detected and enabled
                webtoon_mode = self.manga_settings.get('advanced', {}).get('webtoon_mode', 'auto')
                if format_info['is_webtoon'] and webtoon_mode != 'disabled':
                    if webtoon_mode == 'auto' or webtoon_mode == 'force':
                        self._log("🔄 Webtoon mode active - will process in chunks for better OCR")
                        # Process webtoon in chunks
                        return self._process_webtoon_chunks(image_path, output_path, result)
            
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
                result['success'] = True
                return result
            
            self._log(f"\n✅ Detection complete: {len(regions)} regions found")
            
            # Save debug image if debug mode is enabled
            if self.manga_settings.get('advanced', {}).get('debug_mode', False):
                self._save_debug_image(image_path, regions)
            
            # Step 2: Translate regions
            self._log(f"\n📍 [STEP 2] Translation Phase")
            
            if self.full_page_context_enabled:
                # Full page context translation mode
                self._log(f"\n📄 Using FULL PAGE CONTEXT mode")
                self._log("   This mode sends all text together for more consistent translations", "info")
                
                if self._check_stop():
                    result['interrupted'] = True
                    self._log("\n⏹️ Translation stopped before processing", "warning")
                    return result
                
                translations = self.translate_full_page_context(regions, image_path)
                
                if translations:
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
                    self._log("❌ Full page context translation failed", "error")
                    result['errors'].append("Full page context translation failed")
                    
            else:
                # Individual translation mode with parallel processing support
                self._log(f"\n📝 Using INDIVIDUAL translation mode")
                
                if self.manga_settings.get('advanced', {}).get('parallel_processing', False):
                    self._log("⚡ Parallel processing ENABLED")
                    regions = self._translate_regions_parallel(regions, image_path)
                else:
                    regions = self.translate_regions(regions, image_path)

            # Check if we should continue after translation
            if self._check_stop():
                result['interrupted'] = True
                self._log("⏹️ Translation cancelled before image processing", "warning")
                result['regions'] = [r.to_dict() for r in regions]
                return result

            if not any(region.translated_text for region in regions):
                result['interrupted'] = True
                self._log("⏹️ No regions were translated - translation was interrupted", "warning")
                result['regions'] = [r.to_dict() for r in regions]
                return result
            
            # Step 3: Render translated text
            self._log(f"\n📍 [STEP 3] Image Processing Phase")
            
            # Load image with OpenCV
            import cv2
            self._log(f"🖼️ Loading image with OpenCV...")
            try:
                image = cv2.imread(image_path)
                
                if image is None:
                    self._log(f"   Using PIL to handle Unicode path...", "info")
                    from PIL import Image as PILImage
                    import numpy as np
                    
                    pil_image = PILImage.open(image_path)
                    image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                    self._log(f"   ✅ Successfully loaded with PIL", "info")
                    
            except Exception as e:
                error_msg = f"Failed to load image: {image_path} - {str(e)}"
                self._log(f"❌ {error_msg}", "error")
                result['errors'].append(error_msg)
                return result

            self._log(f"   Image dimensions: {image.shape[1]}x{image.shape[0]}")
            
            # Save intermediate preprocessing image if enabled
            if self.manga_settings.get('advanced', {}).get('save_intermediate', False):
                self._save_intermediate_image(image_path, image, "original")
            
            # Check if we should skip inpainting
            if self.skip_inpainting:
                # User wants to preserve original art
                self._log(f"🎨 Skipping inpainting (preserving original art)", "info")
                self._log(f"   Background opacity: {int(self.text_bg_opacity/255*100)}%", "info")
                inpainted = image.copy()
            else:
                self._log(f"🎭 Creating text mask...")
                mask = self.create_text_mask(image, regions)
                
                # Debug save mask
                try:
                    mask_path = image_path.replace('.', '_mask.')
                    cv2.imwrite(mask_path, mask)
                    mask_percentage = ((mask > 0).sum() / mask.size) * 100
                    self._log(f"   🎭 DEBUG: Saved mask to {mask_path}", "info")
                    self._log(f"   📊 Mask coverage: {mask_percentage:.1f}% of image", "info")
                    
                    # Save mask overlay visualization
                    mask_viz = image.copy()
                    mask_viz[mask > 0] = [0, 0, 255]  # Simple red overlay
                    viz_path = image_path.replace('.', '_mask_overlay.')
                    cv2.imwrite(viz_path, mask_viz)
                    self._log(f"   🎭 DEBUG: Saved mask overlay to {viz_path}", "info")
                    
                    if mask_percentage > 50:
                        self._log(f"   ⚠️ WARNING: Mask covers {mask_percentage:.1f}% - this might be too much!", "warning")
                except Exception as e:
                    self._log(f"   ❌ Failed to save mask debug: {str(e)}", "error")
                
                if self.manga_settings.get('advanced', {}).get('save_intermediate', False):
                    self._save_intermediate_image(image_path, mask, "mask")
                
                self._log(f"🎨 Inpainting to remove original text")
                inpainted = self.inpaint_regions(image, mask)
                
                if self.manga_settings.get('advanced', {}).get('save_intermediate', False):
                    self._save_intermediate_image(image_path, inpainted, "inpainted")
            
            # Render translated text
            self._log(f"✍️ Rendering translated text...")
            self._log(f"   Using enhanced renderer with custom settings", "info")
            final_image = self.render_translated_text(inpainted, regions)
            
            # Save output
            try:
                if not output_path:
                    base, ext = os.path.splitext(image_path)
                    output_path = f"{base}_translated{ext}"
                
                success = cv2.imwrite(output_path, final_image)
                
                if not success:
                    self._log(f"   Using PIL to save with Unicode path...", "info")
                    from PIL import Image as PILImage
                    
                    rgb_image = cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB)
                    pil_image = PILImage.fromarray(rgb_image)
                    pil_image.save(output_path)
                    self._log(f"   ✅ Successfully saved with PIL", "info")
                
                result['output_path'] = output_path
                self._log(f"\n💾 Saved output to: {output_path}")
                
            except Exception as e:
                error_msg = f"Failed to save output image: {str(e)}"
                self._log(f"❌ {error_msg}", "error")
                result['errors'].append(error_msg)
                result['success'] = False
                return result
            
            # Update result
            result['regions'] = [r.to_dict() for r in regions]
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

    def reset_history_manager(self):
        """Reset history manager for new translation batch"""
        self.history_manager = None
        self.history_manager_initialized = False
        self.history_output_dir = None
        self.translation_context = []
        self._log("📚 Reset history manager for new batch", "debug")
    
    def _process_webtoon_chunks(self, image_path: str, output_path: str, result: Dict) -> Dict:
        """Process webtoon in chunks for better OCR"""
        # Implementation for processing tall images in chunks
        
    def _translate_regions_parallel(self, regions: List[TextRegion], image_path: str) -> List[TextRegion]:
        """Translate regions using parallel processing"""
        max_workers = self.manga_settings.get('advanced', {}).get('max_workers', 4)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit translation tasks
            future_to_region = {
                executor.submit(self._translate_single_region, region, i): (i, region) 
                for i, region in enumerate(regions)
            }
            
            # Collect results
            for future in as_completed(future_to_region):
                i, region = future_to_region[future]
                try:
                    translated_text = future.result()
                    regions[i].translated_text = translated_text
                except Exception as e:
                    self._log(f"Translation failed for region {i}: {e}", "error")
        
        return regions

    def _save_intermediate_image(self, original_path: str, image, stage: str):
        """Save intermediate processing stages"""
        debug_dir = os.path.join(os.path.dirname(original_path), 'debug')
        os.makedirs(debug_dir, exist_ok=True)
        
        base_name = os.path.splitext(os.path.basename(original_path))[0]
        output_path = os.path.join(debug_dir, f"{base_name}_{stage}.png")
        
        cv2.imwrite(output_path, image)
        self._log(f"   💾 Saved {stage} image: {output_path}")
