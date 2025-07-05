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
        """Detect text regions using Google Cloud Vision API with preprocessing"""
        self._log(f"🔍 Detecting text regions in: {os.path.basename(image_path)}")
        
        try:
            # Get manga settings from main_gui config
            manga_settings = self.main_gui.config.get('manga_settings', {})
            preprocessing = manga_settings.get('preprocessing', {})
            ocr_settings = manga_settings.get('ocr', {})
            
            # Load and preprocess image if enabled
            if preprocessing.get('enabled', True):
                self._log("📐 Preprocessing enabled - enhancing image quality")
                processed_image_data = self._preprocess_image(image_path, preprocessing)
            else:
                # Read image file without preprocessing
                with open(image_path, 'rb') as image_file:
                    processed_image_data = image_file.read()
            
            # Create Vision API image object
            image = vision.Image(content=processed_image_data)
            
            # Configure text detection based on settings
            detection_mode = ocr_settings.get('text_detection_mode', 'document')
            
            if detection_mode == 'document':
                # Perform document text detection (better for manga/dense text)
                response = self.vision_client.document_text_detection(
                    image=image,
                    image_context=vision.ImageContext(
                        language_hints=ocr_settings.get('language_hints', ['ja', 'ko', 'zh'])
                    )
                )
            else:
                # Regular text detection
                response = self.vision_client.text_detection(
                    image=image,
                    image_context=vision.ImageContext(
                        language_hints=ocr_settings.get('language_hints', ['ja', 'ko', 'zh'])
                    )
                )
            
            if response.error.message:
                raise Exception(f"Cloud Vision API error: {response.error.message}")
            
            regions = []
            confidence_threshold = ocr_settings.get('confidence_threshold', 0.8)
            
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
            
            # Merge nearby regions based on settings
            merge_threshold = ocr_settings.get('merge_nearby_threshold', 50)
            regions = self._merge_nearby_regions(regions, threshold=merge_threshold)
            
            self._log(f"✅ Detected {len(regions)} text regions")
            
            # Save debug images if enabled
            advanced_settings = manga_settings.get('advanced', {})
            if advanced_settings.get('debug_mode', False) or advanced_settings.get('save_intermediate', False):
                self._save_debug_image(image_path, regions)
            
            return regions
            
        except Exception as e:
            self._log(f"❌ Error detecting text: {str(e)}", "error")
            raise

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
                
                # Add detected text preview if in verbose debug mode
                if self.manga_settings.get('advanced', {}).get('save_intermediate', False):
                    text_preview = region.text[:20] + "..." if len(region.text) > 20 else region.text
                    cv2.putText(overlay, text_preview, (x, y + h + 15), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.4, color, 1, cv2.LINE_AA)
            
            # Save main debug image
            if self.manga_settings.get('advanced', {}).get('save_intermediate', False):
                debug_path = os.path.join(debug_dir, f"{base_name}_debug_regions.png")
            else:
                debug_path = image_path.replace('.', '_debug.')
            
            cv2.imwrite(debug_path, overlay)
            self._log(f"   📸 Saved debug image: {debug_path}")
            
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
                
                # Save text density map
                density_map = self._create_text_density_map(img, regions)
                density_path = os.path.join(debug_dir, f"{base_name}_text_density.png")
                cv2.imwrite(density_path, density_map)
                self._log(f"   📊 Saved text density map: {density_path}")
                
                # Save individual region crops
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
                    
                    region_crop = img[y1:y2, x1:x2]
                    region_path = os.path.join(regions_dir, f"region_{i:03d}.png")
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

    def _create_text_density_map(self, img, regions):
        """Create a map showing text density across the image"""
        # Create grid
        grid_size = 50
        height, width = img.shape[:2]
        density_map = np.zeros((height // grid_size + 1, width // grid_size + 1))
        
        # Calculate density
        for region in regions:
            x, y, w, h = region.bounding_box
            grid_x = x // grid_size
            grid_y = y // grid_size
            grid_w = (w // grid_size) + 1
            grid_h = (h // grid_size) + 1
            
            for gy in range(grid_y, min(grid_y + grid_h, density_map.shape[0])):
                for gx in range(grid_x, min(grid_x + grid_w, density_map.shape[1])):
                    density_map[gy, gx] += 1
        
        # Upscale and visualize
        density_resized = cv2.resize(density_map, (width, height), interpolation=cv2.INTER_NEAREST)
        density_normalized = (density_resized / density_resized.max() * 255).astype(np.uint8)
        density_colored = cv2.applyColorMap(density_normalized, cv2.COLORMAP_HOT)
        
        # Blend with original
        result = cv2.addWeighted(img, 0.5, density_colored, 0.5, 0)
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
            translations_to_save = []
            
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
                    # Collect translations to save in batch
                    translations_to_save.append((region.text, translated))
            
            # Save ALL translations as a batch without triggering limits between each one
            if self.history_manager and self.contextual_enabled and translations_to_save:
                try:
                    # Save all translations from this page without checking limits between each
                    for original, translated in translations_to_save:
                        self.history_manager.append_to_history(
                            user_content=original,
                            assistant_content=translated,
                            hist_limit=999999,  # Temporarily disable limit checking
                            reset_on_limit=False,
                            rolling_window=False
                        )
                    
                    # Now apply the actual limit AFTER all saves
                    current_history = self.history_manager.load_history()
                    current_exchanges = len(current_history) // 2
                    
                    if current_exchanges > self.translation_history_limit:
                        if self.rolling_history_enabled:
                            # Keep only the most recent exchanges
                            messages_to_keep = self.translation_history_limit * 2
                            trimmed_history = current_history[-messages_to_keep:]
                            self.history_manager.save_history(trimmed_history)
                            self._log(f"📚 Trimmed history to last {self.translation_history_limit} exchanges")
                        else:
                            # Reset mode - clear older entries
                            messages_to_keep = self.translation_history_limit * 2
                            trimmed_history = current_history[-messages_to_keep:]
                            self.history_manager.save_history(trimmed_history)
                            self._log(f"📚 Reset history to last {self.translation_history_limit} exchanges")
                    
                    self._log(f"📚 Saved {len(translations_to_save)} translations to history", "success")
                    
                    # Show current history status
                    final_exchanges = len(self.history_manager.load_history()) // 2
                    self._log(f"📚 History now contains {final_exchanges} exchanges", "info")
                    
                except Exception as e:
                    self._log(f"⚠️ Failed to save translations to history: {str(e)}", "warning")
            
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
        """Create a binary mask with aggressive dilation for better inpainting"""
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        for region in regions:
            # Use the exact polygon vertices from Cloud Vision
            pts = np.array(region.vertices, np.int32)
            pts = pts.reshape((-1, 1, 2))
            
            # Fill the polygon
            import cv2
            cv2.fillPoly(mask, [pts], 255)
        
        # Get dilation settings from config
        config = self.main_gui.config if hasattr(self.main_gui, 'config') else {}
        dilation_size = config.get('manga_inpaint_dilation', 15)
        
        # Much more aggressive dilation based on quality setting
        if self.inpaint_quality == 'high':
            # Very aggressive for manga - use elliptical kernel for smoother edges
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_size, dilation_size))
            mask = cv2.dilate(mask, kernel, iterations=2)
            
            # Optional: Add slight blur to smooth mask edges
            mask = cv2.GaussianBlur(mask, (5, 5), 0)
            _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        else:
            # Standard dilation
            kernel = np.ones((10, 10), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)
        
        self._log(f"   Created mask with dilation size: {dilation_size if self.inpaint_quality == 'high' else 10}", "info")
        
        return mask
    
    def inpaint_regions(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Inpaint using cloud API or return original"""
        if self.skip_inpainting:
            self._log("   ⏭️ Skipping inpainting (preserving original art)", "info")
            return image.copy()
            
        if hasattr(self, 'use_cloud_inpainting') and self.use_cloud_inpainting:
            return self._cloud_inpaint(image, mask)
        else:
            self._log("   ⚠️ Inpainting not configured, returning original image", "warning")
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
    
    def _merge_nearby_regions(self, regions: List[TextRegion], threshold: int = 50) -> List[TextRegion]:
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
                
                if self._regions_are_nearby(region1, region2, threshold):
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
