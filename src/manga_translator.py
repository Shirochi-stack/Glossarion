# manga_translator.py
"""
Manga Translation Pipeline using Google Cloud Vision API and Your API Key
Handles OCR, translation, and text rendering for manga panels
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
        self.input_token_limit = int(main_gui.token_limit_entry.get() if hasattr(main_gui, 'token_limit_entry') else 120000)
        self.contextual_enabled = main_gui.contextual_var.get() if hasattr(main_gui, 'contextual_var') else False
        
        # Store context for contextual translation
        self.translation_context = []
        
        # Font settings for text rendering
        self.font_path = self._find_font()
        self.min_font_size = 8
        self.max_font_size = 36
        
        self._log("\n🔧 MangaTranslator initialized with settings:")
        self._log(f"   API Delay: {self.api_delay}s")
        self._log(f"   Temperature: {self.temperature}")
        self._log(f"   Max Output Tokens: {self.max_tokens}")
        self._log(f"   Input Token Limit: {self.input_token_limit}")
        self._log(f"   Contextual Translation: {'ENABLED' if self.contextual_enabled else 'DISABLED'}")
        self._log(f"   Font Path: {self.font_path or 'Default'}\n")
    
    def _log(self, message: str, level: str = "info"):
        """Log message to GUI or console"""
        if self.log_callback:
            self.log_callback(message, level)
        else:
            print(message)
    
    def _regions_overlap(self, region1: TextRegion, region2: TextRegion) -> bool:
        """Check if two regions overlap"""
        x1, y1, w1, h1 = region1.bounding_box
        x2, y2, w2, h2 = region2.bounding_box
        
        # Check for intersection
        return not (x1 + w1 < x2 or x2 + w2 < x1 or y1 + h1 < y2 or y2 + h2 < y1)
    
    def _merge_nearby_regions(self, regions: List[TextRegion]) -> List[TextRegion]:
        """Merge text regions that are close together (likely same bubble)"""
        if not regions:
            return regions
        
        merged = []
        used = set()
        
        for i, region1 in enumerate(regions):
            if i in used:
                continue
                
            # Start a new merged region
            merged_text = region1.text
            merged_vertices = list(region1.vertices)
            x_coords = [v[0] for v in region1.vertices]
            y_coords = [v[1] for v in region1.vertices]
            
            # Look for nearby regions to merge
            for j, region2 in enumerate(regions[i+1:], i+1):
                if j in used:
                    continue
                
                # Check if regions are close enough to merge
                if self._regions_are_nearby(region1, region2):
                    self._log(f"   🔗 Merging region {i} with region {j}")
                    used.add(j)
                    
                    # Combine text with space
                    merged_text += " " + region2.text
                    
                    # Expand bounding box
                    x_coords.extend([v[0] for v in region2.vertices])
                    y_coords.extend([v[1] for v in region2.vertices])
            
            # Create merged region with combined bounding box
            min_x, max_x = min(x_coords), max(x_coords)
            min_y, max_y = min(y_coords), max(y_coords)
            
            merged_vertices = [
                (min_x, min_y),
                (max_x, min_y),
                (max_x, max_y),
                (min_x, max_y)
            ]
            
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
        
        # Calculate centers
        center1_x = x1 + w1 // 2
        center1_y = y1 + h1 // 2
        center2_x = x2 + w2 // 2
        center2_y = y2 + h2 // 2
        
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
        
        for font in font_candidates:
            if os.path.exists(font):
                return font
        
        return None
    
    def detect_text_regions(self, image_path: str) -> List[TextRegion]:
        """Detect all text regions using Google Cloud Vision API"""
        regions = []
        
        self._log(f"🔍 Detecting text regions in: {os.path.basename(image_path)}")
        
        try:
            # Read image
            with open(image_path, 'rb') as image_file:
                content = image_file.read()
            
            self._log(f"📊 Image loaded: {len(content):,} bytes")
            
            image = vision.Image(content=content)
            
            # Perform text detection
            self._log(f"☁️ Calling Google Cloud Vision API...")
            start_time = time.time()
            response = self.vision_client.document_text_detection(image=image)
            api_time = time.time() - start_time
            self._log(f"✅ Vision API responded in {api_time:.2f} seconds")
            
            if response.error.message:
                raise Exception(f'Vision API error: {response.error.message}')
            
            # Process text blocks from the response
            if response.full_text_annotation:
                pages = response.full_text_annotation.pages
                self._log(f"📄 Found {len(pages)} pages in response")
                
                for page_idx, page in enumerate(pages):
                    blocks = page.blocks
                    self._log(f"📦 Page {page_idx}: {len(blocks)} text blocks detected")
                    
                    for block_idx, block in enumerate(blocks):
                        # Get block vertices (polygon outline)
                        vertices = [(v.x, v.y) for v in block.bounding_box.vertices]
                        
                        # Calculate bounding rectangle
                        xs = [v[0] for v in vertices]
                        ys = [v[1] for v in vertices]
                        bbox = (min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))
                        
                        # Extract text from block
                        block_text = ""
                        for paragraph in block.paragraphs:
                            for word in paragraph.words:
                                word_text = ''.join([symbol.text for symbol in word.symbols])
                                block_text += word_text + " "
                        
                        block_text = block_text.strip()
                        
                        if block_text:
                            self._log(f"   📝 Block {block_idx}: '{block_text[:50]}...' at {bbox}")
                            
                            region = TextRegion(
                                text=block_text,
                                vertices=vertices,
                                bounding_box=bbox,
                                confidence=block.confidence,
                                region_type='text_block'
                            )
                            regions.append(region)
            else:
                self._log(f"⚠️ No text annotation found in response", "warning")
            
            self._log(f"✨ Total regions detected: {len(regions)}")
            
            # Merge nearby regions that likely belong to the same bubble
            merged_regions = self._merge_nearby_regions(regions)
            self._log(f"🔀 After merging nearby text: {len(merged_regions)} regions")
            
            # Check for overlapping regions that might cause double rendering
            for i, region1 in enumerate(merged_regions):
                for j, region2 in enumerate(merged_regions[i+1:], i+1):
                    if self._regions_overlap(region1, region2):
                        self._log(f"⚠️ Warning: Regions {i} and {j} overlap!", "warning")
                        self._log(f"   Region {i}: '{region1.text[:30]}...' at {region1.bounding_box}", "warning")
                        self._log(f"   Region {j}: '{region2.text[:30]}...' at {region2.bounding_box}", "warning")
            
            return merged_regions
            
        except Exception as e:
            self._log(f"❌ Vision API error: {str(e)}", "error")
            raise
    
    def translate_text(self, text: str, context: str = "", image_path: str = None, region: TextRegion = None) -> str:
        """Translate text using API with GUI system prompt and full image context"""
        try:
            self._log(f"\n🌐 Starting translation for text: '{text[:50]}...'")
            
            # Build messages using system prompt from GUI
            system_prompt = self.main_gui.prompt_profiles.get(
                self.main_gui.profile_var.get(), 
                "Translate the following text."
            )
            
            self._log(f"📋 Using profile: {self.main_gui.profile_var.get()}")
            self._log(f"📝 System prompt: {system_prompt[:100]}...")
            
            messages = [{"role": "system", "content": system_prompt}]
            
            # Add contextual translations if enabled
            if self.contextual_enabled and self.translation_context:
                context_count = len(self.translation_context[-3:])
                self._log(f"🔗 Contextual enabled: Adding {context_count} previous translations")
                
                # Include previous translations as context
                for ctx in self.translation_context[-3:]:  # Last 3 translations
                    messages.extend([
                        {"role": "user", "content": ctx["original"]},
                        {"role": "assistant", "content": ctx["translated"]}
                    ])
            else:
                self._log(f"🔗 Contextual: {'Enabled but no context yet' if self.contextual_enabled else 'Disabled'}")
            
            # Add full image context if available
            if image_path:
                try:
                    import base64
                    from PIL import Image as PILImage
                    
                    self._log(f"🖼️ Adding full page visual context for translation")
                    
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
            self._log(f"📊 Token estimate - Text: {text_tokens}, Images: {image_tokens} (Total: {estimated_tokens} / {self.input_token_limit})")
            
            if estimated_tokens > self.input_token_limit:
                self._log(f"⚠️ Token limit exceeded, trimming context", "warning")
                # Keep system prompt, image, and current text only
                if image_path:
                    messages = [messages[0], messages[-1]]  
                else:
                    messages = [messages[0], {"role": "user", "content": text}]
            
            # Call API with GUI settings
            self._log(f"🚀 Calling API...")
            self._log(f"   - Temperature: {self.temperature}")
            self._log(f"   - Max tokens: {self.max_tokens}")
            
            start_time = time.time()
            response = self.client.send(
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            api_time = time.time() - start_time
            self._log(f"✅ API responded in {api_time:.2f} seconds")
            
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
            
            # Store in context for future translations
            if self.contextual_enabled:
                self.translation_context.append({
                    "original": text,
                    "translated": translated
                })
                self._log(f"💾 Stored in context (total: {len(self.translation_context)} entries)")
            
            return translated
            
        except Exception as e:
            self._log(f"❌ Translation error: {str(e)}", "error")
            self._log(f"   Error type: {type(e).__name__}", "error")
            import traceback
            self._log(f"   Traceback: {traceback.format_exc()}", "error")
            return text
    
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
        """Simple inpainting by filling with white (typical manga background)"""
        # For manga, most speech bubbles are white
        # So we'll just fill masked areas with white
        result = image.copy()
        result[mask > 0] = 255  # Set to white
        
        return result
    
    def render_translated_text(self, image: np.ndarray, regions: List[TextRegion]) -> np.ndarray:
        """Render translated text onto image"""
        # Convert to PIL for text rendering
        import cv2
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)
        
        for region in regions:
            if not region.translated_text:
                continue
            
            x, y, w, h = region.bounding_box
            
            # Find optimal font size
            font_size, lines = self._fit_text_to_region(
                region.translated_text, w, h, draw
            )
            
            # Load font
            if self.font_path:
                font = ImageFont.truetype(self.font_path, font_size)
            else:
                font = ImageFont.load_default()
            
            # Calculate vertical centering
            line_height = font_size * 1.2
            total_height = len(lines) * line_height
            start_y = y + (h - total_height) // 2
            
            # Draw each line
            for i, line in enumerate(lines):
                # Get line width for centering
                text_bbox = draw.textbbox((0, 0), line, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                
                # Center horizontally
                text_x = x + (w - text_width) // 2
                text_y = start_y + i * line_height
                
                # Draw with white outline for visibility
                outline_width = max(1, font_size // 15)
                
                # Draw outline
                for dx in range(-outline_width, outline_width + 1):
                    for dy in range(-outline_width, outline_width + 1):
                        if dx != 0 or dy != 0:
                            draw.text((text_x + dx, text_y + dy), line, 
                                    font=font, fill=(255, 255, 255))
                
                # Draw main text in black
                draw.text((text_x, text_y), line, font=font, fill=(0, 0, 0))
        
        # Convert back to numpy array
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
    def _fit_text_to_region(self, text: str, max_width: int, max_height: int, draw: ImageDraw) -> Tuple[int, List[str]]:
        """Find optimal font size and text wrapping"""
        # Use 80% of the region for text (leave margins)
        usable_width = int(max_width * 0.8)
        usable_height = int(max_height * 0.8)
        
        # Try different font sizes
        for font_size in range(self.max_font_size, self.min_font_size, -1):
            if self.font_path:
                font = ImageFont.truetype(self.font_path, font_size)
            else:
                font = ImageFont.load_default()
            
            # Wrap text
            lines = self._wrap_text(text, font, usable_width, draw)
            
            # Check if it fits vertically
            line_height = font_size * 1.2
            total_height = len(lines) * line_height
            
            if total_height <= usable_height:
                return font_size, lines
        
        # If nothing fits, use minimum size
        if self.font_path:
            font = ImageFont.truetype(self.font_path, self.min_font_size)
        else:
            font = ImageFont.load_default()
        
        lines = self._wrap_text(text, font, usable_width, draw)
        
        # Truncate if needed
        max_lines = int(usable_height // (self.min_font_size * 1.2))
        if len(lines) > max_lines:
            lines = lines[:max_lines-1] + [lines[max_lines-1][:10] + '...']
        
        return self.min_font_size, lines
    
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
                    # Word is too long, add it anyway
                    lines.append(word)
                    current_line = []
        
        if current_line:
            lines.append(' '.join(current_line))
        
        return lines
    
    def process_single_image(self, image_path: str, output_path: Optional[str] = None) -> Dict[str, Any]:
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
            'errors': []
        }
        
        try:
            # Step 1: Detect text regions using Google Cloud Vision
            self._log(f"📍 [STEP 1] Text Detection Phase")
            regions = self.detect_text_regions(image_path)
            
            if not regions:
                error_msg = "No text regions detected by Cloud Vision"
                self._log(f"⚠️ {error_msg}", "warning")
                result['errors'].append(error_msg)
                return result
            
            self._log(f"\n✅ Detection complete: {len(regions)} regions found")
            
            # Step 2: Translate each region
            self._log(f"\n📍 [STEP 2] Translation Phase")
            
            for i, region in enumerate(regions):
                self._log(f"\n🔄 Region {i+1}/{len(regions)}:")
                self._log(f"   Original: '{region.text[:50]}...'")
                self._log(f"   Location: {region.bounding_box}")
                
                region.translated_text = self.translate_text(
                region.text, 
                image_path=image_path,
                region=region
                )
                
                self._log(f"   Translated: '{region.translated_text[:50]}...'")
                
                # Respect API delay between translations
                if i < len(regions) - 1:
                    self._log(f"⏱️ Waiting {self.api_delay}s before next translation...")
                    time.sleep(self.api_delay)
            
            # Step 3: Process image
            self._log(f"\n📍 [STEP 3] Image Processing Phase")
            
            import cv2
            self._log(f"🖼️ Loading image with OpenCV...")
            image = cv2.imread(image_path)
            self._log(f"   Image dimensions: {image.shape[1]}x{image.shape[0]}")
            
            # Create mask for text regions
            self._log(f"🎭 Creating text mask...")
            mask = self.create_text_mask(image, regions)
            
            # Inpaint to remove original text
            self._log(f"🎨 Inpainting to remove original text...")
            inpainted = self.inpaint_regions(image, mask)
            
            # Render translated text
            self._log(f"✍️ Rendering translated text...")
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
            result['success'] = True
            result['regions'] = [r.to_dict() for r in regions]
            
            self._log(f"\n✅ TRANSLATION PIPELINE COMPLETE", "success")
            self._log(f"{'='*60}\n")
            
            # Clear context if it gets too large
            if len(self.translation_context) > 10:
                old_count = len(self.translation_context)
                self.translation_context = self.translation_context[-5:]
                self._log(f"🧹 Trimmed context from {old_count} to {len(self.translation_context)} entries")
            
        except Exception as e:
            error_msg = f"Error processing image: {str(e)}\n{traceback.format_exc()}"
            self._log(f"\n❌ PIPELINE ERROR:", "error")
            self._log(f"   {str(e)}", "error")
            self._log(f"   Type: {type(e).__name__}", "error")
            self._log(traceback.format_exc(), "error")
            result['errors'].append(error_msg)
        
        return result