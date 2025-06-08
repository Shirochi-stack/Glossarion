"""
Image Translation Module for EPUB Translator
Handles detection, extraction, and translation of images containing text
Includes support for web novel images and watermark handling
"""

import os
import json
import base64
import zipfile
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter
import io
from typing import List, Dict, Optional, Tuple
import re
from bs4 import BeautifulSoup
import logging
import time

logger = logging.getLogger(__name__)

class ImageTranslator:
    def __init__(self, client, output_dir: str, source_lang: str = "korean", system_prompt: str = ""):
        """
        Initialize the image translator
        
        Args:
            client: UnifiedClient instance for API calls
            output_dir: Directory to save translated images
            source_lang: Source language for translation
            system_prompt: System prompt from GUI to use for translation
        """
        self.client = client
        self.output_dir = output_dir
        self.source_lang = source_lang
        self.system_prompt = system_prompt  # Store GUI system prompt
        self.images_dir = os.path.join(output_dir, "images")
        self.translated_images_dir = os.path.join(output_dir, "translated_images")
        os.makedirs(self.translated_images_dir, exist_ok=True)
        
        # Track processed images to avoid duplicates
        self.processed_images = {}
        self.image_translations = {}
        
        # Configuration from environment
        self.process_webnovel = os.getenv("PROCESS_WEBNOVEL_IMAGES", "1") == "1"
        self.webnovel_min_height = int(os.getenv("WEBNOVEL_MIN_HEIGHT", "1000"))
        self.image_max_tokens = int(os.getenv("IMAGE_MAX_TOKENS", "8192"))

        }
        
    def extract_images_from_chapter(self, chapter_html: str) -> List[Dict]:
        """
        Extract image references from chapter HTML
        
        Returns:
            List of dicts with image info: {src, alt, width, height}
        """
        soup = BeautifulSoup(chapter_html, 'html.parser')
        images = []
        
        for img in soup.find_all('img'):
            img_info = {
                'src': img.get('src', ''),
                'alt': img.get('alt', ''),
                'width': img.get('width'),
                'height': img.get('height'),
                'style': img.get('style', '')
            }
            
            if img_info['src']:
                images.append(img_info)
                
        return images
    
    def should_translate_image(self, image_path: str, check_illustration: bool = True) -> bool:
        """
        Determine if an image should be translated based on various heuristics
        
        Args:
            image_path: Path to the image file
            check_illustration: Whether to check if it's likely an illustration
            
        Returns:
            True if image likely contains translatable text
        """
        # Skip if already processed
        if image_path in self.processed_images:
            return False
            
        # Check file extension - ADD GIF SUPPORT
        ext = os.path.splitext(image_path)[1].lower()
        if ext not in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']:
            return False
            
        # Check file size (skip very small images)
        if os.path.exists(image_path):
            size = os.path.getsize(image_path)
            if size < 5000:  # Less than 5KB (lowered threshold for GIFs)
                return False
                
        # For GIF files from web novels, always process them
        if ext == '.gif' and 'chapter' in os.path.basename(image_path).lower():
            print(f"   📜 Web novel GIF detected: {os.path.basename(image_path)}")
            return True
            
        # Check file size (skip very small images)
        if os.path.exists(image_path):
            size = os.path.getsize(image_path)
            if size < 10000:  # Less than 10KB
                return False
                
        # Check image dimensions
        try:
            with Image.open(image_path) as img:
                width, height = img.size
                # Skip very small images (likely icons)
                if width < 100 or height < 100:
                    return False
                    
                # Calculate aspect ratio
                aspect_ratio = width / height
                
                # Check for web novel/long text images (very tall, narrow images)
                if self.process_webnovel and height > self.webnovel_min_height and aspect_ratio < 0.5:
                    # This is likely a web novel chapter or long text screenshot
                    print(f"   📜 Web novel/long text image detected: {os.path.basename(image_path)}")
                    return True
                    
                # Skip OTHER extreme aspect ratios (but not tall text images)
                if aspect_ratio > 5:  # Very wide images
                    return False
                    
                # Additional check for illustrations (typically larger, square-ish images)
                if check_illustration:
                    # Large images with normal aspect ratios are often illustrations
                    if width > 800 and height > 600 and 0.5 < aspect_ratio < 2:
                        # Check filename for illustration indicators
                        filename = os.path.basename(image_path).lower()
                        illustration_indicators = [
                            'illust', 'illustration', 'art', 'artwork', 'drawing',
                            'painting', 'sketch', 'design', 'visual', 'graphic',
                            'image', 'picture', 'fig', 'figure', 'plate'
                        ]
                        
                        # If filename suggests it's an illustration, skip
                        for indicator in illustration_indicators:
                            if indicator in filename:
                                print(f"   📎 Skipping likely illustration: {filename}")
                                return False
                                
        except Exception:
            return False
            
        # Check filename patterns that suggest text content
        filename = os.path.basename(image_path).lower()
        
        # Strong indicators of text content (including web novel patterns)
        text_indicators = [
            'text', 'title', 'chapter', 'page', 'dialog', 'dialogue',
            'bubble', 'sign', 'note', 'letter', 'message', 'notice',
            'banner', 'caption', 'subtitle', 'heading', 'label',
            'menu', 'interface', 'ui', 'screen', 'display',
            'novel', 'webnovel', 'lightnovel', 'wn', 'ln',  # Web novel indicators
            'chap', 'ch', 'episode', 'ep'  # Chapter indicators
        ]
        
        # Strong indicators to skip
        skip_indicators = [
            'cover', 'logo', 'decoration', 'ornament', 'border',
            'background', 'wallpaper', 'texture', 'pattern',
            'icon', 'button', 'avatar', 'profile', 'portrait',
            'landscape', 'scenery', 'character', 'hero', 'heroine'
        ]
        
        # Check for text indicators
        for indicator in text_indicators:
            if indicator in filename:
                print(f"   📝 Text-likely image detected: {filename}")
                return True
                
        # Check for skip indicators
        for indicator in skip_indicators:
            if indicator in filename:
                print(f"   🎨 Skipping decorative/character image: {filename}")
                return False
        
        # For ambiguous cases, if it's a tall image, assume it might be text
        try:
            with Image.open(image_path) as img:
                width, height = img.size
                if height > width * 2:  # Height is more than twice the width
                    print(f"   📜 Tall image detected, assuming possible text content")
                    return True
        except:
            pass
        
        # Default to False to avoid processing regular illustrations
        return False
    
    def preprocess_image_for_watermarks(self, image_path: str) -> Optional[bytes]:
        """
        Preprocess images to improve text visibility when watermarks are present
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Preprocessed image bytes or None
        """
        try:
            # Open image
            img = Image.open(image_path)
            
            # Convert to RGB if necessary
            if img.mode not in ('RGB', 'RGBA'):
                img = img.convert('RGB')
            
            # Enhance contrast to make text stand out more from watermarks
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.5)  # Increase contrast
            
            # Enhance brightness slightly
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(1.1)  # Slight brightness increase
            
            # Optional: Apply slight sharpening to make text clearer
            img = img.filter(ImageFilter.SHARPEN)
            
            # Convert to bytes
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='PNG')
            img_bytes.seek(0)
            
            return img_bytes.read()
            
        except Exception as e:
            logger.warning(f"Could not preprocess image: {e}")
            return None
        
    def translate_image(self, image_path: str, context: str = "") -> Optional[str]:
        """
        Translate text in an image using vision API - with two-step process
        """
        try:
            print(f"   🔍 translate_image called for: {image_path}")
            
            if not os.path.exists(image_path):
                logger.warning(f"Image not found: {image_path}")
                print(f"   ❌ Image file does not exist!")
                return None
            
            # Special handling for GIF files
            if image_path.lower().endswith('.gif'):
                print(f"   🔧 Converting GIF to PNG for better OCR")
                
                with Image.open(image_path) as img:
                    img.seek(0)
                    if img.mode not in ('RGB', 'RGBA'):
                        img = img.convert('RGB')
                    img_bytes = io.BytesIO()
                    img.save(img_bytes, format='PNG')
                    img_bytes.seek(0)
                    image_data = img_bytes.read()
            else:
                with open(image_path, 'rb') as f:
                    image_data = f.read()
            
            # Determine if it's a long text image
            is_long_text = False
            try:
                with Image.open(image_path) as img:
                    width, height = img.size
                    aspect_ratio = width / height if height > 0 else 1
                    
                    if hasattr(self, 'webnovel_min_height'):
                        min_height = int(getattr(self, 'webnovel_min_height', 1000))
                    else:
                        min_height = 1000
                        
                    is_long_text = height > min_height and aspect_ratio < 0.5
                    print(f"   📐 Image: {width}x{height}, aspect: {aspect_ratio:.2f}, long_text: {is_long_text}")
                    
            except Exception as e:
                print(f"   ⚠️ Error checking image dimensions: {e}")
                is_long_text = False
            
            # Use appropriate token limit
            max_tokens = self.image_max_tokens if is_long_text else 2048
            
            # STEP 1: OCR - Extract text only
            print(f"📋 Step 1: Extracting text via OCR...")
            
            ocr_instructions = f"""
    Extract ALL text from this image EXACTLY as it appears.
    Do NOT translate - output the original text only.
    Maintain the original formatting and line breaks.
    If there's no text in the image, respond with "NO_TEXT_FOUND".
    For web novel or long text images: Extract ALL visible text from top to bottom.
    Output the raw text only, no explanations."""

            ocr_messages = [
                {"role": "system", "content": ocr_instructions},
                {"role": "user", "content": "Extract all text from this image. Output ONLY the original text, no translation."}
            ]
            
            # Use vision API for OCR
            ocr_response, ocr_finish = self.client.send_image(
                ocr_messages,
                image_data,
                temperature=0.1,
                max_tokens=max_tokens,
                context='image_ocr'
            )
            
            # Check if no text was found
            if "NO_TEXT_FOUND" in ocr_response or not ocr_response.strip():
                print(f"   ℹ️ No text detected in image")
                return None
            
            print(f"   ✅ OCR extracted {len(ocr_response)} characters")
            
            # Save OCR result for debugging
            ocr_filename = f"ocr_{os.path.basename(image_path)}.txt"
            ocr_filepath = os.path.join(self.translated_images_dir, ocr_filename)
            try:
                with open(ocr_filepath, 'w', encoding='utf-8') as f:
                    f.write(ocr_response)
                print(f"   💾 Saved OCR text to: {ocr_filename}")
            except Exception as e:
                print(f"   ⚠️ Could not save OCR file: {e}")
            
            # STEP 2: Translate the extracted text using text API
            print(f"📝 Step 2: Translating extracted text...")
            
            # Build translation messages with the system prompt
            translation_messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": ocr_response}
            ]
            
            # Use regular text translation
            try:
                translation_response, trans_finish = self.client.send(
                    translation_messages,
                    temperature=0.3,
                    max_tokens=max_tokens,
                    context='translation'  # Use translation context
                )
                
                translated_text = translation_response.strip()
                
            except Exception as e:
                print(f"   ❌ Translation failed: {e}")
                translated_text = f"[Translation Error: {str(e)}]"
                trans_finish = "error"
            
            if not translated_text:
                print(f"   ❌ Translation returned empty result")
                return None
                
            print(f"   ✅ Translation completed ({len(translated_text)} characters)")
            
            # Check if translation was truncated
            if trans_finish in ["length", "max_tokens"]:
                print(f"   ⚠️ Translation was TRUNCATED! Consider increasing Max tokens.")
                translated_text += "\n\n[TRANSLATION TRUNCATED DUE TO TOKEN LIMIT]"
            
            # Clean any AI artifacts
            if translated_text.startswith(('Sure', 'Here', "I'll translate", 'Certainly')):
                lines = translated_text.split('\n')
                if len(lines) > 1:
                    translated_text = '\n'.join(lines[1:]).strip()
            
            # Store the result for caching
            self.processed_images[image_path] = translated_text
            
            # Create HTML output
            img_rel_path = os.path.relpath(image_path, self.output_dir)
            
            # For long text images, make the original collapsible
            if is_long_text:
                html_output = f"""<div class="image-with-translation webnovel-image">
        <details>
            <summary>📖 View Original Image</summary>
            <img src="{img_rel_path}" alt="Original image" />
        </details>
        <div class="image-translation">
            <p><em>[Image text translation:]</em></p>
            {self._format_translation_as_html(translated_text)}
        </div>
    </div>"""
            else:
                html_output = f"""<div class="image-with-translation">
        <img src="{img_rel_path}" alt="Original image" />
        <div class="image-translation">
            <p><em>[Image text translation:]</em></p>
            {self._format_translation_as_html(translated_text)}
        </div>
    </div>"""
            
            return html_output
            
        except Exception as e:
            logger.error(f"Error translating image {image_path}: {e}")
            print(f"   ❌ Exception in translate_image: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _format_translation_as_html(self, text: str) -> str:
        """Format translated text as HTML paragraphs"""
        # Split by double newlines for paragraphs
        paragraphs = text.split('\n\n')
        html_parts = []
        
        for para in paragraphs:
            para = para.strip()
            if para:
                # Check if it's dialogue (starts with quotes)
                if para.startswith(('"', '"', '「', '『', '"')):
                    html_parts.append(f'<p class="dialogue">{para}</p>')
                else:
                    html_parts.append(f'<p>{para}</p>')
        
        return '\n'.join(html_parts)
    
    def update_chapter_with_translated_images(self, chapter_html: str, image_translations: Dict[str, str]) -> str:
        """
        Update chapter HTML to include image translations
        
        Args:
            chapter_html: Original chapter HTML
            image_translations: Dict mapping original image paths to translation HTML
            
        Returns:
            Updated HTML
        """
        soup = BeautifulSoup(chapter_html, 'html.parser')
        
        for img in soup.find_all('img'):
            src = img.get('src', '')
            if src in image_translations:
                # Replace the img tag with the translation HTML
                translation_html = image_translations[src]
                new_element = BeautifulSoup(translation_html, 'html.parser')
                img.replace_with(new_element)
                
        return str(soup)
    
    def save_translation_log(self, chapter_num: int, translations: Dict[str, str]):
        """
        Save a log of all translations for a chapter
        
        Args:
            chapter_num: Chapter number
            translations: Dict of image path to translated text
        """
        if not translations:
            return
            
        log_dir = os.path.join(self.translated_images_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        log_file = os.path.join(log_dir, f'chapter_{chapter_num}_translations.json')
        
        log_data = {
            'chapter': chapter_num,
            'timestamp': os.environ.get('TZ', 'UTC'),
            'translations': {}
        }
        
        for img_path, translation in translations.items():
            # Extract just the text from HTML if needed
            if '<div class="image-translation">' in translation:
                soup = BeautifulSoup(translation, 'html.parser')
                text_div = soup.find('div', class_='image-translation')
                if text_div:
                    # Remove the header paragraph
                    header = text_div.find('p')
                    if header and '[Image text translation:]' in header.text:
                        header.decompose()
                    text = text_div.get_text(separator='\n').strip()
                else:
                    text = translation
            else:
                text = translation
                
            log_data['translations'][os.path.basename(img_path)] = text
        
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, ensure_ascii=False, indent=2)
            
        print(f"   📝 Saved translation log: {os.path.basename(log_file)}")
