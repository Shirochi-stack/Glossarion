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
    def __init__(self, client, output_dir: str, profile_name: str = "", system_prompt: str = "", temperature: float = 0.3):
        """
        Initialize the image translator
        
        Args:
            client: UnifiedClient instance for API calls
            output_dir: Directory to save translated images
            profile_name: Source language for translation
            system_prompt: System prompt from GUI to use for translation
        """
        self.client = client
        self.output_dir = output_dir
        self.profile_name = profile_name
        self.system_prompt = system_prompt
        self.temperature = temperature  # Add this
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
        self.chunk_height = int(os.getenv("IMAGE_CHUNK_HEIGHT", "2000"))


        
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
            
    def translate_image(self, image_path: str, context: str = "", check_stop_fn=None) -> Optional[str]:
        """
        Translate text in an image using vision API - with chunking for tall images and stop support
        """
        try:
            print(f"   🔍 translate_image called for: {image_path}")
            
            # Check for stop at the beginning
            if check_stop_fn and check_stop_fn():
                print("   ❌ Image translation stopped by user")
                return None
            
            if not os.path.exists(image_path):
                logger.warning(f"Image not found: {image_path}")
                print(f"   ❌ Image file does not exist!")
                return None
            
            # Open and process the image
            with Image.open(image_path) as img:
                width, height = img.size
                aspect_ratio = width / height if height > 0 else 1
                print(f"   📐 Image dimensions: {width}x{height}, aspect ratio: {aspect_ratio:.2f}")
                
                # Convert to RGB if necessary (for GIF and other formats)
                if img.mode not in ('RGB', 'RGBA'):
                    img = img.convert('RGB')
                
                # Determine if it's a long text image
                is_long_text = height > self.webnovel_min_height and aspect_ratio < 0.5
                
                # Set max height for chunks
                MAX_HEIGHT = self.chunk_height  # Maximum height per chunk for good OCR
                
                all_translations = []
                was_stopped = False
                
                if height > MAX_HEIGHT:
                    # Image needs to be split into chunks
                    print(f"   ✂️ Image too tall ({height}px), splitting into chunks of {MAX_HEIGHT}px...")
                    
                    # Calculate number of chunks needed
                    num_chunks = (height + MAX_HEIGHT - 1) // MAX_HEIGHT
                    overlap = 100  # Pixels of overlap between chunks to avoid cutting text
                    
                    for i in range(num_chunks):
                        # Check for stop before processing each chunk
                        if check_stop_fn and check_stop_fn():
                            print(f"   ❌ Stopped at chunk {i+1}/{num_chunks}")
                            was_stopped = True
                            break
                        
                        # Calculate chunk boundaries with overlap
                        start_y = max(0, i * MAX_HEIGHT - (overlap if i > 0 else 0))
                        end_y = min(height, (i + 1) * MAX_HEIGHT)
                        
                        print(f"   📄 Processing chunk {i+1}/{num_chunks} (y: {start_y}-{end_y})")
                        
                        # Crop the chunk
                        chunk = img.crop((0, start_y, width, end_y))
                        
                        # Convert chunk to bytes
                        chunk_bytes = io.BytesIO()
                        chunk.save(chunk_bytes, format='PNG', optimize=False)
                        chunk_bytes.seek(0)
                        chunk_data = chunk_bytes.read()
                        
                        # Build messages for this chunk
                        chunk_context = f"This is part {i+1} of {num_chunks} of a longer image. {context}"
                        messages = [
                            {"role": "system", "content": self.system_prompt},
                            {"role": "user", "content": context if context else ""}
                        ]
                        # Translate this chunk
                        try:
                            print(f"   🔄 Calling vision API for chunk {i+1}...")
                            
                            # Check for stop right before API call
                            if check_stop_fn and check_stop_fn():
                                print(f"   ❌ Stopped before API call for chunk {i+1}")
                                was_stopped = True
                                break
                            
                            translation_response, trans_finish = self.client.send_image(
                                messages,
                                chunk_data,
                                temperature=self.temperature,
                                max_tokens=self.image_max_tokens,
                                context='image_translation'
                            )
                            
                            if translation_response and translation_response.strip():
                                # Clean AI artifacts from chunk
                                chunk_text = translation_response.strip()
                                if chunk_text.startswith(('Sure', 'Here', "I'll translate", 'Certainly')):
                                    lines = chunk_text.split('\n')
                                    if len(lines) > 1:
                                        chunk_text = '\n'.join(lines[1:]).strip()
                                
                                all_translations.append(chunk_text)
                                print(f"   ✅ Chunk {i+1} translated ({len(chunk_text)} chars)")
                                
                                # Check if truncated
                                if trans_finish in ["length", "max_tokens"]:
                                    print(f"   ⚠️ Chunk {i+1} was TRUNCATED!")
                                    
                            else:
                                print(f"   ⚠️ Chunk {i+1} returned no text")
                                
                        except Exception as e:
                            print(f"   ❌ Error translating chunk {i+1}: {e}")
                            all_translations.append(f"[Error in chunk {i+1}: {str(e)}]")
                        
                        # Check for stop after processing chunk
                        if check_stop_fn and check_stop_fn():
                            print(f"   ❌ Stopped after chunk {i+1}/{num_chunks}")
                            was_stopped = True
                            break
                        
                        # Small delay between chunks to avoid rate limiting
                        if i < num_chunks - 1:
                            # Check for stop during delay (0.5s in 0.1s increments)
                            for _ in range(5):
                                if check_stop_fn and check_stop_fn():
                                    print("   ❌ Stopped during chunk delay")
                                    was_stopped = True
                                    break
                                time.sleep(0.1)
                            
                            if was_stopped:
                                break
                    
                    # Combine all chunk translations
                    if all_translations:
                        translated_text = "\n\n".join(all_translations)
                        if was_stopped:
                            translated_text += "\n\n[TRANSLATION STOPPED BY USER]"
                        print(f"   ✅ Combined {len(all_translations)} chunks into final translation")
                    else:
                        print(f"   ❌ No successful translations from any chunks")
                        return None
                        
                else:
                    # Image is small enough to process in one go
                    print(f"   👍 Image height OK ({height}px), processing as single image...")
                    
                    # Check for stop before processing
                    if check_stop_fn and check_stop_fn():
                        print("   ❌ Image translation stopped by user")
                        return None
                    
                    # Convert entire image to bytes
                    img_bytes = io.BytesIO()
                    img.save(img_bytes, format='PNG', optimize=False)
                    img_bytes.seek(0)
                    image_data = img_bytes.read()
                    
                    # Build messages
                    messages = [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": f"Translate all text in this image from {self.profile_name} to English. {context}"}
                    ]
                    
                    # Single API call for translation
                    try:
                        print(f"   🔄 Calling vision API...")
                        print(f"   📊 Using temperature: {self.temperature}")
                        print(f"   📊 Max tokens: {self.image_max_tokens}")
                        
                        # Final stop check before API call
                        if check_stop_fn and check_stop_fn():
                            print("   ❌ Stopped before API call")
                            return None
                        
                        translation_response, trans_finish = self.client.send_image(
                            messages,
                            image_data,
                            temperature=self.temperature,
                            max_tokens=self.image_max_tokens,
                            context='image_translation'
                        )
                        
                        print(f"   📡 API response received, finish_reason: {trans_finish}")
                        translated_text = translation_response.strip()
                        
                        # Check if translation was truncated
                        if trans_finish in ["length", "max_tokens"]:
                            print(f"   ⚠️ Translation was TRUNCATED! Consider increasing Max tokens.")
                            translated_text += "\n\n[TRANSLATION TRUNCATED DUE TO TOKEN LIMIT]"
                            
                    except Exception as e:
                        print(f"   ❌ Translation failed: {e}")
                        print(f"   ❌ Error type: {type(e).__name__}")
                        translated_text = f"[Translation Error: {str(e)}]"
            
            # Check if we got any translation
            if not translated_text or not translated_text.strip():
                print(f"   ❌ Translation returned empty result")
                return None
            
            # Clean any remaining AI artifacts (unless stopped)
            if not was_stopped and translated_text.startswith(('Sure', 'Here', "I'll translate", 'Certainly')):
                lines = translated_text.split('\n')
                if len(lines) > 1:
                    translated_text = '\n'.join(lines[1:]).strip()
            
            print(f"   ✅ Final translation completed ({len(translated_text)} characters)")
            
            # Store the result for caching
            self.processed_images[image_path] = translated_text
            
            # Save translation for debugging
            trans_filename = f"translated_{os.path.basename(image_path)}.txt"
            trans_filepath = os.path.join(self.translated_images_dir, trans_filename)
            try:
                with open(trans_filepath, 'w', encoding='utf-8') as f:
                    f.write(translated_text)
                print(f"   💾 Saved translation to: {trans_filename}")
            except Exception as e:
                print(f"   ⚠️ Could not save translation file: {e}")
            
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
            <p><em>[Image text translation{' (partial)' if was_stopped else ''}:]</em></p>
            {self._format_translation_as_html(translated_text)}
        </div>
    </div>"""
            else:
                html_output = f"""<div class="image-with-translation">
        <img src="{img_rel_path}" alt="Original image" />
        <div class="image-translation">
            <p><em>[Image text translation{' (partial)' if was_stopped else ''}:]</em></p>
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
