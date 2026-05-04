"""
Image Translation Module for EPUB Translator
Handles detection, extraction, and translation of images containing text
Includes support for web novel images and watermark handling
"""

import os
import json
import base64
import zipfile
import hashlib
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter
import io
from typing import List, Dict, Optional, Tuple
import re
from bs4 import BeautifulSoup
import logging
import time
import queue
import threading
import sys
import unicodedata
# OpenCV availability check
try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
import numpy as np
from unified_api_client import UnifiedClientError

logger = logging.getLogger(__name__)

DEFAULT_VISION_OCR_PROMPT = (
    "Extract only the readable text that is physically present in the image, in natural reading order. "
    "If the image itself is a cover page, character art, scene illustration, decorative image, or otherwise not a page of readable story text, reply with exactly: No. "
    "Return only the base source text. Preserve paragraph breaks and intentional textual layout when they affect meaning, but do not reproduce every visual wrap from the image; merge wrapped lines that belong to the same sentence or paragraph. "
    "Preserve visible textual styling and marks when possible, including brackets, parentheses, quote marks, emphasis, strikethrough/deleted text, symbols, and emotes/emoticons. "
    "For Chinese/Japanese/Korean text with small pronunciation guides above or beside the main characters, OCR only the main/base characters and ignore the pronunciation guides. "
    "For pinyin-over-Chinese images, output the Chinese characters only; do not output the pinyin unless the pinyin is standalone text with no matching Chinese base text. "
    "Do not translate, summarize, explain, annotate, transliterate, romanize, or add pronunciation guides. "
    "Do not output duplicate reading lines such as pinyin, romaji, furigana, Jyutping, or Latin readings when they are attached to the same base text. "
    "If no readable story text is present, reply with exactly: No."
)

STALE_VISION_OCR_PROMPT_MARKERS = (
    "Preserve the original line breaks as faithfully as possible",
    "do not collapse separate visual lines into one paragraph",
)

DEFAULT_VISION_OCR_USER_PROMPT = (
    "OCR this image/chunk. Return only the main/base source text. "
    "If this image/chunk is cover art, an illustration, decorative art, or has no readable story text, reply with exactly: No. "
    "Ignore pinyin/romaji/furigana/Jyutping pronunciation guides attached to base characters. Do not translate."
    "\n\nContext:\n{context}"
)

def requires_cv2(func):
    """Decorator to skip methods that require OpenCV"""
    def wrapper(self, *args, **kwargs):
        if not CV2_AVAILABLE:
            # Return sensible defaults based on the function
            if func.__name__ == '_detect_watermark_pattern':
                return False, None
            elif func.__name__ in ['_remove_periodic_watermark', 
                                  '_adaptive_histogram_equalization',
                                  '_bilateral_filter',
                                  '_enhance_text_regions']:
                # Return the image array unchanged
                return args[0] if args else None
            else:
                return None
        return func(self, *args, **kwargs)
    return wrapper
    
def send_image_with_interrupt(client, messages, image_data, temperature, max_tokens, stop_check_fn, chunk_timeout=None, context='image_translation'):
    """Send image API request with interrupt capability and timeout retry"""
    import queue
    import threading
    from unified_api_client import UnifiedClientError
    
    result_queue = queue.Queue()
    
    def api_call():
        try:
            start_time = time.time()
            result = client.send_image(messages, image_data, temperature=temperature, 
                                     max_tokens=max_tokens, context=context)
            elapsed = time.time() - start_time
            result_queue.put((result, elapsed))
        except Exception as e:
            result_queue.put(e)
    
    api_thread = threading.Thread(target=api_call)
    api_thread.daemon = True
    api_thread.start()
    
    # Respect caller-provided timeout; if None, wait indefinitely
    timeout = chunk_timeout
    check_interval = 0.5
    elapsed = 0

    def _should_interrupt_wait():
        # Graceful stop lets in-flight API calls finish; immediate stop sets
        # TRANSLATION_CANCELLED and should abort the wait promptly.
        if os.environ.get("GRACEFUL_STOP") == "1":
            return os.environ.get("TRANSLATION_CANCELLED") == "1"
        if os.environ.get("TRANSLATION_CANCELLED") == "1":
            return True
        return bool(stop_check_fn and stop_check_fn())
    
    while True:
        try:
            result = result_queue.get(timeout=check_interval)
            if isinstance(result, Exception):
                raise result
            if isinstance(result, tuple):
                api_result, api_time = result
                # Check if it took too long
                if chunk_timeout is not None and api_time > chunk_timeout:
                    raise UnifiedClientError(f"Image API call took {api_time:.1f}s (timeout: {chunk_timeout}s)")
                return api_result
            return result
        except queue.Empty:
            if _should_interrupt_wait():
                raise UnifiedClientError("Image translation stopped by user", error_type="cancelled")
            elapsed += check_interval
            if chunk_timeout is not None and elapsed >= chunk_timeout:
                raise UnifiedClientError(f"Image API call timed out after {chunk_timeout} seconds")

def _clear_vision_key_context_for_text_request(client, context):
    """Prevent Vision-key pool state from leaking into normal text/glossary calls."""
    if context in ('vision_ocr', 'image_ocr', 'image_scan'):
        return
    try:
        from unified_api_client import UnifiedClient
        vision_pool = getattr(UnifiedClient, '_qa_scan_key_pool', None)
        if vision_pool is not None and getattr(client, '_api_key_pool', None) is vision_pool:
            if '_api_key_pool' in getattr(client, '__dict__', {}):
                del client.__dict__['_api_key_pool']
            try:
                client._multi_key_mode = bool(getattr(client, 'use_multi_keys', client._multi_key_mode))
            except Exception:
                pass
        try:
            if vision_pool and hasattr(vision_pool, 'release_thread_assignment'):
                vision_pool.release_thread_assignment()
        except Exception:
            pass
    except Exception:
        pass


def send_text_with_interrupt(client, messages, temperature, max_tokens, stop_check_fn, chunk_timeout=None, context='translation'):
    """Send text API request with graceful/force stop behavior matching image calls."""
    import queue
    import threading
    from unified_api_client import UnifiedClient, UnifiedClientError

    result_queue = queue.Queue()

    def api_call():
        try:
            _clear_vision_key_context_for_text_request(client, context)
            start_time = time.time()
            result = client.send(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                context=context,
            )
            elapsed = time.time() - start_time
            result_queue.put((result, elapsed))
        except Exception as e:
            result_queue.put(e)

    api_thread = threading.Thread(target=api_call)
    api_thread.daemon = True
    api_thread.start()

    check_interval = 0.5
    elapsed = 0

    def _force_cancel():
        try:
            if hasattr(client, 'cancel_current_operation'):
                client.cancel_current_operation()
        except Exception:
            pass
        try:
            UnifiedClient.hard_cancel_all()
        except Exception:
            pass

    def _should_interrupt_wait():
        if os.environ.get("TRANSLATION_CANCELLED") == "1":
            _force_cancel()
            return True
        if os.environ.get("GRACEFUL_STOP") == "1":
            return False
        return bool(stop_check_fn and stop_check_fn())

    while True:
        try:
            result = result_queue.get(timeout=check_interval)
            if isinstance(result, Exception):
                raise result
            if isinstance(result, tuple):
                api_result, api_time = result
                if chunk_timeout is not None and api_time > chunk_timeout:
                    raise UnifiedClientError(f"Text API call took {api_time:.1f}s (timeout: {chunk_timeout}s)")
                return api_result
            return result
        except queue.Empty:
            if _should_interrupt_wait():
                raise UnifiedClientError("Text translation stopped by user", error_type="cancelled")
            elapsed += check_interval
            if chunk_timeout is not None and elapsed >= chunk_timeout:
                raise UnifiedClientError(f"Text API call timed out after {chunk_timeout} seconds")

class ImageTranslator:
    def __init__(self, client, output_dir: str, profile_name: str = "", system_prompt: str = "", 
                 temperature: float = 0.3, log_callback=None, progress_manager=None,
                 history_manager=None, chunk_context_manager=None):
        """
        Initialize the image translator
        
        Args:
            client: UnifiedClient instance for API calls
            output_dir: Directory to save translated images
            profile_name: Source language for translation
            system_prompt: System prompt from GUI to use for translation
            temperature: Temperature for translation
            log_callback: Optional callback function for logging
            progress_manager: Shared ProgressManager instance for synchronization
        """
        self.client = client
        self.output_dir = output_dir
        self.profile_name = profile_name
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.log_callback = log_callback
        self.progress_manager = progress_manager  # Use shared progress manager
        self.images_dir = os.path.join(output_dir, "images")
        self.ocr_dir = os.path.join(output_dir, "OCR")
        # Backward-compatible alias for callers that still reference translated_images_dir.
        self.translated_images_dir = self.ocr_dir
        os.makedirs(self.ocr_dir, exist_ok=True)
        self.api_delay = float(os.getenv("SEND_INTERVAL_SECONDS", "2"))
        
        # Track processed images to avoid duplicates
        self.processed_images = {}
        self.image_translations = {}
        
        # Configuration from environment
        self.process_webnovel = os.getenv("PROCESS_WEBNOVEL_IMAGES", "1") == "1"
        self.webnovel_min_height = int(os.getenv("WEBNOVEL_MIN_HEIGHT", "1000"))
        self.image_max_tokens = int(os.getenv("MAX_OUTPUT_TOKENS", "8192"))
        self.chunk_height = int(os.getenv("IMAGE_CHUNK_HEIGHT", "2000"))
        self.vision_ocr_prompt = os.getenv("VISION_OCR_PROMPT", DEFAULT_VISION_OCR_PROMPT).strip() or DEFAULT_VISION_OCR_PROMPT
        if any(marker in self.vision_ocr_prompt for marker in STALE_VISION_OCR_PROMPT_MARKERS):
            self.vision_ocr_prompt = DEFAULT_VISION_OCR_PROMPT
        self.vision_ocr_user_prompt = os.getenv("VISION_OCR_USER_PROMPT", DEFAULT_VISION_OCR_USER_PROMPT).strip() or DEFAULT_VISION_OCR_USER_PROMPT
        self.last_vision_translation_finish_reason = None
        self.last_vision_translation_error = None
        self._vision_glossary_processed_hashes = set()
        self._ensure_ocr_cache_valid()
        
        # DEBUG: Log the actual max tokens being used for image translation
        print(f"🔍 ImageTranslator initialized with max_output_tokens: {self.image_max_tokens}")
        print(f"🔍 Environment MAX_OUTPUT_TOKENS: {os.getenv('MAX_OUTPUT_TOKENS', 'NOT SET')}")
        
        # Add context tracking for image chunks
        self.contextual_enabled = os.getenv("CONTEXTUAL", "1") == "1"
        self.history_manager = history_manager
        self.chunk_context_manager = chunk_context_manager
        _raw = os.getenv("REMOVE_AI_ARTIFACTS", "off")
        if _raw == "0": _raw = "off"
        elif _raw == "1": _raw = "medium"
        self.remove_ai_artifacts = _raw if _raw in ("off", "low", "medium", "high") else "off"

        
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

    def compress_image(self, image_path):
        """
        Compress an image based on settings from environment variables
        
        Args:
            image_path: Path to the input image
            
        Returns:
            Path to compressed image (temporary or saved)
        """
        try:
            # Check if compression is enabled
            if os.getenv("ENABLE_IMAGE_COMPRESSION", "0") != "1":
                return image_path  # Return original if compression disabled
            
            print(f"   🗜️ Compressing image: {os.path.basename(image_path)}")
            
            # Load compression settings from environment
            target_format = os.getenv("IMAGE_COMPRESSION_FORMAT", "auto")
            max_dimension = int(os.getenv("MAX_IMAGE_DIMENSION", "2048"))
            max_size_mb = float(os.getenv("MAX_IMAGE_SIZE_MB", "10"))
            
            quality_settings = {
                'webp': int(os.getenv("WEBP_QUALITY", "85")),
                'jpeg': int(os.getenv("JPEG_QUALITY", "85")),
                'png': int(os.getenv("PNG_COMPRESSION", "6"))
            }
            
            auto_compress = os.getenv("AUTO_COMPRESS_ENABLED", "1") == "1"
            preserve_transparency = os.getenv("PRESERVE_TRANSPARENCY", "0") == "1"  # Default is now False
            preserve_original_format = os.getenv("PRESERVE_ORIGINAL_FORMAT", "0") == "1"  # New option
            optimize_for_ocr = os.getenv("OPTIMIZE_FOR_OCR", "1") == "1"
            progressive = os.getenv("PROGRESSIVE_ENCODING", "1") == "1"
            save_compressed = os.getenv("SAVE_COMPRESSED_IMAGES", "0") == "1"
            
            # Open image
            with Image.open(image_path) as img:
                original_format = img.format.lower() if img.format else 'png'
                has_transparency = img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info)
                
                # Special handling for GIF files
                is_gif = original_format == 'gif'
                if is_gif and not preserve_original_format:
                    print(f"   🎞️ GIF detected - converting to static image for better compression")
                    # For animated GIFs, we'll take the first frame
                    # Convert to RGBA to preserve any transparency
                    if img.mode == 'P' and 'transparency' in img.info:
                        img = img.convert('RGBA')
                    elif img.mode not in ('RGB', 'RGBA'):
                        img = img.convert('RGB')
                elif is_gif and preserve_original_format:
                    print(f"   🎞️ GIF detected - preserving original format as requested")
                
                # Calculate original size
                original_size_mb = os.path.getsize(image_path) / (1024 * 1024)
                print(f"   📊 Original: {img.width}x{img.height}, {original_size_mb:.2f}MB, format: {original_format}")
                
                # Get chunk height from environment - this comes from the GUI setting
                chunk_height = int(os.getenv("IMAGE_CHUNK_HEIGHT", "1500"))
                print(f"   📏 Using chunk height from settings: {chunk_height}px")
                
                # Check if resizing is needed - BUT NOT FOR TALL IMAGES THAT WILL BE CHUNKED!
                needs_resize = img.width > max_dimension or img.height > max_dimension
                
                # CRITICAL: Check if this is a tall image that will be chunked
                # If so, DO NOT resize the height!
                is_tall_text_image = img.height > chunk_height
                
                if needs_resize:
                    if is_tall_text_image:
                        # Only resize width if needed, NEVER touch the height for tall images
                        if img.width > max_dimension:
                            # Keep aspect ratio but don't exceed max width
                            ratio = max_dimension / img.width
                            new_width = max_dimension
                            new_height = int(img.height * ratio)
                            print(f"   ⚠️ Tall image ({img.height}px > chunk height {chunk_height}px)")
                            print(f"   📐 Resizing width only: {img.width} → {new_width} (height: {img.height} → {new_height})")
                            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                        else:
                            print(f"   ✅ Tall image ({img.height}px) - keeping dimensions (will be chunked into {(img.height + chunk_height - 1) // chunk_height} chunks)")
                    else:
                        # Normal resize for regular images (not tall enough to chunk)
                        ratio = min(max_dimension / img.width, max_dimension / img.height)
                        new_size = (int(img.width * ratio), int(img.height * ratio))
                        img = img.resize(new_size, Image.Resampling.LANCZOS)
                        print(f"   📐 Regular image resized to: {new_size[0]}x{new_size[1]}")
                
                # Auto-select format if needed
                if preserve_original_format and target_format == 'auto':
                    # Keep the original format
                    target_format = original_format
                    # Special handling for formats that might not be ideal
                    if original_format == 'bmp':
                        target_format = 'png'  # Convert BMP to PNG as BMP is uncompressed
                    print(f"   📸 Preserving original format: {target_format}")
                elif target_format == 'auto':
                    # For GIFs with text (web novel chapters), prefer PNG or WebP
                    if is_gif:
                        if has_transparency and preserve_transparency:
                            target_format = 'png'  # Better for text with transparency
                        else:
                            target_format = 'webp'  # Good compression for text
                    elif has_transparency and preserve_transparency:
                        target_format = 'webp'
                    elif optimize_for_ocr and img.width * img.height > 1000000:
                        target_format = 'webp'
                    elif original_size_mb > 5:
                        target_format = 'webp'
                    else:
                        target_format = 'jpeg'
                    print(f"   🎯 Auto-selected format: {target_format}")
                
                # Handle transparency conversion if needed
                if target_format == 'jpeg' and (has_transparency or img.mode == 'RGBA'):
                    # Convert to RGB with white background
                    rgb_img = Image.new('RGB', img.size, (255, 255, 255))
                    if img.mode == 'RGBA':
                        rgb_img.paste(img, mask=img.split()[3])
                    else:
                        rgb_img.paste(img)
                    img = rgb_img
                
                # Apply OCR optimization if enabled
                if optimize_for_ocr:
                    # Skip OCR optimization for GIF files in palette mode when preserving format
                    if target_format == 'gif' and img.mode in ('P', 'L'):
                        print(f"   ⚠️ Applying OCR optimization to GIF (converting modes temporarily)")
                        # Convert to RGB temporarily for enhancement, then convert back
                        original_mode = img.mode
                        transparency_info = None
                        
                        if img.mode == 'P':
                            # Preserve transparency info if present
                            transparency_info = img.info.get('transparency', None)
                            # Convert to RGBA if has transparency, otherwise RGB
                            if transparency_info is not None:
                                img = img.convert('RGBA')
                            else:
                                img = img.convert('RGB')
                        elif img.mode == 'L':
                            img = img.convert('RGB')
                        
                        # Apply enhancements
                        from PIL import ImageEnhance
                        enhancer = ImageEnhance.Contrast(img)
                        img = enhancer.enhance(1.2)
                        enhancer = ImageEnhance.Sharpness(img)
                        img = enhancer.enhance(1.1)
                        
                        # Extra sharpening for GIF text
                        img = enhancer.enhance(1.2)
                        
                        # Convert back to original mode for GIF saving
                        if original_mode == 'P':
                            # Quantize back to palette mode
                            img = img.quantize(colors=256, method=2)  # MEDIANCUT
                            if transparency_info is not None:
                                img.info['transparency'] = transparency_info
                        elif original_mode == 'L':
                            img = img.convert('L')
                    else:
                        # Normal OCR optimization for non-GIF formats or RGB-mode images
                        from PIL import ImageEnhance
                        enhancer = ImageEnhance.Contrast(img)
                        img = enhancer.enhance(1.2)
                        enhancer = ImageEnhance.Sharpness(img)
                        img = enhancer.enhance(1.1)
                        
                        # Extra sharpening for GIF text which might be lower quality
                        if is_gif:
                            img = enhancer.enhance(1.2)
                
                # Prepare save parameters based on format
                save_params = {}
                
                if target_format == 'webp':
                    # For WebP, decide whether to keep transparency
                    if has_transparency and preserve_transparency:
                        save_params = {
                            'format': 'WEBP',
                            'quality': quality_settings['webp'],
                            'method': 6,
                            'lossless': False,
                            'exact': True  # Preserve transparency
                        }
                    else:
                        # Convert to RGB with white background for WebP without transparency
                        if img.mode in ('RGBA', 'LA', 'P'):
                            rgb_img = Image.new('RGB', img.size, (255, 255, 255))
                            if img.mode == 'RGBA':
                                rgb_img.paste(img, mask=img.split()[3])
                            elif img.mode == 'LA':
                                rgb_img.paste(img, mask=img.split()[1])
                            else:  # P mode
                                if 'transparency' in img.info:
                                    img = img.convert('RGBA')
                                    rgb_img.paste(img, mask=img.split()[3])
                                else:
                                    rgb_img.paste(img)
                            img = rgb_img
                        
                        save_params = {
                            'format': 'WEBP',
                            'quality': quality_settings['webp'],
                            'method': 6,
                            'lossless': False
                        }
                        
                elif target_format == 'jpeg':
                    save_params = {
                        'format': 'JPEG',
                        'quality': quality_settings['jpeg'],
                        'optimize': True,
                        'progressive': progressive
                    }
                    
                elif target_format == 'png':
                    # For PNG, handle transparency properly
                    if not (has_transparency and preserve_transparency):
                        # Convert to RGB with white background if not preserving transparency
                        if img.mode in ('RGBA', 'LA', 'P'):
                            rgb_img = Image.new('RGB', img.size, (255, 255, 255))
                            if img.mode == 'RGBA':
                                rgb_img.paste(img, mask=img.split()[3])
                            elif img.mode == 'LA':
                                rgb_img.paste(img, mask=img.split()[1])
                            else:  # P mode
                                if 'transparency' in img.info:
                                    img = img.convert('RGBA')
                                    rgb_img.paste(img, mask=img.split()[3])
                                else:
                                    rgb_img.paste(img)
                            img = rgb_img
                    elif img.mode == 'P' and 'transparency' in img.info:
                        # Convert palette mode with transparency to RGBA
                        img = img.convert('RGBA')
                    
                    save_params = {
                        'format': 'PNG',
                        'compress_level': quality_settings['png'],
                        'optimize': True
                    }
                
                elif target_format == 'gif':
                    # GIF format - limited but preserving original when requested
                    print(f"   ⚠️ Warning: GIF format has limited colors (256) and may reduce text quality")
                    if img.mode not in ('P', 'L'):
                        # Convert to palette mode for GIF
                        img = img.quantize(colors=256, method=2)  # MEDIANCUT method
                    
                    save_params = {
                        'format': 'GIF',
                        'optimize': True
                    }
                
                # Auto-compress to meet token target if specified
                if auto_compress:
                    target_tokens = int(os.getenv("TARGET_IMAGE_TOKENS", "1000"))
                    # For text-heavy images (like web novel GIFs), be less aggressive
                    if is_gif or 'chapter' in os.path.basename(image_path).lower():
                        target_mb = min(max_size_mb, 3.0)  # Allow up to 3MB for text clarity
                    else:
                        target_mb = min(max_size_mb, 2.0)  # Regular images
                    print(f"   🎯 Auto-compress target: {target_mb:.1f}MB for token efficiency")
                    max_size_mb = target_mb
                
                # Save compressed image
                output_path = None
                quality = save_params.get('quality', 85)
                
                # Try different quality levels to meet size target
                while quality > 10:
                    from io import BytesIO
                    buffer = BytesIO()
                    
                    if 'quality' in save_params:
                        save_params['quality'] = quality
                    
                    img.save(buffer, **save_params)
                    compressed_size_mb = len(buffer.getvalue()) / (1024 * 1024)
                    
                    if compressed_size_mb <= max_size_mb or quality <= 10:
                        # Size is acceptable or we've reached minimum quality
                        if save_compressed:
                            # FIXED: Handle PyInstaller paths properly
                            try:
                                # Try to determine the proper output directory
                                # First check if self.output_dir is absolute and exists
                                if hasattr(self, 'output_dir') and self.output_dir and os.path.isabs(self.output_dir):
                                    base_output_dir = self.output_dir
                                else:
                                    # Fall back to using the directory of the source image
                                    base_output_dir = os.path.dirname(image_path)
                                    # Look for a typical output structure
                                    if 'OCR' not in base_output_dir:
                                        # Try to find or create the OCR directory
                                        parent_dir = base_output_dir
                                        while parent_dir and not os.path.exists(os.path.join(parent_dir, 'OCR')):
                                            new_parent = os.path.dirname(parent_dir)
                                            if new_parent == parent_dir:  # Reached root
                                                break
                                            parent_dir = new_parent
                                        
                                        if parent_dir and os.path.exists(os.path.join(parent_dir, 'OCR')):
                                            base_output_dir = parent_dir
                                        else:
                                            # Create OCR in the same directory as the source
                                            base_output_dir = os.path.dirname(image_path)
                                
                                compressed_dir = os.path.join(base_output_dir, "OCR", "compressed")
                                
                                # Ensure the directory exists with proper error handling
                                try:
                                    os.makedirs(compressed_dir, exist_ok=True)
                                except OSError as e:
                                    print(f"   ⚠️ Failed to create compressed directory: {e}")
                                    # Fall back to source image directory
                                    compressed_dir = os.path.join(os.path.dirname(image_path), "compressed")
                                    os.makedirs(compressed_dir, exist_ok=True)
                                
                                base_name = os.path.basename(image_path)
                                name, original_ext = os.path.splitext(base_name)
                                
                                # Add source format info to filename if converting from GIF
                                if is_gif and target_format != 'gif':
                                    name = f"{name}_from_gif"
                                
                                ext = '.webp' if target_format == 'webp' else f'.{target_format}'
                                output_path = os.path.join(compressed_dir, f"{name}_compressed{ext}")
                                
                                # Write the file with proper error handling
                                try:
                                    with open(output_path, 'wb') as f:
                                        f.write(buffer.getvalue())
                                    print(f"   💾 Saved compressed image: {output_path}")
                                except OSError as e:
                                    print(f"   ❌ Failed to save compressed image: {e}")
                                    # Fall back to temporary file
                                    raise  # This will trigger the temporary file fallback below
                                    
                            except Exception as e:
                                print(f"   ⚠️ Failed to save to permanent location: {e}")
                                # Fall back to temporary file
                                import tempfile
                                ext = '.webp' if target_format == 'webp' else f'.{target_format}'
                                with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                                    tmp.write(buffer.getvalue())
                                    output_path = tmp.name
                                print(f"   📝 Created temp compressed image instead")
                        else:
                            # Save to temporary file
                            import tempfile
                            ext = '.webp' if target_format == 'webp' else f'.{target_format}'
                            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                                tmp.write(buffer.getvalue())
                                output_path = tmp.name
                            
                            print(f"   📝 Created temp compressed image")
                        
                        compression_ratio = (1 - compressed_size_mb / original_size_mb) * 100
                        if compression_ratio > 0:
                            print(f"   ✅ Compressed: {original_size_mb:.2f}MB → {compressed_size_mb:.2f}MB "
                                  f"({compression_ratio:.1f}% reduction, quality: {quality})")
                        else:
                            print(f"   ⚠️ Compression increased size: {original_size_mb:.2f}MB → {compressed_size_mb:.2f}MB "
                                  f"({abs(compression_ratio):.1f}% larger, quality: {quality})")
                        
                        # Special note for GIF conversions
                        if is_gif:
                            print(f"   🎞️ GIF converted to {target_format.upper()} for better compression")
                        
                        return output_path
                    
                    # Reduce quality and try again
                    quality -= 5
                    print(f"   🔄 Size {compressed_size_mb:.2f}MB > target {max_size_mb:.2f}MB, "
                          f"reducing quality to {quality}")
                
                # If we couldn't meet the target, return the best we got
                print(f"   ⚠️ Could not meet size target, using minimum quality")
                return output_path if output_path else image_path
                
        except Exception as e:
            print(f"   ❌ Compression failed: {e}")
            import traceback
            traceback.print_exc()
            return image_path  # Return original on error

    def _process_image_with_compression(self, image_path, context, check_stop_fn):
        """Process image with optional compression before translation"""
        try:
            # Apply compression if enabled
            if os.getenv("ENABLE_IMAGE_COMPRESSION", "0") == "1":
                compressed_path = self.compress_image(image_path)
                if compressed_path != image_path:
                    # Use compressed image for translation
                    result = self._process_single_image_original(compressed_path, context, check_stop_fn)
                    
                    # Clean up temp file if needed
                    if not os.getenv("SAVE_COMPRESSED_IMAGES", "0") == "1":
                        try:
                            os.unlink(compressed_path)
                        except:
                            pass
                    
                    return result
            
            # No compression, use original method
            return self._process_single_image_original(image_path, context, check_stop_fn)
            
        except Exception as e:
            print(f"   ❌ Error in image processing: {e}")
            return None

    def _process_image_chunks_single_api(self, img, width, height, context, check_stop_fn):
            """Process all image chunks in a single API call with compression support"""
            
            num_chunks = (height + self.chunk_height - 1) // self.chunk_height
            overlap_percentage = float(os.getenv('IMAGE_CHUNK_OVERLAP_PERCENT', '1'))
            overlap = int(self.chunk_height * (overlap_percentage / 100))
            
            print("   🚀 Using SINGLE API CALL mode for " + str(num_chunks) + " chunks")
            print(f"   📐 Chunk overlap: {overlap_percentage}% ({overlap} pixels)")
            #print("   📊 This is more efficient and produces better translations")
            #print("   ⏳ Estimated time: 30-90 seconds total")
            
            # Check for stop at the very beginning
            if check_stop_fn and check_stop_fn():
                print("   ❌ Image translation stopped by user")
                return None
            
            # Load progress for resumability
            prog = self.load_progress()
            image_basename = os.path.basename(self.current_image_path) if hasattr(self, 'current_image_path') else str(hash(str(img)))
            
            # Detect original image format from filename or image
            original_format = 'png'  # default
            if hasattr(self, 'current_image_path'):
                ext = os.path.splitext(self.current_image_path)[1].lower()
                if ext in ['.gif', '.jpg', '.jpeg', '.png', '.webp']:
                    original_format = ext[1:]  # Remove the dot
                    if original_format == 'jpg':
                        original_format = 'jpeg'
            
            # Check if we should preserve original format
            preserve_original_format = os.getenv("PRESERVE_ORIGINAL_FORMAT", "0") == "1"
            
            # Try to extract chapter number
            chapter_num = None
            if hasattr(self, 'current_chapter_num'):
                chapter_num = self.current_chapter_num
            else:
                import re
                match = re.search(r'ch(?:apter)?[\s_-]*(\d+)', image_basename, re.IGNORECASE)
                if match:
                    chapter_num = match.group(1)
            
            # Create unique key
            if chapter_num:
                image_key = "ch" + str(chapter_num) + "_" + image_basename
            else:
                image_key = image_basename
            
            # Check if already processed
            if "single_api_chunks" not in prog:
                prog["single_api_chunks"] = {}
            
            if image_key in prog["single_api_chunks"] and prog["single_api_chunks"][image_key].get("completed"):
                print("   ⏭️ Image already translated, using cached result")
                return prog["single_api_chunks"][image_key]["translation"]
            
            # Prepare chunks
            try:
                content_parts = []
                
                print("   📦 Preparing " + str(num_chunks) + " image chunks...")
                
                # Check if we should save debug images
                save_cleaned = os.getenv('SAVE_CLEANED_IMAGES', '0') == '1'
                if save_cleaned:
                    debug_dir = os.path.join(self.ocr_dir, "debug_chunks")
                    os.makedirs(debug_dir, exist_ok=True)
                    print("   🔍 Debug mode: Saving chunks to " + debug_dir)
                    
                    # Create subdirectory for compressed chunks
                    compressed_debug_dir = os.path.join(debug_dir, "compressed")
                    os.makedirs(compressed_debug_dir, exist_ok=True)
                
                # Check if compression is enabled
                compression_enabled = os.getenv("ENABLE_IMAGE_COMPRESSION", "0") == "1"
                total_uncompressed_size = 0
                total_compressed_size = 0
                
                # Temporarily set the original format in environment for _image_to_bytes_with_compression
                old_env_format = os.environ.get("ORIGINAL_IMAGE_FORMAT", "")
                if preserve_original_format and original_format:
                    os.environ["ORIGINAL_IMAGE_FORMAT"] = original_format
                
                for i in range(num_chunks):
                    # Check for stop during preparation
                    if check_stop_fn and check_stop_fn():
                        print("   ❌ Stopped while preparing chunk " + str(i+1) + "/" + str(num_chunks))
                        # Restore environment
                        if old_env_format:
                            os.environ["ORIGINAL_IMAGE_FORMAT"] = old_env_format
                        elif "ORIGINAL_IMAGE_FORMAT" in os.environ:
                            del os.environ["ORIGINAL_IMAGE_FORMAT"]
                        return None
                        
                    # Calculate chunk boundaries with overlap
                    start_y = max(0, i * self.chunk_height - (overlap if i > 0 else 0))
                    end_y = min(height, (i + 1) * self.chunk_height)
                    
                    # Crop the chunk
                    chunk = img.crop((0, start_y, width, end_y))
                    
                    # Save uncompressed debug chunk if enabled
                    if save_cleaned:
                        # Use original format for debug chunks if preserving format
                        if preserve_original_format and original_format == 'gif':
                            chunk_ext = 'gif'
                            # Need to convert to palette mode for GIF
                            if chunk.mode not in ('P', 'L'):
                                chunk_to_save = chunk.quantize(colors=256, method=2)  # MEDIANCUT
                            else:
                                chunk_to_save = chunk
                        else:
                            chunk_ext = 'png'
                            chunk_to_save = chunk
                        
                        chunk_filename = image_key + "_chunk_" + str(i+1) + "_of_" + str(num_chunks) + "_y" + str(start_y) + "-" + str(end_y) + "." + chunk_ext
                        chunk_path = os.path.join(debug_dir, chunk_filename)
                        
                        if chunk_ext == 'gif':
                            chunk_to_save.save(chunk_path, "GIF", optimize=True)
                        else:
                            chunk_to_save.save(chunk_path, "PNG")
                        
                        print("   💾 Saved debug chunk: " + chunk_filename)
                        
                        # Get uncompressed size
                        uncompressed_size = os.path.getsize(chunk_path)
                        total_uncompressed_size += uncompressed_size
                    
                    # Convert chunk to bytes with compression if enabled
                    if compression_enabled:
                        print(f"   🗜️ Compressing chunk {i+1}/{num_chunks}...")
                        
                        # Use the compression method
                        chunk_bytes = self._image_to_bytes_with_compression(chunk)
                        
                        # Determine format based on compression settings
                        format_setting = os.getenv("IMAGE_COMPRESSION_FORMAT", "auto")
                        if format_setting == "auto":
                            if preserve_original_format and original_format == 'gif':
                                # If original was GIF and we're preserving format, use GIF
                                format_used = 'gif'
                            else:
                                # Check if chunk has transparency
                                has_transparency = chunk.mode in ('RGBA', 'LA') or (chunk.mode == 'P' and 'transparency' in chunk.info)
                                preserve_transparency = os.getenv("PRESERVE_TRANSPARENCY", "0") == "1"
                                if has_transparency and preserve_transparency:
                                    format_used = 'png'
                                else:
                                    format_used = 'webp'  # Default to WebP for best compression
                        else:
                            format_used = format_setting
                        
                        # Calculate compression stats
                        compressed_size = len(chunk_bytes)
                        if save_cleaned:
                            # Get the actual original size of the chunk before compression
                            original_chunk_buffer = io.BytesIO()
                            chunk.save(original_chunk_buffer, format='PNG')
                            actual_original_size = len(original_chunk_buffer.getvalue())
                            compression_ratio = (1 - compressed_size / actual_original_size) * 100
                            print(f"   📊 Chunk {i+1}: {uncompressed_size:,} → {compressed_size:,} bytes ({compression_ratio:.1f}% reduction, format: {format_used.upper()})")
                            total_compressed_size += compressed_size
                            
                            # Save compressed chunk for debugging
                            compressed_chunk_filename = image_key + "_chunk_" + str(i+1) + "_compressed." + format_used.lower()
                            compressed_chunk_path = os.path.join(compressed_debug_dir, compressed_chunk_filename)
                            with open(compressed_chunk_path, 'wb') as f:
                                f.write(chunk_bytes)
                            print(f"   💾 Saved compressed chunk: {compressed_chunk_filename}")
                    else:
                        # No compression - use original format if preserving, otherwise PNG
                        if preserve_original_format and original_format == 'gif':
                            chunk_bytes = self._image_to_bytes(chunk, format='GIF')
                            format_used = 'gif'
                        else:
                            chunk_bytes = self._image_to_bytes(chunk, format='PNG')
                            format_used = 'png'
                        
                        if save_cleaned:
                            total_compressed_size += len(chunk_bytes)
                    
                    # Convert to base64
                    chunk_base64 = base64.b64encode(chunk_bytes).decode('utf-8')
                    
                    # Add image to content with appropriate format
                    content_parts.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/{format_used.lower()};base64," + chunk_base64
                        }
                    })
                
                # Restore original environment variable
                if old_env_format:
                    os.environ["ORIGINAL_IMAGE_FORMAT"] = old_env_format
                elif "ORIGINAL_IMAGE_FORMAT" in os.environ:
                    del os.environ["ORIGINAL_IMAGE_FORMAT"]
                
                # Count the number of images in content_parts
                num_images = sum(1 for part in content_parts if part.get("type") == "image_url")
                
                # Show overall compression stats if enabled
                if compression_enabled and save_cleaned and total_uncompressed_size > 0:
                    overall_compression = (1 - total_compressed_size / total_uncompressed_size) * 100
                    print(f"\n   📊 Overall compression stats:")
                    print(f"      Total uncompressed: {total_uncompressed_size:,} bytes ({total_uncompressed_size / 1024 / 1024:.2f} MB)")
                    print(f"      Total compressed: {total_compressed_size:,} bytes ({total_compressed_size / 1024 / 1024:.2f} MB)")
                    print(f"      Reduction: {overall_compression:.1f}%")
                    print(f"      Savings: {(total_uncompressed_size - total_compressed_size):,} bytes\n")
                
            except Exception as e:
                # Make sure to restore environment
                if 'old_env_format' in locals():
                    if old_env_format:
                        os.environ["ORIGINAL_IMAGE_FORMAT"] = old_env_format
                    elif "ORIGINAL_IMAGE_FORMAT" in os.environ:
                        del os.environ["ORIGINAL_IMAGE_FORMAT"]
                        
                print("   ❌ Error preparing chunks: " + str(e))
                import traceback
                traceback.print_exc()
                print("   🔄 Falling back to sequential chunk processing...")
                return self._process_image_chunks(img, width, height, context, check_stop_fn)
            
            # Calculate token estimate based on provider
            if 'gemini' in self.client.model.lower():
                # Gemini charges flat 258 tokens per image
                estimated_image_tokens = num_images * 258
            elif 'gpt-4' in self.client.model.lower() or 'gpt-4o' in self.client.model.lower():
                # GPT-4V uses ~85 tokens per 512x512 tile
                # Adjust estimate based on compression
                if compression_enabled:
                    # Compressed images use fewer tokens
                    tiles_per_chunk = max(1, (self.chunk_height * width * 0.7) // (512 * 512))
                else:
                    tiles_per_chunk = max(1, (self.chunk_height * width) // (512 * 512))
                estimated_image_tokens = num_images * tiles_per_chunk * 85
            elif 'claude' in self.client.model.lower():
                # Claude varies by resolution, estimate based on compression
                if compression_enabled:
                    estimated_image_tokens = num_images * 1500  # Compressed images
                else:
                    estimated_image_tokens = num_images * 2000  # Uncompressed
            else:
                # Default conservative estimate
                estimated_image_tokens = num_images * 1000
            
            # Calculate text tokens
            text_tokens = sum(len(part.get("text", "")) for part in content_parts if part.get("type") == "text") // 4
            estimated_text_tokens = len(self.system_prompt) // 4 + text_tokens + 200
            total_estimated_tokens = estimated_image_tokens + estimated_text_tokens
            
            print("   📊 Token estimate:")
            print("      Number of images: " + str(num_images))
            print("      Image tokens: ~" + "{:,}".format(estimated_image_tokens) + " (model: " + self.client.model + ")")
            if compression_enabled:
                print("      Compression: ENABLED ✅")
            print("      Text tokens: ~" + "{:,}".format(estimated_text_tokens))
            print("      Total: ~" + "{:,}".format(total_estimated_tokens) + " tokens")
            
            # Make the API call
            try:
                # Build messages
                messages = [{"role": "system", "content": self.system_prompt}]
                messages.append({
                    "role": "user",
                    "content": content_parts
                })
                
                print("\n   🔄 Sending " + str(num_chunks) + " chunks to API in single call...")
                if compression_enabled:
                    print("   🗜️ Using compressed chunks for efficient API usage")
                
                # Final stop check before API call
                if check_stop_fn and check_stop_fn():
                    print("   ❌ Stopped before API call")
                    return None
                
                # Use send_image_with_interrupt for interruptible API call
                start_time = time.time()
                
                # Get timeout settings
                chunk_timeout = int(os.getenv('CHUNK_TIMEOUT', '0'))
                retry_timeout = os.getenv('RETRY_TIMEOUT', '0') == '1'
                
                # Make interruptible API call
                # Since we already have images in content_parts, we need to use regular send, not send_image
                try:
                    # Create a wrapper to make regular send interruptible
                    result_queue = queue.Queue()
                    
                    def api_call():
                        try:
                            start = time.time()
                            result = self.client.send(
                                messages=messages,
                                temperature=self.temperature,
                                max_tokens=self.image_max_tokens
                            )
                            elapsed_time = time.time() - start
                            result_queue.put((result, elapsed_time))
                        except Exception as e:
                            result_queue.put(e)
                    
                    api_thread = threading.Thread(target=api_call)
                    api_thread.daemon = True
                    api_thread.start()
                    
                    # Check for completion or stop
                    timeout = chunk_timeout
                    check_interval = 0.5
                    elapsed_check = 0
                    
                    while True:
                        try:
                            result = result_queue.get(timeout=check_interval)
                            if isinstance(result, Exception):
                                raise result
                            if isinstance(result, tuple):
                                response, elapsed_time = result
                                elapsed = elapsed_time
                                break
                        except queue.Empty:
                            if check_stop_fn and check_stop_fn():
                                raise UnifiedClientError("Translation stopped by user")
                            elapsed_check += check_interval
                            if timeout is not None and elapsed_check >= timeout:
                                raise UnifiedClientError("API call timed out after " + str(timeout) + " seconds")
                        
                except UnifiedClientError as e:
                    if "stopped by user" in str(e).lower():
                        print("   ❌ Translation stopped by user during API call")
                        return None
                    elif "timed out" in str(e).lower():
                        print("   ⏱️ API call timed out: " + str(e))
                        print("   🔄 Falling back to sequential chunk processing...")
                        return self._process_image_chunks(img, width, height, context, check_stop_fn)
                    else:
                        raise
                
                # Handle the result based on what's returned
                if isinstance(response, tuple):
                    response, elapsed_time = response
                    # Handle case where elapsed_time might be 'stop' or other non-numeric
                    try:
                        elapsed = float(elapsed_time)
                    except (ValueError, TypeError):
                        elapsed = time.time() - start_time
                
                # Success!
                print("   📡 API response received in " + "{:.1f}".format(elapsed) + "s")
                
                # Check if response is valid
                if not response:
                    print("   ❌ No response from API")
                    print("   🔄 Falling back to sequential chunk processing...")
                    return self._process_image_chunks(img, width, height, context, check_stop_fn)

                # Extract content from UnifiedResponse
                if hasattr(response, 'content'):
                    translation_response = response.content
                elif hasattr(response, 'text'):
                    translation_response = response.text
                else:
                    translation_response = str(response)

                # Unescape the response text if it has escaped characters
                if '\\n' in translation_response or translation_response.startswith('('):
                    print("   🔧 Detected escaped text, unescaping...")
                    translation_response = self._unescape_response_text(translation_response)

                # Check if we got actual content
                if not translation_response or not translation_response.strip():
                    print("   ❌ Empty response content from API")
                    print("   🔄 Falling back to sequential chunk processing...")
                    return self._process_image_chunks(img, width, height, context, check_stop_fn)

                # Process response
                trans_finish = getattr(response, 'finish_reason', 'unknown')

                print("   📡 Finish reason: " + trans_finish)
                print("   📄 Response length: " + str(len(translation_response)) + " characters")

                if trans_finish in ["length", "max_tokens"]:
                    print("   ⚠️ Translation was TRUNCATED! Consider increasing Max tokens.")
                    translation_response += "\n\n[TRANSLATION TRUNCATED DUE TO TOKEN LIMIT]"

                # Clean translation based on REMOVE_AI_ARTIFACTS setting
                if self.remove_ai_artifacts != "off":
                    cleaned_translation = self._clean_translation_response(translation_response)
                    print(f"   🧹 Cleaned translation (artifact removal: {self.remove_ai_artifacts})")
                else:
                    cleaned_translation = translation_response
                    print("   📝 Using raw translation (artifact removal off)")

                # Sanitize invalid glyphs without changing valid punctuation width/forms.
                cleaned_translation = self._sanitize_unicode_characters(cleaned_translation)

                if not cleaned_translation:
                    print("   ❌ No text extracted from response after cleaning")
                    print("   🔄 Falling back to sequential chunk processing...")
                    return self._process_image_chunks(img, width, height, context, check_stop_fn)
                
                # Save to progress
                if "single_api_chunks" not in prog:
                    prog["single_api_chunks"] = {}
                    
                prog["single_api_chunks"][image_key] = {
                    "completed": True,
                    "translation": cleaned_translation,
                    "chunks": num_chunks,
                    "overlap": overlap,
                    "compression_enabled": compression_enabled,
                    "original_format": original_format,
                    "timestamp": time.time()
                }
                self.save_progress(prog)
                
                print("   ✅ Translation complete (" + str(len(cleaned_translation)) + " chars)")
                return cleaned_translation
                
            except Exception as e:
                error_str = str(e)
                error_msg = error_str.lower()
                
                # Log the full error
                print("   ❌ API Error: " + error_str)
                import traceback
                traceback.print_exc()
                
                # Check for stop
                if "stopped by user" in error_msg or (check_stop_fn and check_stop_fn()):
                    print("   ❌ Translation stopped by user")
                    return None
                
                # For any API error at this point, fall back to sequential
                print("   🔄 Single API call failed, falling back to sequential chunk processing...")
                return self._process_image_chunks(img, width, height, context, check_stop_fn)
        
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
    
    def load_progress(self):
        """Load progress tracking for image chunks"""
        if self.progress_manager:
            # Use the shared progress manager's data
            prog = self.progress_manager.prog.copy()
            # Ensure image_chunks key exists
            if "image_chunks" not in prog:
                prog["image_chunks"] = {}
            prog.pop("image_ocr_chunks", None)
            return prog
        else:
            # Fallback to original behavior if no progress manager provided
            progress_file = os.path.join(self.output_dir, "translation_progress.json")
            if os.path.exists(progress_file):
                try:
                    with open(progress_file, 'r', encoding='utf-8') as f:
                        prog = json.load(f)
                    # Ensure image_chunks key exists
                    if "image_chunks" not in prog:
                        prog["image_chunks"] = {}
                    prog.pop("image_ocr_chunks", None)
                    return prog
                except Exception as e:
                    print(f"⚠️ Warning: Could not load progress file: {e}")
                    # Return minimal structure to avoid breaking
                    return {
                        "chapters": {},
                        "content_hashes": {},
                        "chapter_chunks": {},
                        "image_chunks": {},
                        "version": "2.1"
                    }
            # Return the same structure as TranslateKRtoEN expects
            return {
                "chapters": {},
                "content_hashes": {},
                "chapter_chunks": {},
                "image_chunks": {},
                "version": "2.1"
            }

    def save_progress(self, prog):
        """Save progress tracking - with safe writing"""
        prog.pop("image_ocr_chunks", None)
        if self.progress_manager:
            # Update the shared progress manager's data
            self.progress_manager.prog["image_chunks"] = prog.get("image_chunks", {})
            self.progress_manager.prog.pop("image_ocr_chunks", None)
            # Save through the progress manager
            self.progress_manager.save()
        else:
            # Fallback to original behavior if no progress manager provided
            progress_file = os.path.join(self.output_dir, "translation_progress.json")
            try:
                # Write to a temporary file first, with retry for file locks
                temp_file = progress_file + '.tmp'
                max_retries = 5
                for attempt in range(max_retries):
                    try:
                        with open(temp_file, 'w', encoding='utf-8') as f:
                            json.dump(prog, f, ensure_ascii=False, indent=2)
                        break
                    except PermissionError:
                        if attempt < max_retries - 1:
                            time.sleep(0.1 * (2 ** attempt))
                        else:
                            raise
                
                # Atomically replace the original file (with retry)
                for attempt in range(max_retries):
                    try:
                        os.replace(temp_file, progress_file)
                        break
                    except PermissionError:
                        if attempt < max_retries - 1:
                            time.sleep(0.1 * (2 ** attempt))
                        else:
                            raise
            except Exception as e:
                print(f"⚠️ Warning: Failed to save progress: {e}")
                # Clean up temp file if it exists
                if os.path.exists(temp_file):
                    try:
                        os.remove(temp_file)
                    except:
                        pass
    
    def preprocess_image_for_watermarks(self, image_path: str) -> str:
        """
        Enhanced preprocessing for watermark removal and text clarity
        Now returns path to processed image instead of bytes
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Path to processed image (either cleaned permanent file or original)
        """
        try:
            # Check if watermark removal is enabled
            if not os.getenv("ENABLE_WATERMARK_REMOVAL", "1") == "1":
                return image_path  # Return original path
            
            if not getattr(self, "_suppress_image_detail_logs", False):
                print(f"   🧹 Preprocessing image for watermark removal...")
            
            # Open image
            img = Image.open(image_path)
            
            # Convert to RGB if necessary
            if img.mode not in ('RGB', 'RGBA'):
                img = img.convert('RGB')
            
            # Check if advanced watermark removal is enabled AND cv2 is available
            if os.getenv("ADVANCED_WATERMARK_REMOVAL", "0") == "1":
                if CV2_AVAILABLE:
                    print(f"   🔬 Using advanced watermark removal...")
                    
                    # Convert to numpy array for advanced processing
                    img_array = np.array(img)
                    
                    # These will safely return defaults if cv2 is not available
                    has_pattern, pattern_mask = self._detect_watermark_pattern(img_array)
                    if has_pattern:
                        print(f"   🔍 Detected watermark pattern in image")
                        img_array = self._remove_periodic_watermark(img_array, pattern_mask)
                    
                    img_array = self._adaptive_histogram_equalization(img_array)
                    img_array = self._bilateral_filter(img_array)
                    img_array = self._enhance_text_regions(img_array)
                    
                    # Convert back to PIL Image
                    img = Image.fromarray(img_array)
                else:
                    print(f"   ⚠️ Advanced watermark removal requested but OpenCV not available")
            
            # Apply basic PIL enhancements (always works)
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.5)
            
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(1.1)
            
            img = img.filter(ImageFilter.SHARPEN)
            
            # Check if we should save cleaned images
            save_cleaned = os.getenv("SAVE_CLEANED_IMAGES", "1") == "1"
            
            if save_cleaned:
                # Save to permanent location
                cleaned_dir = os.path.join(self.translated_images_dir, "cleaned")
                os.makedirs(cleaned_dir, exist_ok=True)
                
                base_name = os.path.basename(image_path)
                name, ext = os.path.splitext(base_name)
                cleaned_path = os.path.join(cleaned_dir, f"{name}_cleaned{ext}")
                
                img.save(cleaned_path, optimize=True)
                print(f"   💾 Saved cleaned image: {cleaned_path}")
                
                return cleaned_path  # Return path to cleaned image
            else:
                # Save to temporary file
                import tempfile
                _, ext = os.path.splitext(image_path)
                with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                    img.save(tmp.name, optimize=False)
                    if not getattr(self, "_suppress_image_detail_logs", False):
                        print(f"   📝 Created temp cleaned image")
                    return tmp.name  # Return temp path
            
        except Exception as e:
            logger.warning(f"Could not preprocess image: {e}")
            return image_path  # Return original on error
            
    @requires_cv2
    def _detect_watermark_pattern(self, img_array: np.ndarray) -> Tuple[bool, Optional[np.ndarray]]:
        """Detect repeating watermark patterns using FFT"""
        try:
            # Convert to grayscale for pattern detection
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
            
            # Apply FFT to detect periodicity
            f_transform = np.fft.fft2(gray)
            f_shift = np.fft.fftshift(f_transform)
            magnitude = np.log(np.abs(f_shift) + 1)  # Log scale for better visualization
            
            # Look for peaks that indicate repeating patterns
            mean_mag = np.mean(magnitude)
            std_mag = np.std(magnitude)
            threshold = mean_mag + 2 * std_mag
            
            # Create binary mask of high-frequency components
            pattern_mask = magnitude > threshold
            
            # Exclude center (DC component) - represents average brightness
            center_y, center_x = pattern_mask.shape[0] // 2, pattern_mask.shape[1] // 2
            pattern_mask[center_y-10:center_y+10, center_x-10:center_x+10] = False
            
            # Count significant peaks
            pattern_threshold = int(os.getenv("WATERMARK_PATTERN_THRESHOLD", "10"))
            peak_count = np.sum(pattern_mask)
            
            # If we have significant peaks, there's likely a repeating pattern
            has_pattern = peak_count > pattern_threshold
            
            return has_pattern, pattern_mask if has_pattern else None
            
        except Exception as e:
            logger.warning(f"Pattern detection failed: {e}")
            return False, None
            
    @requires_cv2
    def _remove_periodic_watermark(self, img_array: np.ndarray, pattern_mask: np.ndarray) -> np.ndarray:
        """Remove periodic watermark using frequency domain filtering"""
        try:
            result = img_array.copy()
            
            # Process each color channel
            for channel in range(img_array.shape[2] if len(img_array.shape) == 3 else 1):
                if len(img_array.shape) == 3:
                    gray = img_array[:, :, channel]
                else:
                    gray = img_array
                
                # Apply FFT
                f_transform = np.fft.fft2(gray)
                f_shift = np.fft.fftshift(f_transform)
                
                # Apply notch filter to remove periodic components
                f_shift[pattern_mask] = 0
                
                # Inverse FFT
                f_ishift = np.fft.ifftshift(f_shift)
                img_filtered = np.fft.ifft2(f_ishift)
                img_filtered = np.real(img_filtered)
                
                # Ensure values are in valid range
                img_filtered = np.clip(img_filtered, 0, 255)
                
                if len(img_array.shape) == 3:
                    result[:, :, channel] = img_filtered
                else:
                    result = img_filtered
            
            return result.astype(np.uint8)
            
        except Exception as e:
            logger.warning(f"Watermark removal failed: {e}")
            return img_array
            
    @requires_cv2
    def _adaptive_histogram_equalization(self, img_array: np.ndarray) -> np.ndarray:
        """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)"""
        try:
            # Convert to LAB color space for better results
            lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
            
            # Split channels
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel only
            clahe_limit = float(os.getenv("WATERMARK_CLAHE_LIMIT", "3.0"))
            clahe = cv2.createCLAHE(clipLimit=clahe_limit, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            # Merge channels back
            lab = cv2.merge([l, a, b])
            
            # Convert back to RGB
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            
            return enhanced
            
        except Exception as e:
            logger.warning(f"Adaptive histogram equalization failed: {e}")
            return img_array
    
    @requires_cv2
    def _bilateral_filter(self, img_array: np.ndarray) -> np.ndarray:
        """Apply bilateral filter for edge-preserving denoising"""
        try:
            # Bilateral filter removes noise while keeping edges sharp
            filtered = cv2.bilateralFilter(
                img_array, 
                d=9,
                sigmaColor=75,
                sigmaSpace=75
            )
            return filtered
            
        except Exception as e:
            logger.warning(f"Bilateral filtering failed: {e}")
            return img_array
    
    @requires_cv2
    def _enhance_text_regions(self, img_array: np.ndarray) -> np.ndarray:
        """Specifically enhance regions likely to contain text"""
        try:
            # Convert to grayscale for text detection
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            # Step 1: Detect text regions using gradient analysis
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            # Normalize gradient
            gradient_magnitude = (gradient_magnitude / gradient_magnitude.max() * 255).astype(np.uint8)
            
            # Step 2: Create text probability mask
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            gradient_density = cv2.morphologyEx(gradient_magnitude, cv2.MORPH_CLOSE, kernel)
            
            # Threshold to get text regions
            _, text_mask = cv2.threshold(gradient_density, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Dilate to connect text regions
            text_mask = cv2.dilate(text_mask, kernel, iterations=2)
            
            # Step 3: Enhance contrast in text regions
            enhanced = img_array.copy()
            
            # Create 3-channel mask
            text_mask_3ch = cv2.cvtColor(text_mask, cv2.COLOR_GRAY2RGB) / 255.0
            
            # Apply enhancement only to text regions
            enhanced = enhanced.astype(np.float32)
            enhanced = enhanced * (1 + (0.2 * text_mask_3ch))  # 20% enhancement in text regions
            enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
            
            return enhanced
            
        except Exception as e:
            logger.warning(f"Text region enhancement failed: {e}")
            return img_array
    
    def translate_image(self, image_path: str, context: str = "", check_stop_fn=None) -> Optional[str]:
        """
        Translate text in an image using vision API - with chunking for tall images and stop support
        """
        processed_path = None
        compressed_path = None
        
        try:
            self.current_image_path = image_path
            print(f"   🔍 translate_image called for: {image_path}")
            
            self._preserve_current_image = False

            # Check for stop at the beginning
            if check_stop_fn and check_stop_fn():
                print("   ❌ Image translation stopped by user")
                return None
            
            if not os.path.exists(image_path):
                logger.warning(f"Image not found: {image_path}")
                print(f"   ❌ Image file does not exist!")
                return None
            
            # Get configuration
            hide_label = os.getenv("HIDE_IMAGE_TRANSLATION_LABEL", "0") == "1"
            
            # Apply compression FIRST if enabled
            compressed_path = image_path
            if os.getenv("ENABLE_IMAGE_COMPRESSION", "0") == "1":
                compressed_path = self.compress_image(image_path)
                # If compression produced a different file, use it
                if compressed_path != image_path:
                    print(f"   🗜️ Using compressed image for translation")
            
            # Apply watermark preprocessing (on compressed image if applicable)
            processed_path = self.preprocess_image_for_watermarks(compressed_path)
            
            # Open and process the image (now using processed_path)
            with Image.open(processed_path) as img:
                width, height = img.size
                aspect_ratio = width / height if height > 0 else 1
                print(f"   📐 Image dimensions: {width}x{height}, aspect ratio: {aspect_ratio:.2f}")
                
                # Convert to RGB if necessary
                if img.mode not in ('RGB', 'RGBA'):
                    img = img.convert('RGB')
                
                # Determine if it's a long text image
                is_long_text = height > self.webnovel_min_height and aspect_ratio < 0.5
                
                # Process chunks or single image
                if height > self.chunk_height:
                    # Check if single API mode is enabled
                    if self._use_ocr_first_pipeline():
                        print("   🔎 Vision OCR-first mode enabled; OCRing tall-image chunks, then translating combined OCR once")
                        translated_text = self._process_image_chunks(img, width, height, context, check_stop_fn)
                    elif os.getenv("SINGLE_API_IMAGE_CHUNKS", "1") == "1":
                        translated_text = self._process_image_chunks_single_api(img, width, height, context, check_stop_fn)
                    else:
                        translated_text = self._process_image_chunks(img, width, height, context, check_stop_fn)
                else:
                    translated_text = self._process_single_image(img, context, check_stop_fn)
                
                if not translated_text:
                    return None
            
            # Store the result for caching (use original path as key)
            self.processed_images[image_path] = translated_text
            
            # Save translation for debugging
            self._save_translation_debug(image_path, translated_text)
            
            # Create HTML output - use processed_path for the image reference
            # Handle cross-drive paths on Windows
            try:
                img_rel_path = os.path.relpath(processed_path, self.output_dir)
            except ValueError as e:
                # This happens when paths are on different drives in Windows
                print(f"   ⚠️ Cross-drive path detected, copying image to output directory")
                
                # Copy the processed image to the output directory's images folder
                import shutil
                images_output_dir = os.path.join(self.output_dir, "images")
                os.makedirs(images_output_dir, exist_ok=True)
                
                # Generate a unique filename to avoid conflicts
                base_name = os.path.basename(processed_path)
                dest_path = os.path.join(images_output_dir, base_name)
                
                # Handle potential naming conflicts
                if os.path.exists(dest_path):
                    name, ext = os.path.splitext(base_name)
                    counter = 1
                    while os.path.exists(dest_path):
                        dest_path = os.path.join(images_output_dir, f"{name}_{counter}{ext}")
                        counter += 1
                
                # Copy the file
                shutil.copy2(processed_path, dest_path)
                print(f"   📋 Copied image to: {dest_path}")
                
                # Calculate relative path from the copied location
                img_rel_path = os.path.relpath(dest_path, self.output_dir)
                
                # Update processed_path for cleanup logic
                processed_path = dest_path
            
            html_output = self._create_html_output(img_rel_path, translated_text, is_long_text, 
                                                 hide_label, check_stop_fn and check_stop_fn())
            
            return html_output
            
        except Exception as e:
            is_cancelled = False
            try:
                is_cancelled = isinstance(e, UnifiedClientError) and getattr(e, "error_type", None) == "cancelled"
            except Exception:
                is_cancelled = False
            if is_cancelled or "stopped by user" in str(e).lower() or "cancelled" in str(e).lower():
                print(f"   ⏹️ Image translation cancelled: {e}")
                return None

            logger.error(f"Error translating image {image_path}: {e}")
            print(f"   ❌ Exception in translate_image: {e}")
            import traceback
            traceback.print_exc()
            return None
            
        finally:
            # Clean up temp files if they were created
            # Clean up compressed file if it's temporary
            if compressed_path and compressed_path != image_path:
                if not os.getenv("SAVE_COMPRESSED_IMAGES", "0") == "1":
                    try:
                        if os.path.exists(compressed_path):
                            os.unlink(compressed_path)
                            print(f"   🧹 Cleaned up temp compressed file")
                    except Exception as e:
                        logger.warning(f"Could not delete temp compressed file: {e}")
            
            # Clean up processed file if it's temporary
            if processed_path and processed_path != image_path and processed_path != compressed_path:
                if not os.getenv("SAVE_CLEANED_IMAGES", "0") == "1":
                    try:
                        if os.path.exists(processed_path):
                            os.unlink(processed_path)
                            print(f"   🧹 Cleaned up temp processed file")
                    except Exception as e:
                        logger.warning(f"Could not delete temp processed file: {e}")

    def ocr_image(self, image_path: str, context: str = "", check_stop_fn=None) -> Optional[str]:
        """OCR an image and save/reuse OCR text without running the translation phase."""
        processed_path = None
        compressed_path = None
        try:
            self.current_image_path = image_path
            self._preserve_current_image = False
            print(f"   🔎 ocr_image called for: {image_path}")
            if check_stop_fn and check_stop_fn():
                print("   ❌ Image OCR stopped by user")
                return None
            if not os.path.exists(image_path):
                print(f"   ❌ Image file does not exist: {image_path}")
                return None

            disk_ocr = self._load_saved_ocr_text(kind="single", image_basename=os.path.basename(image_path))
            if disk_ocr:
                if self._is_ocr_no_response(disk_ocr):
                    self._preserve_current_image = True
                    print("   Skipping OCR image; cached OCR/single says cover/illustration")
                    return None
                print(f"   Skipping OCR image; found existing OCR/single file ({len(disk_ocr.strip())} chars)")
                return disk_ocr.strip()

            compressed_path = image_path
            if os.getenv("ENABLE_IMAGE_COMPRESSION", "0") == "1":
                compressed_path = self.compress_image(image_path)

            processed_path = self.preprocess_image_for_watermarks(compressed_path)
            with Image.open(processed_path) as img:
                width, height = img.size
                if not getattr(self, "_suppress_image_detail_logs", False):
                    print(f"   📐 OCR image dimensions: {width}x{height}")
                if img.mode not in ('RGB', 'RGBA'):
                    img = img.convert('RGB')
                if height > self.chunk_height:
                    return self._process_image_chunks_ocr_first_combined(
                        img, width, height, context, check_stop_fn, translate_after=False
                    )
                image_bytes = self._image_to_bytes_with_compression(img)
                ocr_text = self._call_vision_ocr_api(image_bytes, context, check_stop_fn)
                if self._is_ocr_no_response(ocr_text):
                    self._preserve_current_image = True
                    print("   OCR marked image as cover/illustration; skipping OCR text use")
                    return None
                return ocr_text
        except Exception as e:
            is_cancelled = False
            try:
                is_cancelled = isinstance(e, UnifiedClientError) and getattr(e, "error_type", None) == "cancelled"
            except Exception:
                is_cancelled = False
            if is_cancelled or "stopped by user" in str(e).lower() or "cancelled" in str(e).lower():
                print(f"   ⏹️ Image OCR cancelled: {e}")
                return None
            print(f"   ❌ Exception in ocr_image: {e}")
            logger.error(f"Error OCRing image {image_path}: {e}")
            return None
        finally:
            if compressed_path and compressed_path != image_path and os.getenv("SAVE_COMPRESSED_IMAGES", "0") != "1":
                try:
                    if os.path.exists(compressed_path):
                        os.unlink(compressed_path)
                except Exception:
                    pass
            if processed_path and processed_path != image_path and processed_path != compressed_path and os.getenv("SAVE_CLEANED_IMAGES", "0") != "1":
                try:
                    if os.path.exists(processed_path):
                        os.unlink(processed_path)
                except Exception:
                    pass


    def _process_single_image(self, img, context, check_stop_fn):
        """Process a single image that doesn't need chunking"""
        
        # Clear any previous context
        self.image_chunk_context = []
        
        print(f"   👍 Image height OK ({img.height}px), processing as single image...")
        
        # Check for stop before processing
        if check_stop_fn and check_stop_fn():
            print("   ❌ Image translation stopped by user")
            return None
        
        # Convert image to bytes using compression settings
        image_bytes = self._image_to_bytes_with_compression(img)
        
        # Call API
        translation = self._call_vision_api(image_bytes, context, check_stop_fn)
        
        if translation:
            if self.remove_ai_artifacts != "off":
                translation = self._clean_translation_response(translation)
            # Sanitize invalid glyphs without changing valid punctuation width/forms.
            translation = self._sanitize_unicode_characters(translation)
            return translation
        else:
            print(f"   ❌ Translation returned empty result")
            return None

    def _use_ocr_first_pipeline(self) -> bool:
        """Return True when Vision mode should OCR images before text translation."""
        output_mode = os.getenv("OUTPUT_MODE", "").strip().lower()
        force = os.getenv("VISION_OCR_FIRST", "auto").strip().lower()
        if force in ("0", "false", "no", "off"):
            return False
        if force in ("1", "true", "yes", "on"):
            return True
        return output_mode == "vision"

    @staticmethod
    def _is_ocr_no_response(text) -> bool:
        """Return True when the OCR model used the cover/illustration sentinel."""
        if text is None:
            return False
        normalized = str(text).strip()
        normalized = re.sub(r'^[\s"`\'*_~]+|[\s"`\'*_~.。!！]+$', '', normalized).strip()
        return normalized.lower() == "no"

    def _call_vision_ocr_api(
        self,
        image_data,
        assistant_prompt,
        check_stop_fn,
        chunk_idx=None,
        image_basename=None,
        image_idx=None,
        chapter_num=None,
    ):
        """OCR an image/chunk using the dedicated Vision OCR prompt."""
        messages = [{"role": "system", "content": self.vision_ocr_prompt}]
        user_prompt_template = (self.vision_ocr_user_prompt or DEFAULT_VISION_OCR_USER_PROMPT).strip()
        context_text = (assistant_prompt or "").strip()
        if "{context}" in user_prompt_template:
            user_prompt = user_prompt_template.replace("{context}", context_text).strip()
        elif context_text:
            user_prompt = f"{user_prompt_template}\n\nContext:\n{context_text}".strip()
        else:
            user_prompt = user_prompt_template
        messages.append({"role": "user", "content": user_prompt})

        effective_chapter_num = chapter_num if chapter_num is not None else getattr(self, 'current_chapter_num', None)
        effective_image_idx = image_idx if image_idx is not None else getattr(self, 'current_image_index', 0)
        if effective_chapter_num is not None:
            chunk_suffix = f"_chunk_{chunk_idx:03d}" if chunk_idx is not None else ""
            try:
                chapter_part = f"{int(effective_chapter_num):03d}"
            except Exception:
                chapter_part = str(effective_chapter_num)
            self.client.set_output_filename(
                f"ocr_{chapter_part}_Chapter_{effective_chapter_num}_image_{effective_image_idx}{chunk_suffix}.txt"
            )

        retry_timeout_enabled = os.getenv("RETRY_TIMEOUT", "1") == "1"
        chunk_timeout = int(os.getenv("CHUNK_TIMEOUT", "1800")) if retry_timeout_enabled else None

        ocr_response, finish_reason = send_image_with_interrupt(
            self.client,
            messages,
            image_data,
            self.temperature,
            self.image_max_tokens,
            check_stop_fn,
            chunk_timeout,
            'vision_ocr'
        )

        if finish_reason in ["length", "max_tokens"]:
            print("   ⚠️ OCR response was truncated. Consider increasing Max tokens.")

        ocr_text = (ocr_response or "").strip()
        if self.remove_ai_artifacts != "off":
            ocr_text = self._clean_translation_response(ocr_text)
        ocr_text = self._sanitize_unicode_characters(ocr_text)
        self._save_ocr_text(
            ocr_text,
            kind="chunks" if chunk_idx is not None else "single",
            image_basename=image_basename,
            chunk_idx=chunk_idx,
            image_idx=effective_image_idx,
            chapter_num=effective_chapter_num,
        )
        return ocr_text.strip()

    def _translate_ocr_text(self, ocr_text, assistant_prompt, check_stop_fn):
        """Translate OCR text as a normal text request."""
        self.last_vision_translation_finish_reason = None
        self.last_vision_translation_error = None
        if not ocr_text or not ocr_text.strip():
            print("   ℹ️ OCR returned no readable text")
            return None

        if self._is_ocr_no_response(ocr_text):
            print("   OCR marked image as cover/illustration; skipping OCR translation")
            return None

        if check_stop_fn and check_stop_fn():
            print("   ❌ Stopped before OCR text translation")
            return None

        ocr_text = self._inject_context_headers_into_ocr(ocr_text, assistant_prompt)
        system_prompt = self._prepare_vision_ocr_translation_prompt(ocr_text, check_stop_fn)

        user_parts = []
        if assistant_prompt and assistant_prompt.strip():
            user_parts.append(assistant_prompt.strip())
        user_parts.append(
            "Translate the following OCR text according to the system prompt. "
            "Return only the translated text. Preserve the OCR paragraph and line-break structure; "
            "do not collapse separate source lines into one line."
        )
        user_parts.append("<OCR_TEXT>\n" + ocr_text.strip() + "\n</OCR_TEXT>")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "\n\n".join(user_parts)}
        ]

        single_pass_enabled = False
        try:
            from TransateKRtoEN import _build_single_pass_glossary_messages, _single_pass_glossary_mode
            single_pass_enabled = bool(_single_pass_glossary_mode())
            if single_pass_enabled:
                messages = _build_single_pass_glossary_messages(messages, ocr_text)
        except Exception as e:
            print(f"   ⚠️ Vision Single Pass Glossary setup skipped: {e}")

        if hasattr(self, 'current_chapter_num'):
            chapter_num = self.current_chapter_num
            image_idx = getattr(self, 'current_image_index', 0)
            self.client.set_output_filename(f"response_{chapter_num:03d}_Chapter_{chapter_num}_image_{image_idx}.html")

        retry_timeout_enabled = os.getenv("RETRY_TIMEOUT", "1") == "1"
        chunk_timeout = int(os.getenv("CHUNK_TIMEOUT", "1800")) if retry_timeout_enabled else None

        try:
            response = send_text_with_interrupt(
                self.client,
                messages,
                self.temperature,
                self.image_max_tokens,
                stop_check_fn=check_stop_fn,
                chunk_timeout=chunk_timeout,
                context='translation',
            )
        except UnifiedClientError as e:
            err_text = str(e).lower()
            self.last_vision_translation_error = str(e)
            if getattr(e, "error_type", None) == "prohibited_content" or "content blocked" in err_text or "recitation" in err_text:
                self.last_vision_translation_finish_reason = "prohibited_content"
                print("   🚫 Vision OCR translation hit content filter/prohibited content")
                return None
            if getattr(e, "error_type", None) == "cancelled" or "stopped by user" in err_text or "cancelled" in err_text:
                self.last_vision_translation_finish_reason = "cancelled"
                return None
            raise

        finish_reason = None
        if isinstance(response, tuple):
            finish_reason = response[1] if len(response) > 1 else None
            response = response[0]

        if finish_reason in ("content_filter", "prohibited_content", "error"):
            self.last_vision_translation_finish_reason = finish_reason
            return None

        if hasattr(response, 'content'):
            translated = response.content
        elif hasattr(response, 'text'):
            translated = response.text
        else:
            translated = str(response)

        translated = (translated or "").strip()
        if finish_reason:
            self.last_vision_translation_finish_reason = finish_reason
        if single_pass_enabled:
            try:
                from TransateKRtoEN import _split_single_pass_glossary_response, _persist_single_pass_glossary
                translated, glossary_block = _split_single_pass_glossary_response(translated)
                if glossary_block:
                    _persist_single_pass_glossary(
                        self.output_dir,
                        glossary_block,
                        chapter_num=getattr(self, 'current_chapter_num', None),
                        source_text=ocr_text,
                        chapter_file=os.path.basename(getattr(self, 'current_image_path', '')),
                    )
            except Exception as e:
                print(f"   ⚠️ Vision Single Pass Glossary persistence skipped: {e}")
        if self.remove_ai_artifacts != "off":
            translated = self._clean_translation_response(translated)
        translated = self._sanitize_unicode_characters(translated)
        return translated.strip() or None

    def _extract_context_header_text(self, context):
        if not context:
            return ""
        match = re.search(
            r"\[CHAPTER_HEADER_TEXT\]\s*(.*?)\s*\[/CHAPTER_HEADER_TEXT\]",
            str(context),
            flags=re.IGNORECASE | re.DOTALL,
        )
        if not match:
            return ""
        lines = [line.strip() for line in match.group(1).splitlines() if line.strip()]
        return "\n".join(lines).strip()

    def _inject_context_headers_into_ocr(self, ocr_text, context):
        header_text = self._extract_context_header_text(context)
        if not header_text:
            return ocr_text
        source = (ocr_text or "").strip()
        if not source:
            return header_text
        if header_text in source:
            return source
        print("   📝 Injected chapter header HTML into OCR text")
        return f"{header_text}\n\n{source}"

    def _prepare_vision_ocr_translation_prompt(self, ocr_text, check_stop_fn):
        """Apply automatic glossary modes to OCR text before the final translation call."""
        mode = (os.getenv("AUTO_GLOSSARY_MODE") or "").strip().lower()
        if mode in ("single_pass", "single-pass", "off", "no_glossary", "off_fuzzy_automap", "off_no_automap", ""):
            return self.system_prompt
        if mode not in ("minimal", "balanced", "full"):
            return self.system_prompt

        glossary_path = None
        if os.getenv("VISION_GLOSSARY_PREPASS_DONE", "0") != "1":
            glossary_path = self._ensure_vision_ocr_glossary(ocr_text, mode, check_stop_fn)
        if not glossary_path:
            glossary_path = self._find_vision_glossary_file()
        if not glossary_path:
            return self.system_prompt

        try:
            with open(glossary_path, 'r', encoding='utf-8') as f:
                glossary_text = f.read()
            if not glossary_text.strip():
                return self.system_prompt
            append_prompt = os.getenv(
                "GLOSSARY_APPEND_PROMPT",
                "- Follow this reference glossary for consistent translation (Do not output any raw entries):\n"
            )
            print(f"   📑 Vision OCR glossary applied: {os.path.basename(glossary_path)}")
            return f"{self.system_prompt}\n\n{append_prompt}\n{glossary_text}"
        except Exception as e:
            print(f"   ⚠️ Failed to append Vision OCR glossary: {e}")
            return self.system_prompt

    def _find_vision_glossary_file(self):
        try:
            glossary_dir = os.path.join(self.output_dir, "Glossary")
            candidates = [
                os.path.join(self.output_dir, "glossary.csv"),
                os.path.join(self.output_dir, "glossary.json"),
                os.path.join(glossary_dir, "vision_ocr_glossary.csv"),
                os.path.join(glossary_dir, "vision_ocr_glossary.json"),
            ]
            if os.path.isdir(glossary_dir):
                for name in os.listdir(glossary_dir):
                    if name.lower().endswith((".csv", ".json")):
                        candidates.append(os.path.join(glossary_dir, name))
            for path in candidates:
                if os.path.exists(path) and os.path.getsize(path) > 0:
                    return path
        except Exception:
            pass
        return None

    def _ensure_vision_ocr_glossary(self, ocr_text, mode, check_stop_fn):
        """Generate/update glossary entries from OCR text for Vision mode."""
        if not ocr_text or not ocr_text.strip():
            return None
        try:
            import hashlib
            text_hash = hashlib.sha256(ocr_text.encode('utf-8', errors='ignore')).hexdigest()
            if text_hash in self._vision_glossary_processed_hashes:
                return self._find_vision_glossary_file()

            if check_stop_fn and check_stop_fn():
                return None

            import extract_glossary_from_epub as glossary_extractor
            glossary_dir = os.path.join(self.output_dir, "Glossary")
            os.makedirs(glossary_dir, exist_ok=True)
            json_path = os.path.join(glossary_dir, "vision_ocr_glossary.json")
            csv_path = os.path.join(glossary_dir, "vision_ocr_glossary.csv")

            system_prompt, user_prompt = glossary_extractor.build_prompt(ocr_text)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            retry_timeout_enabled = os.getenv("RETRY_TIMEOUT", "1") == "1"
            chunk_timeout = int(os.getenv("CHUNK_TIMEOUT", "1800")) if retry_timeout_enabled else None
            glossary_tokens = os.getenv("GLOSSARY_MAX_OUTPUT_TOKENS", "").strip()
            max_tokens = int(glossary_tokens) if glossary_tokens and glossary_tokens != "-1" else self.image_max_tokens
            temperature = float(os.getenv("GLOSSARY_TEMPERATURE", str(self.temperature)) or self.temperature)

            print(f"   📑 Vision OCR auto glossary ({mode}): generating entries from OCR text...")
            response = send_text_with_interrupt(
                self.client,
                messages,
                temperature,
                max_tokens,
                stop_check_fn=check_stop_fn,
                chunk_timeout=chunk_timeout,
                context='glossary',
            )
            if isinstance(response, tuple):
                response = response[0]
            if hasattr(response, 'content'):
                raw = response.content
            elif hasattr(response, 'text'):
                raw = response.text
            else:
                raw = str(response)

            parsed = glossary_extractor.parse_api_response(raw or "")
            valid = []
            for entry in parsed:
                if glossary_extractor.validate_extracted_entry(entry):
                    if isinstance(entry.get("raw_name"), str):
                        entry["raw_name"] = entry["raw_name"].strip()
                    valid.append(entry)
            if not valid:
                print("   📑 Vision OCR auto glossary: no valid entries found")
                return self._find_vision_glossary_file()

            existing = []
            for path in (csv_path, json_path):
                if os.path.exists(path):
                    try:
                        existing.extend(glossary_extractor._load_glossary_file(path))
                    except Exception:
                        pass
            deduped = glossary_extractor.skip_duplicate_entries(existing + valid, output_dir=glossary_dir)
            glossary_extractor.save_glossary_json(deduped, json_path)
            glossary_extractor.save_glossary_csv(deduped, json_path)
            self._vision_glossary_processed_hashes.add(text_hash)
            print(f"   📑 Vision OCR auto glossary: saved {len(valid)} new entries ({len(deduped)} total)")
            return csv_path if os.path.exists(csv_path) else json_path
        except Exception as e:
            print(f"   ⚠️ Vision OCR auto glossary failed: {e}")
            return self._find_vision_glossary_file()


    def _image_to_bytes_with_compression(self, img):
        """Convert PIL Image to bytes with compression settings applied"""
        # Check if compression is enabled
        if os.getenv("ENABLE_IMAGE_COMPRESSION", "0") == "1":
            # Get compression settings
            format_setting = os.getenv("IMAGE_COMPRESSION_FORMAT", "auto")
            webp_quality = int(os.getenv("WEBP_QUALITY", "85"))
            jpeg_quality = int(os.getenv("JPEG_QUALITY", "85"))
            png_compression = int(os.getenv("PNG_COMPRESSION", "6"))
            preserve_transparency = os.getenv("PRESERVE_TRANSPARENCY", "0") == "1"
            optimize_for_ocr = os.getenv("OPTIMIZE_FOR_OCR", "1") == "1"
            
            # Store original mode for GIF handling
            original_mode = img.mode
            transparency_info = None
            
            # Check if this is a chunk from a GIF (palette mode)
            is_gif_chunk = img.mode in ('P', 'L')
            
            # Apply OCR optimization if enabled
            if optimize_for_ocr:
                # Handle GIF chunks in palette mode
                if is_gif_chunk:
                    print(f"   🎨 Chunk is in {img.mode} mode - converting for optimization")
                    
                    if img.mode == 'P':
                        # Preserve transparency info if present
                        transparency_info = img.info.get('transparency', None)
                        # Convert to RGBA if has transparency, otherwise RGB
                        if transparency_info is not None:
                            img = img.convert('RGBA')
                        else:
                            img = img.convert('RGB')
                    elif img.mode == 'L':
                        img = img.convert('RGB')
                
                # Apply enhancements (now safe for all modes)
                from PIL import ImageEnhance
                enhancer = ImageEnhance.Contrast(img)
                img = enhancer.enhance(1.2)
                enhancer = ImageEnhance.Sharpness(img)
                img = enhancer.enhance(1.1)
                
                # Extra sharpening for GIF-sourced chunks
                if is_gif_chunk:
                    img = enhancer.enhance(1.2)
                    print(f"   ✨ Applied extra sharpening for GIF-sourced chunk")
            
            # Auto-select format if needed
            if format_setting == "auto":
                # Check if we should preserve original format
                preserve_original_format = os.getenv("PRESERVE_ORIGINAL_FORMAT", "0") == "1"
                original_format = os.getenv("ORIGINAL_IMAGE_FORMAT", "").lower()
                
                # If preserving format and we know the original format
                if preserve_original_format and original_format:
                    if original_format == 'gif':
                        format_setting = 'gif'
                        print(f"   🎞️ Preserving GIF format for chunk")
                    elif original_format in ['png', 'jpeg', 'jpg', 'webp']:
                        format_setting = original_format.replace('jpg', 'jpeg')
                        print(f"   📸 Preserving {format_setting.upper()} format for chunk")
                    else:
                        # Fallback to PNG for unknown formats
                        format_setting = "png"
                        print(f"   📸 Using PNG for chunk (unknown original format: {original_format})")
                # Legacy fallback: If chunk is in palette mode and preserve format is on, assume GIF
                elif preserve_original_format and is_gif_chunk:
                    format_setting = 'gif'
                    print(f"   🎞️ Preserving GIF format for chunk (palette mode detected)")
                else:
                    # Check image characteristics for auto-selection
                    has_transparency = img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info)
                    
                    # For chunks, prefer WebP for best compression unless transparency is needed
                    if has_transparency and preserve_transparency:
                        format_setting = "png"  # PNG for transparency
                    else:
                        format_setting = "webp"  # WebP for best compression
                    
                    print(f"   🎯 Auto-selected format for chunk: {format_setting}")
            
            # Use the selected format with compression
            if format_setting == "webp":
                print(f"   🗜️ Compressing chunk as WebP (quality: {webp_quality})")
                return self._image_to_bytes(img, format='WEBP', quality=webp_quality)
            elif format_setting == "jpeg":
                print(f"   🗜️ Compressing chunk as JPEG (quality: {jpeg_quality})")
                return self._image_to_bytes(img, format='JPEG', quality=jpeg_quality)
            elif format_setting == "png":
                # PNG uses compression level, not quality
                print(f"   🗜️ Compressing chunk as PNG (compression: {png_compression})")
                img_bytes = io.BytesIO()
                img.save(img_bytes, format='PNG', compress_level=png_compression, optimize=True)
                img_bytes.seek(0)
                data = img_bytes.read()
                
                # Log compression info
                print(f"   📊 Chunk size: {len(data) / 1024:.1f}KB")
                return data
            elif format_setting == "gif":
                # GIF format for chunks
                print(f"   🎞️ Saving chunk as GIF")
                img_bytes = io.BytesIO()
                # Convert to palette mode if needed
                if img.mode not in ('P', 'L'):
                    img = img.quantize(colors=256, method=2)  # MEDIANCUT
                img.save(img_bytes, format='GIF', optimize=True)
                img_bytes.seek(0)
                data = img_bytes.read()
                
                # Log compression info
                print(f"   📊 Chunk size: {len(data) / 1024:.1f}KB")
                return data
        
        # Default: use existing method without compression
        return self._image_to_bytes(img)

    def _image_to_bytes(self, img, format='PNG', quality=None):
            """Convert PIL Image to bytes with various format options"""
            img_bytes = io.BytesIO()
            
            if format == 'WEBP':
                # WebP is much better for manga/text images
                # Ensure RGB mode for WebP (no RGBA in some cases)
                if img.mode == 'RGBA' and not os.getenv("PRESERVE_TRANSPARENCY", "0") == "1":
                    # Create white background
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    background.paste(img, mask=img.split()[3])
                    img = background
                elif img.mode not in ['RGB', 'L', 'RGBA']:
                    img = img.convert('RGB')
                    
                if quality:
                    img.save(img_bytes, format='WEBP', quality=quality, method=6)
                else:
                    img.save(img_bytes, format='WEBP', lossless=True)
            elif format == 'JPEG':
                # JPEG doesn't support transparency, so convert RGBA to RGB
                if img.mode == 'RGBA':
                    # Create white background
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    background.paste(img, mask=img.split()[3])
                    img = background
                elif img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Save as JPEG with specified quality
                if quality:
                    img.save(img_bytes, format='JPEG', quality=quality, optimize=True, 
                            progressive=os.getenv("PROGRESSIVE_ENCODING", "1") == "1")
                else:
                    img.save(img_bytes, format='JPEG', quality=85, optimize=True)
            elif format == 'GIF':
                # GIF format handling
                if img.mode not in ('P', 'L'):
                    # Convert to palette mode for GIF
                    img = img.quantize(colors=256, method=2)  # MEDIANCUT method
                
                # Save as GIF
                img.save(img_bytes, format='GIF', optimize=True)
            else:
                # Default PNG format
                compress_level = int(os.getenv("PNG_COMPRESSION", "6"))
                img.save(img_bytes, format='PNG', compress_level=compress_level, optimize=True)
            
            img_bytes.seek(0)
            data = img_bytes.read()
            
            # Log the size for debugging
            size_kb = len(data) / 1024
            if size_kb > 0 and not getattr(self, "_suppress_image_detail_logs", False):  # Log if file is over 0kb
                print(f"   💾 Image Chunk Size: {size_kb:.1f}KB ")
            
            return data

    def _vision_ocr_batch_enabled(self) -> bool:
        return os.getenv("VISION_OCR_BATCH_TRANSLATION", "1").strip().lower() in ("1", "true", "yes", "on")

    def _vision_ocr_batch_size(self) -> int:
        try:
            return max(1, int(os.getenv("VISION_OCR_BATCH_SIZE", "10")))
        except Exception:
            return 1

    def _image_chunk_overlap_pixels(self) -> int:
        try:
            overlap_percentage = float(os.getenv('IMAGE_CHUNK_OVERLAP_PERCENT', '1'))
            return max(0, int(self.chunk_height * (overlap_percentage / 100)))
        except Exception:
            return 100

    def _ocr_cache_signature(self):
        """Settings that change which images/chunks the saved OCR text represents."""
        return {
            "cache_version": 1,
            "webnovel_min_height": str(os.getenv("WEBNOVEL_MIN_HEIGHT", str(self.webnovel_min_height))),
            "max_images_per_chapter": str(os.getenv("MAX_IMAGES_PER_CHAPTER", "10")),
            "image_chunk_height": str(os.getenv("IMAGE_CHUNK_HEIGHT", str(self.chunk_height))),
            "image_chunk_overlap_percent": str(os.getenv("IMAGE_CHUNK_OVERLAP_PERCENT", "1")),
        }

    def _ocr_cache_path(self):
        return os.path.join(getattr(self, 'ocr_dir', os.path.join(self.output_dir, "OCR")), ".cache")

    def _write_ocr_cache(self, signature=None):
        try:
            os.makedirs(self.ocr_dir, exist_ok=True)
            payload = {
                "settings": signature or self._ocr_cache_signature(),
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            with open(self._ocr_cache_path(), 'w', encoding='utf-8') as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"   ⚠️ Could not write OCR cache metadata: {e}")

    def _clear_ocr_cache_outputs(self):
        """Remove cached OCR artifacts while staying inside output/OCR."""
        ocr_dir = os.path.abspath(getattr(self, 'ocr_dir', os.path.join(self.output_dir, "OCR")))
        for name in ("chunks", "combined", "chapters", "full", "single"):
            target = os.path.abspath(os.path.join(ocr_dir, name))
            if not (target == ocr_dir or target.startswith(ocr_dir + os.sep)):
                continue
            if not os.path.isdir(target):
                continue
            for root, dirs, files in os.walk(target, topdown=False):
                for filename in files:
                    try:
                        os.remove(os.path.join(root, filename))
                    except Exception:
                        pass
                for dirname in dirs:
                    try:
                        os.rmdir(os.path.join(root, dirname))
                    except Exception:
                        pass
            try:
                os.rmdir(target)
            except Exception:
                pass

    def _ensure_ocr_cache_valid(self):
        signature = self._ocr_cache_signature()
        cache_path = self._ocr_cache_path()
        previous = None
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    previous = json.load(f).get("settings")
            except Exception:
                previous = None

        if previous is None:
            if any(os.path.isdir(os.path.join(self.ocr_dir, name)) for name in ("chunks", "combined", "chapters", "full", "single")):
                print("   🔁 OCR cache metadata missing; invalidating saved OCR text")
                self._clear_ocr_cache_outputs()
            self._write_ocr_cache(signature)
            return

        if previous != signature:
            print("   🔁 OCR cache settings changed; invalidating saved OCR text")
            print(f"      Previous: {previous}")
            print(f"      Current:  {signature}")
            self._clear_ocr_cache_outputs()
            self._write_ocr_cache(signature)

    def _safe_ocr_stem(self, image_basename=None, image_idx=None, chapter_num=None):
        image_basename = image_basename or os.path.basename(getattr(self, 'current_image_path', 'image'))
        stem, _ = os.path.splitext(image_basename)
        stem = re.sub(r'[<>:"/\\|?*\x00-\x1F]', '_', stem).strip(' ._') or 'image'
        if sys.platform == "darwin":
            stem = unicodedata.normalize("NFC", stem)
            encoded = stem.encode("utf-8", errors="ignore")
            if len(encoded) > 180:
                digest = hashlib.sha1(encoded).hexdigest()[:10]
                trimmed = []
                byte_count = 0
                for char in stem:
                    char_bytes = char.encode("utf-8", errors="ignore")
                    if byte_count + len(char_bytes) > 160:
                        break
                    trimmed.append(char)
                    byte_count += len(char_bytes)
                stem = (''.join(trimmed).rstrip(' ._') or 'image') + f"_{digest}"
        chapter_num = chapter_num if chapter_num is not None else getattr(self, 'current_chapter_num', None)
        image_idx = image_idx if image_idx is not None else getattr(self, 'current_image_index', None)
        parts = []
        if chapter_num is not None:
            try:
                parts.append(f"ch{int(chapter_num):03d}")
            except Exception:
                parts.append(f"ch{chapter_num}")
        if image_idx is not None:
            try:
                parts.append(f"img{int(image_idx):02d}")
            except Exception:
                parts.append(f"img{image_idx}")
        parts.append(stem)
        return "_".join(str(part) for part in parts if str(part))

    def _save_ocr_text(self, text, kind="chunks", image_basename=None, chunk_idx=None, image_idx=None, chapter_num=None):
        """Save OCR text under output/OCR for inspection and reuse."""
        if not text:
            return None
        try:
            ocr_dir = getattr(self, 'ocr_dir', os.path.join(self.output_dir, "OCR"))
            target_dir = os.path.join(ocr_dir, kind)
            os.makedirs(target_dir, exist_ok=True)
            suffix = f"_chunk_{chunk_idx:03d}" if chunk_idx is not None else ""
            filename = f"{self._safe_ocr_stem(image_basename, image_idx=image_idx, chapter_num=chapter_num)}{suffix}.txt"
            path = os.path.join(target_dir, filename)
            with open(path, 'w', encoding='utf-8') as f:
                f.write(text)
            print(f"   Saved OCR text: {path}")
            return path
        except Exception as e:
            print(f"   Could not save OCR text: {e}")
            return None

    def _ocr_text_path(self, kind="chunks", image_basename=None, chunk_idx=None, image_idx=None, chapter_num=None):
        ocr_dir = getattr(self, 'ocr_dir', os.path.join(self.output_dir, "OCR"))
        suffix = f"_chunk_{chunk_idx:03d}" if chunk_idx is not None else ""
        filename = f"{self._safe_ocr_stem(image_basename, image_idx=image_idx, chapter_num=chapter_num)}{suffix}.txt"
        return os.path.join(ocr_dir, kind, filename)

    def _load_saved_ocr_text(self, kind="chunks", image_basename=None, chunk_idx=None, image_idx=None, chapter_num=None):
        try:
            path = self._ocr_text_path(
                kind=kind,
                image_basename=image_basename,
                chunk_idx=chunk_idx,
                image_idx=image_idx,
                chapter_num=chapter_num,
            )
            if not os.path.exists(path):
                return None
            with open(path, 'r', encoding='utf-8') as f:
                text = f.read()
            return text if text.strip() else None
        except Exception:
            return None

    def _combine_vision_ocr_chunks(self, ocr_chunks):
        """Combine OCR chunks while trimming obvious overlap duplicates."""
        combined_lines = []
        for chunk_text in ocr_chunks:
            lines = [line.rstrip() for line in (chunk_text or "").splitlines()]
            lines = [line for line in lines if line.strip()]
            if not lines:
                continue
            max_overlap = min(20, len(combined_lines), len(lines))
            overlap = 0
            for n in range(max_overlap, 0, -1):
                if combined_lines[-n:] == lines[:n]:
                    overlap = n
                    break
            if overlap:
                lines = lines[overlap:]
            for line in lines:
                if combined_lines and combined_lines[-1].strip() == line.strip():
                    continue
                combined_lines.append(line)
        return "\n".join(combined_lines).strip()

    def _process_image_chunks_ocr_first_combined(self, img, width, height, context, check_stop_fn, translate_after=True):
        """OCR all chunks first, then translate the combined OCR text once."""
        num_chunks = (height + self.chunk_height - 1) // self.chunk_height
        overlap = self._image_chunk_overlap_pixels()
        batch_enabled = self._vision_ocr_batch_enabled()
        batch_size = min(num_chunks, self._vision_ocr_batch_size()) if batch_enabled else 1

        print(f"   Vision OCR-first: splitting tall image into {num_chunks} OCR chunks, then translating combined OCR once")
        if batch_enabled and batch_size > 1:
            print(f"   Vision OCR batch mode enabled: {batch_size} parallel OCR workers")
            print("   Stop will take effect after the current OCR batch completes")
        else:
            print("   Stop will take effect after the current OCR chunk completes")

        save_debug_chunks = os.getenv('SAVE_CLEANED_IMAGES', '0') == '1'
        save_compressed_chunks = os.getenv('SAVE_COMPRESSED_IMAGES', '0') == '1'
        debug_dir = None
        if save_debug_chunks or save_compressed_chunks:
            debug_dir = os.path.join(self.ocr_dir, "debug_chunks")
            os.makedirs(debug_dir, exist_ok=True)
            print(f"   Debug mode: Saving OCR chunks to {debug_dir}")

        image_basename = os.path.basename(self.current_image_path) if hasattr(self, 'current_image_path') else str(hash(str(img)))

        image_chunk_prompt_template = os.getenv(
            "IMAGE_CHUNK_PROMPT",
            "This is part {chunk_idx} of {total_chunks} of a longer image. You must maintain the narrative flow with the previous chunks while following all system prompt guidelines previously mentioned. {context}"
        )

        chunk_jobs = []
        ocr_by_index = {}
        was_stopped = False

        for i in range(num_chunks):
            disk_ocr = self._load_saved_ocr_text(kind="chunks", image_basename=image_basename, chunk_idx=i + 1)
            if disk_ocr:
                if self._is_ocr_no_response(disk_ocr):
                    print(f"   Skipping OCR chunk {i+1}/{num_chunks}; cached OCR says cover/illustration")
                else:
                    ocr_by_index[i] = disk_ocr
                    print(f"   Skipping OCR chunk {i+1}/{num_chunks}; found existing OCR/chunks file")
                continue

            if check_stop_fn and check_stop_fn():
                print(f"   Stopped before OCR chunk {i+1}/{num_chunks}")
                was_stopped = True
                break

            start_y = max(0, i * self.chunk_height - (overlap if i > 0 else 0))
            end_y = min(height, (i + 1) * self.chunk_height)
            current_filename = os.path.basename(self.current_image_path) if hasattr(self, 'current_image_path') else 'unknown'
            print(f"   OCR chunk {i+1}/{num_chunks} (y: {start_y}-{end_y}) for {current_filename}")
            if self.log_callback and hasattr(self.log_callback, '__self__') and hasattr(self.log_callback.__self__, 'append_chunk_progress'):
                self.log_callback.__self__.append_chunk_progress(
                    i + 1,
                    num_chunks,
                    "image",
                    f"Image OCR: {current_filename}"
                )

            chunk = img.crop((0, start_y, width, end_y))
            chunk_bytes = self._image_to_bytes_with_compression(chunk)

            if save_debug_chunks and debug_dir:
                chunk_path = os.path.join(debug_dir, f"ocr_chunk_{i+1}_original.png")
                chunk.save(chunk_path)
                print(f"   Saved OCR chunk: {chunk_path}")

            if save_compressed_chunks and debug_dir and os.getenv("ENABLE_IMAGE_COMPRESSION", "0") == "1":
                compressed_dir = os.path.join(self.ocr_dir, "compressed", "chunks")
                os.makedirs(compressed_dir, exist_ok=True)
                compressed_chunk_path = os.path.join(compressed_dir, f"ocr_chunk_{i+1}_compressed.bin")
                with open(compressed_chunk_path, 'wb') as f:
                    f.write(chunk_bytes)
                print(f"   Saved compressed OCR chunk: {compressed_chunk_path}")

            chunk_prompt = image_chunk_prompt_template.format(
                chunk_idx=i + 1,
                total_chunks=num_chunks,
                context="OCR this chunk only. Preserve visible source text order. Output main/base characters only; ignore attached pinyin/romaji/furigana/Jyutping readings. Do not translate."
            )
            chunk_jobs.append((i, chunk_bytes, chunk_prompt))

        def _ocr_job(job):
            i, chunk_bytes, chunk_prompt = job
            if check_stop_fn and check_stop_fn():
                return i, None
            print(f"   Step 1/2: OCR chunk {i+1}/{num_chunks} with dedicated Vision OCR prompt...")
            ocr_text = self._call_vision_ocr_api(chunk_bytes, chunk_prompt, check_stop_fn, i + 1)
            if self._is_ocr_no_response(ocr_text):
                print(f"   OCR chunk {i+1}/{num_chunks} marked as cover/illustration; excluding from combined OCR")
                return i, None
            if ocr_text:
                print(f"   OCR chunk {i+1}/{num_chunks} complete ({len(ocr_text)} chars)")
            else:
                print(f"   OCR chunk {i+1}/{num_chunks} returned no text")
            return i, ocr_text

        previous_batch_env = os.environ.get("BATCH_TRANSLATION")
        try:
            if batch_enabled and batch_size > 1 and len(chunk_jobs) > 1:
                os.environ["BATCH_TRANSLATION"] = "1"
                from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
                from unified_api_client import UnifiedClient, UnifiedClientError
                batching_mode = os.getenv("BATCHING_MODE", "aggressive").strip().lower()
                if batching_mode not in ("aggressive", "direct", "conservative"):
                    batching_mode = "aggressive"

                if batching_mode == "aggressive":
                    print(f"   Vision OCR no-batching mode: keeping {batch_size} OCR chunk worker slot(s) filled")
                    executor = ThreadPoolExecutor(max_workers=batch_size)
                    force_cancelled = False
                    try:
                        pending = {}
                        next_job_idx = 0

                        def _submit_next_ocr_chunk():
                            nonlocal next_job_idx
                            if next_job_idx >= len(chunk_jobs):
                                return False
                            job = chunk_jobs[next_job_idx]
                            pending[executor.submit(_ocr_job, job)] = job
                            next_job_idx += 1
                            return True

                        while len(pending) < batch_size and _submit_next_ocr_chunk():
                            pass

                        while pending:
                            done, _ = wait(pending.keys(), timeout=0.5, return_when=FIRST_COMPLETED)
                            if not done:
                                if os.environ.get("TRANSLATION_CANCELLED") == "1":
                                    done = set()
                                else:
                                    continue
                            for future in done:
                                pending.pop(future, None)
                                i, ocr_text = future.result()
                                if ocr_text:
                                    ocr_by_index[i] = ocr_text

                            if os.environ.get("TRANSLATION_CANCELLED") == "1":
                                force_cancelled = True
                                was_stopped = True
                                try:
                                    UnifiedClient.hard_cancel_all()
                                except Exception:
                                    pass
                                for future in pending:
                                    future.cancel()
                                raise UnifiedClientError("Vision OCR batch stopped by user", error_type="cancelled")

                            while len(pending) < batch_size and _submit_next_ocr_chunk():
                                pass
                    finally:
                        executor.shutdown(wait=not force_cancelled, cancel_futures=True)
                else:
                    if batching_mode == "conservative":
                        group_multiplier = max(1, int(os.getenv("BATCH_GROUP_SIZE", "3") or "3"))
                        fixed_group_size = min(len(chunk_jobs), batch_size * group_multiplier)
                        print(f"   Vision OCR conservative batching: group size {fixed_group_size}, parallel {batch_size}")
                    else:
                        fixed_group_size = batch_size
                        print(f"   Vision OCR direct batching: group size {fixed_group_size}, parallel {batch_size}")

                    for batch_start in range(0, len(chunk_jobs), fixed_group_size):
                        if check_stop_fn and check_stop_fn():
                            was_stopped = True
                            break
                        current_batch = chunk_jobs[batch_start:batch_start + fixed_group_size]
                        print(f"   OCR group {batch_start // fixed_group_size + 1}: {len(current_batch)} chunk request(s)")
                        executor = ThreadPoolExecutor(max_workers=min(batch_size, len(current_batch)))
                        force_cancelled = False
                        try:
                            pending = {executor.submit(_ocr_job, job) for job in current_batch}
                            while pending:
                                done, pending = wait(pending, timeout=0.5, return_when=FIRST_COMPLETED)
                                for future in done:
                                    i, ocr_text = future.result()
                                    if ocr_text:
                                        ocr_by_index[i] = ocr_text

                                if os.environ.get("TRANSLATION_CANCELLED") == "1":
                                    force_cancelled = True
                                    was_stopped = True
                                    try:
                                        UnifiedClient.hard_cancel_all()
                                    except Exception:
                                        pass
                                    for future in pending:
                                        future.cancel()
                                    raise UnifiedClientError("Vision OCR batch stopped by user", error_type="cancelled")
                        finally:
                            executor.shutdown(wait=not force_cancelled, cancel_futures=True)

                        if check_stop_fn and check_stop_fn():
                            was_stopped = True
                            break
            else:
                for job in chunk_jobs:
                    i, ocr_text = _ocr_job(job)
                    if ocr_text:
                        ocr_by_index[i] = ocr_text
                    if i < num_chunks - 1 and not was_stopped:
                        self._api_delay_with_stop_check(check_stop_fn)
                    if check_stop_fn and check_stop_fn():
                        was_stopped = True
                        break
        finally:
            if previous_batch_env is None:
                os.environ.pop("BATCH_TRANSLATION", None)
            else:
                os.environ["BATCH_TRANSLATION"] = previous_batch_env

        ordered_ocr = [ocr_by_index[i] for i in range(num_chunks) if ocr_by_index.get(i)]
        if not ordered_ocr:
            self._preserve_current_image = True
            print("   No translatable OCR chunks from tall image; preserving original image")
            return None

        combined_ocr = self._combine_vision_ocr_chunks(ordered_ocr)
        if not combined_ocr:
            print("   Combined OCR was empty")
            return None
        combined_ocr = self._inject_context_headers_into_ocr(combined_ocr, context)
        self._save_ocr_text(combined_ocr, kind="combined", image_basename=image_basename)
        if not translate_after:
            return combined_ocr

        print(f"   Step 2/2: Translating combined OCR from {len(ordered_ocr)}/{num_chunks} chunks ({len(combined_ocr)} chars)...")
        combined_prompt = (
            f"The OCR text below was assembled from {len(ordered_ocr)} tall-image chunk(s). "
            "Translate it as one continuous passage, preserving narrative flow and removing OCR-only duplication from chunk overlap."
        )
        translated = self._translate_ocr_text(combined_ocr, combined_prompt, check_stop_fn)
        if translated:
            print(f"   Combined OCR text translated ({len(translated)} chars)")
            if was_stopped:
                translated += "\n\n[TRANSLATION STOPPED BY USER]"
        return translated

    def _process_image_chunks(self, img, width, height, context, check_stop_fn):
        """Process a tall image by splitting it into chunks with contextual support"""
        if self._use_ocr_first_pipeline():
            return self._process_image_chunks_ocr_first_combined(img, width, height, context, check_stop_fn)

        num_chunks = (height + self.chunk_height - 1) // self.chunk_height
        overlap = 100  # Pixels of overlap between chunks
        
        print(f"   ✂️ Image too tall ({height}px), splitting into {num_chunks} chunks of {self.chunk_height}px...")
        
        # Clear context for new image
        self.image_chunk_context = []
        
        # Add retry info if enabled
        if os.getenv("RETRY_TIMEOUT", "1") == "1":
            timeout_seconds = int(os.getenv("CHUNK_TIMEOUT", "1800"))
            print(f"   ⏱️ Auto-retry enabled: Will retry if chunks take > {timeout_seconds}s")
        
        print(f"   ⏳ This may take {num_chunks * 30}-{num_chunks * 60} seconds to complete")
        print(f"   ℹ️ Stop will take effect after current chunk completes")
        
        # Check if we should save debug chunks
        save_debug_chunks = os.getenv('SAVE_CLEANED_IMAGES', '0') == '1'
        save_compressed_chunks = os.getenv('SAVE_COMPRESSED_IMAGES', '0') == '1'
        
        if save_debug_chunks or save_compressed_chunks:
            debug_dir = os.path.join(self.ocr_dir, "debug_chunks")
            os.makedirs(debug_dir, exist_ok=True)
            print(f"   🔍 Debug mode: Saving chunks to {debug_dir}")
        
        # Load progress - maintaining full structure
        prog = self.load_progress()
        
        # Create unique key for this image - include chapter info if available
        image_basename = os.path.basename(self.current_image_path) if hasattr(self, 'current_image_path') else str(hash(str(img)))
        
        # Try to extract chapter number from context or path
        chapter_num = None
        if hasattr(self, 'current_chapter_num'):
            chapter_num = self.current_chapter_num
        else:
            # Try to extract from filename
            import re
            match = re.search(r'ch(?:apter)?[\s_-]*(\d+)', image_basename, re.IGNORECASE)
            if match:
                chapter_num = match.group(1)
        
        # Create a more unique key that includes chapter info
        if chapter_num:
            image_key = f"ch{chapter_num}_{image_basename}"
        else:
            image_key = image_basename
        
        # Initialize image chunk tracking
        if "image_chunks" not in prog:
            prog["image_chunks"] = {}
            
        if image_key not in prog["image_chunks"]:
            prog["image_chunks"][image_key] = {
                "total": num_chunks,
                "completed": [],
                "chunks": {},
                "height": height,
                "width": width,
                "chapter": chapter_num,  # Store chapter association
                "filename": image_basename
            }
        
        all_translations = []
        was_stopped = False
        
        # Process chunks
        for i in range(num_chunks):
            # Check if this chunk was already translated
            if i in prog["image_chunks"][image_key]["completed"]:
                saved_chunk = prog["image_chunks"][image_key]["chunks"].get(str(i))
                if saved_chunk:
                    all_translations.append(saved_chunk)
                    print(f"   ⏭️ Chunk {i+1}/{num_chunks} already translated, skipping")
                    continue
            
            # Check for stop before processing each chunk
            if check_stop_fn and check_stop_fn():
                print(f"   ❌ Stopped at chunk {i+1}/{num_chunks}")
                was_stopped = True
                break
            
            # Calculate chunk boundaries with overlap
            start_y = max(0, i * self.chunk_height - (overlap if i > 0 else 0))
            end_y = min(height, (i + 1) * self.chunk_height)
            
            current_filename = os.path.basename(self.current_image_path) if hasattr(self, 'current_image_path') else 'unknown'
            print(f"   📄 Processing chunk {i+1}/{num_chunks} (y: {start_y}-{end_y}) for {current_filename}")
            if self.log_callback and hasattr(self.log_callback, '__self__') and hasattr(self.log_callback.__self__, 'append_chunk_progress'):
                self.log_callback.__self__.append_chunk_progress(
                    i + 1, 
                    num_chunks, 
                    "image", 
                    f"Image: {os.path.basename(self.current_image_path) if hasattr(self, 'current_image_path') else 'unknown'}"
                )
            
            print(f"   ⏳ Estimated time: 30-60 seconds for this chunk")
                
            # Crop and process the chunk
            chunk = img.crop((0, start_y, width, end_y))
            
            # Convert chunk to bytes with compression
            chunk_bytes = self._image_to_bytes_with_compression(chunk)
            
            # Save debug chunks if enabled
            if save_debug_chunks or save_compressed_chunks:
                # Save original chunk
                if save_debug_chunks:
                    chunk_path = os.path.join(debug_dir, f"chunk_{i+1}_original.png")
                    chunk.save(chunk_path)
                    print(f"   💾 Saved original chunk: {chunk_path}")
                
                # Save compressed chunk if enabled
                if save_compressed_chunks and os.getenv("ENABLE_IMAGE_COMPRESSION", "0") == "1":
                    compressed_dir = os.path.join(self.ocr_dir, "compressed", "chunks")
                    os.makedirs(compressed_dir, exist_ok=True)
                    
                    # Use compression settings to save chunk
                    format_setting = os.getenv("IMAGE_COMPRESSION_FORMAT", "auto")
                    if format_setting == "auto":
                        format_setting = "webp"  # Default to WebP for chunks
                    
                    # Create a temporary in-memory file for the compressed chunk
                    from io import BytesIO
                    compressed_buffer = BytesIO()
                    
                    if format_setting == "webp":
                        quality = int(os.getenv("WEBP_QUALITY", "85"))
                        chunk.save(compressed_buffer, format='WEBP', quality=quality, method=6)
                        compressed_chunk_path = os.path.join(compressed_dir, f"chunk_{i+1}_compressed.webp")
                    elif format_setting == "jpeg":
                        quality = int(os.getenv("JPEG_QUALITY", "85"))
                        # Convert RGBA to RGB for JPEG
                        if chunk.mode == 'RGBA':
                            rgb_chunk = Image.new('RGB', chunk.size, (255, 255, 255))
                            rgb_chunk.paste(chunk, mask=chunk.split()[3])
                            chunk_to_save = rgb_chunk
                        else:
                            chunk_to_save = chunk
                        chunk_to_save.save(compressed_buffer, format='JPEG', quality=quality, optimize=True)
                        compressed_chunk_path = os.path.join(compressed_dir, f"chunk_{i+1}_compressed.jpg")
                    else:  # PNG
                        compress_level = int(os.getenv("PNG_COMPRESSION", "6"))
                        chunk.save(compressed_buffer, format='PNG', compress_level=compress_level, optimize=True)
                        compressed_chunk_path = os.path.join(compressed_dir, f"chunk_{i+1}_compressed.png")
                    
                    # Write the compressed chunk to disk
                    with open(compressed_chunk_path, 'wb') as f:
                        f.write(compressed_buffer.getvalue())
                    
                    # Get actual original chunk size before compression
                    chunk_buffer = BytesIO()
                    chunk.save(chunk_buffer, format='PNG')
                    actual_original_size = len(chunk_buffer.getvalue()) / 1024  # KB

                    # Log compression info
                    compressed_size = len(compressed_buffer.getvalue()) / 1024  # KB
                    compression_ratio = (1 - compressed_size / actual_original_size) * 100 if actual_original_size > 0 else 0
                    
                    print(f"   💾 Saved compressed chunk: {compressed_chunk_path}")
                    print(f"   📊 Chunk compression: {actual_original_size:.1f}KB → {compressed_size:.1f}KB ({compression_ratio:.1f}% reduction)")
            
            # Get custom image chunk prompt template from environment
            image_chunk_prompt_template = os.getenv(
                "IMAGE_CHUNK_PROMPT",
                "This is part {chunk_idx} of {total_chunks} of a longer image. You must maintain the narrative flow with the previous chunks while following all system prompt guidelines previously mentioned. {context}"
            )
            
            # Build assistant prompt for this chunk (text only)
            chunk_prompt = image_chunk_prompt_template.format(
                chunk_idx=i+1,
                total_chunks=num_chunks,
                context=""
            )
            
            # Translate chunk WITH CONTEXT; send chunk prompt as assistant message
            translation = self._call_vision_api(chunk_bytes, chunk_prompt, check_stop_fn)
            
            if translation:
                # Clean AI artifacts from chunk
                if self.remove_ai_artifacts != "off":
                    chunk_text = self._clean_translation_response(translation)
                else:
                    chunk_text = translation
                # Sanitize invalid glyphs without changing valid punctuation width/forms.
                chunk_text = self._sanitize_unicode_characters(chunk_text)
                all_translations.append(chunk_text)
                print(f"   🔍 DEBUG: Chunk {i+1} length: {len(chunk_text)} chars")
                if len(chunk_text) > 10000:  # Flag suspiciously large chunks
                    print(f"   ⚠️ WARNING: Chunk unusually large!")
                    print(f"   First 500 chars: {chunk_text[:500]}")
                    print(f"   Last 500 chars: {chunk_text[-500:]}")
                
                # Store context for next chunks
                if self.contextual_enabled:
                    self.image_chunk_context.append({
                        "assistant_prompt": chunk_prompt,
                        "assistant": chunk_text
                    })
                
                # Save chunk progress
                prog["image_chunks"][image_key]["completed"].append(i)
                prog["image_chunks"][image_key]["chunks"][str(i)] = chunk_text
                self.save_progress(prog)
                
                print(f"   ✅ Chunk {i+1} translated and saved ({len(chunk_text)} chars)")
            else:
                print(f"   ⚠️ Chunk {i+1} returned no text")
            
            # Delay between chunks if not the last one
            if i < num_chunks - 1 and not was_stopped:
                self._api_delay_with_stop_check(check_stop_fn)
                if check_stop_fn and check_stop_fn():
                    was_stopped = True
                    break
        
        # Combine all chunk translations
        if all_translations:
            translated_text = "\n\n".join(all_translations)
            if was_stopped:
                translated_text += "\n\n[TRANSLATION STOPPED BY USER]"
            print(f"   ✅ Combined {len(all_translations)} chunks into final translation")
            return translated_text
        else:
            print(f"   ❌ No successful translations from any chunks")
            return None

    def set_current_chapter(self, chapter_num):
        """Set the current chapter number for progress tracking"""
        self.current_chapter_num = chapter_num

    def _call_vision_api(self, image_data, assistant_prompt, check_stop_fn):
        """Make the actual API call for vision translation with retry support"""
        if self._use_ocr_first_pipeline():
            try:
                print("   🔎 Step 1/2: OCR image with dedicated Vision OCR prompt...")
                ocr_text = self._call_vision_ocr_api(image_data, assistant_prompt, check_stop_fn)
                if not ocr_text:
                    return None
                if self._is_ocr_no_response(ocr_text):
                    self._preserve_current_image = True
                    print("   OCR marked image as cover/illustration; preserving original image and skipping translation")
                    return None
                ocr_text = self._inject_context_headers_into_ocr(ocr_text, assistant_prompt)
                self._save_ocr_text(ocr_text, kind="single")
                print(f"   ✅ OCR complete ({len(ocr_text)} chars)")
                print("   🌐 Step 2/2: Translating OCR text...")
                translated = self._translate_ocr_text(ocr_text, assistant_prompt, check_stop_fn)
                if translated:
                    print(f"   ✅ OCR text translated ({len(translated)} chars)")
                return translated
            except Exception as e:
                err = str(e)
                if "DeepSeek OpenAI-compatible chat endpoints only accept text" in err:
                    print(
                        "   ❌ DeepSeek cannot perform the OCR image step. "
                        "Use a vision-capable model/endpoint for Vision OCR, then DeepSeek can translate the OCR text."
                    )
                raise

        # Build messages - NO HARDCODED PROMPT
        messages = [
            {"role": "system", "content": self.system_prompt}
        ]
        
        # Add context from previous chunks if contextual is enabled
        if hasattr(self, 'contextual_enabled') and self.contextual_enabled:
            if hasattr(self, 'image_chunk_context') and self.image_chunk_context:
                # Include ALL previous chunks from this image, not just last 2
                print(f"   📚 Including ALL {len(self.image_chunk_context)} previous chunks as context")
                
                for ctx in self.image_chunk_context:
                    if ctx.get("assistant_prompt"):
                        messages.append({"role": "assistant", "content": ctx["assistant_prompt"]})
                    messages.append({"role": "assistant", "content": ctx["assistant"]})
        
        # Add current chunk prompt as assistant; user message only carries the image payload (no extra text)
        if assistant_prompt and assistant_prompt.strip():
            messages.append({"role": "assistant", "content": assistant_prompt})
        # User message carries only the image payload; no empty text part
        messages.append({"role": "user"})
        if hasattr(self, 'current_chapter_num'):
            chapter_num = self.current_chapter_num
            image_idx = getattr(self, 'current_image_index', 0)
            output_filename = f"response_{chapter_num:03d}_Chapter_{chapter_num}_image_{image_idx}.html"
            self.client.set_output_filename(output_filename)        

        retry_timeout_enabled = os.getenv("RETRY_TIMEOUT", "1") == "1"
        chunk_timeout = int(os.getenv("CHUNK_TIMEOUT", "1800")) if retry_timeout_enabled else None
        max_timeout_retries = 2
        
        # Store original values
        original_max_tokens = self.image_max_tokens
        original_temp = self.temperature
        
        # Initialize retry counters
        timeout_retry_count = 0
        
        while True:
            try:
                current_max_tokens = self.image_max_tokens
                current_temp = self.temperature
                
                print(f"   🔄 Calling vision API...")
                print(f"   📊 Using temperature: {current_temp}")
                print(f"   📊 Output Token Limit: {current_max_tokens}")
                
                if chunk_timeout:
                    print(f"   ⏱️ Timeout enabled: {chunk_timeout} seconds")
                
                # Final stop check before API call
                if check_stop_fn and check_stop_fn():
                    print("   ❌ Stopped before API call")
                    return None
                
                # Use the new interrupt function
                translation_response, trans_finish = send_image_with_interrupt(
                    self.client,
                    messages,
                    image_data,
                    current_temp,
                    current_max_tokens,
                    check_stop_fn,
                    chunk_timeout,
                    'image_translation'
                )
                
                print(f"   📡 API response received, finish_reason: {trans_finish}")
                
                # Check if translation was truncated
                if trans_finish in ["length", "max_tokens"]:
                    print(f"   ⚠️ Translation was TRUNCATED! Consider increasing Max tokens.")
                    translation_response += "\n\n[TRANSLATION TRUNCATED DUE TO TOKEN LIMIT]"
                
                # Success - restore original values if they were changed
                if timeout_retry_count > 0:
                    self.image_max_tokens = original_max_tokens
                    self.temperature = original_temp
                    print(f"   ✅ Restored original settings after successful retry")
                
                return translation_response.strip()
                
            except Exception as e:
                from unified_api_client import UnifiedClientError
                error_msg = str(e)
                print(f"\n🔍 DEBUG: Image Translation Failed")
                print(f"   Error: {error_msg}")
                print(f"   Error Type: {type(e).__name__}")
                
                # Handle user stop
                if "stopped by user" in error_msg:
                    print("   ❌ Image translation stopped by user")
                    return None
                # Handle timeout specifically
                if "took" in error_msg and "timeout:" in error_msg:
                    if timeout_retry_count < max_timeout_retries:
                        timeout_retry_count += 1
                        print(f"    ⏱️ Chunk took too long, retry {timeout_retry_count}/{max_timeout_retries}")
                        
                        print(f"    🔄 Retrying")
                       
                        time.sleep(2)
                        continue
                    else:
                        print(f"   ❌ Max timeout retries reached for image")
                        # Restore original values
                        self.image_max_tokens = original_max_tokens
                        self.temperature = original_temp
                        return f"[Image Translation Error: Timeout after {max_timeout_retries} retries]"
                
                # Handle other timeouts
                elif "timed out" in error_msg and "timeout:" not in error_msg:
                    print(f"   ⚠️ {error_msg}, retrying...")
                    time.sleep(5)
                    continue
                
                # For other errors, restore values and return error
                if timeout_retry_count > 0:
                    self.image_max_tokens = original_max_tokens
                    self.temperature = original_temp
                
                print(f"   ❌ Translation failed: {e}")
                print(f"   ❌ Error type: {type(e).__name__}")
                return f"[Image Translation Error: {str(e)}]"


    def _clean_translation_response(self, response):
        """Clean AI artifacts from translation response while preserving content"""
        if not response or not response.strip():
            return response
        
        # First, preserve the original response length for debugging
        original_length = len(response)
        
        # Remove common AI prefixes - but be more careful
        lines = response.split('\n')
        
        # Check if first line is just a prefix without content
        if len(lines) > 1 and lines[0].strip() and lines[0].strip().lower() in [
            'sure', 'here', "i'll translate", 'certainly', 'okay', 
            'here is the translation:', 'translation:', "here's the translation:",
            "i'll translate the text from the image:", "let me translate that for you:"
        ]:
            # Remove only the first line if it's just a prefix
            response = '\n'.join(lines[1:]).strip()
        elif len(lines) > 1 and lines[0].strip() and any(
            lines[0].strip().lower().startswith(prefix) 
            for prefix in ['sure,', 'here,', "i'll translate", 'certainly,', 'okay,']
        ):
            # Check if the first line contains actual translation content after the prefix
            first_line = lines[0].strip()
            # Look for a colon or period that might separate prefix from content
            for sep in [':', '.', ',']:
                if sep in first_line:
                    parts = first_line.split(sep, 1)
                    if len(parts) > 1 and parts[1].strip():
                        # There's content after the separator, keep it
                        lines[0] = parts[1].strip()
                        response = '\n'.join(lines).strip()
                        break
            else:
                # No separator found with content, remove the whole first line
                response = '\n'.join(lines[1:]).strip()
        
        # Log if we removed significant content
        cleaned_length = len(response)
        if cleaned_length == 0 and original_length > 0:
            print(f"   ⚠️ WARNING: Cleaning removed all content! Original: {original_length} chars")
            print(f"   ⚠️ First 200 chars were: {response[:200]}")
        elif cleaned_length < original_length * 0.5:
            print(f"   ⚠️ WARNING: Cleaning removed >50% of content! {original_length} → {cleaned_length}")
        
        return response

    def _save_translation_debug(self, image_path, translated_text):
        """Save translation to file for debugging"""
        trans_filename = f"translated_{os.path.basename(image_path)}.txt"
        trans_dir = os.path.join(self.ocr_dir, "translations")
        os.makedirs(trans_dir, exist_ok=True)
        trans_filepath = os.path.join(trans_dir, trans_filename)
        
        try:
            with open(trans_filepath, 'w', encoding='utf-8') as f:
                f.write(translated_text)
            print(f"   💾 Saved translation to: {trans_filename}")
        except Exception as e:
            print(f"   ⚠️ Could not save translation file: {e}")

    def _remove_http_links(self, text: str) -> str:
        """Remove HTTP/HTTPS URLs from text while preserving other content"""
        # Pattern to match URLs
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+(?:\.[^\s<>"{}|\\^`\[\]]+)*'
        
        # Replace URLs with empty string
        cleaned_text = re.sub(url_pattern, '', text)
        
        # Clean up extra whitespace that may result from URL removal
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        
        return cleaned_text

    def _sanitize_unicode_characters(self, text: str) -> str:
        """Remove invalid Unicode characters and common fallback boxes"""
        if not text:
            return text
        import re
        original = text
        # Replacement character and common geometric fallbacks
        text = text.replace('\ufffd', '')
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        text = text.replace('\u2028', '\n').replace('\u2029', '\n')
        for ch in ['□','◇','◆','■','▢','▣','▤','▥','▦','▧','▨','▩']:
            text = text.replace(ch, '')
        text = re.sub(r'[\u200b-\u200f\u202a-\u202f\u205f-\u206f\ufeff]', '', text)
        text = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F-\x9F]', '', text)
        try:
            text = text.encode('utf-8', errors='ignore').decode('utf-8')
        except UnicodeError:
            pass
        # Normalize only horizontal whitespace. OCR/vision mode relies on
        # line breaks for reading order, so never collapse \n into spaces.
        text = re.sub(r'[ \t\f\v]+', ' ', text)
        text = re.sub(r' *\n *', '\n', text)
        text = re.sub(r'\n{4,}', '\n\n\n', text)
        text = text.strip()
        return text
    
    def _create_html_output(self, img_rel_path, translated_text, is_long_text, hide_label, was_stopped):
        print(f"   🔍 DEBUG: Creating HTML output")
        print(f"   Total translation length: {len(translated_text)} chars")
        if len(translated_text) > 50000:
            print(f"   ⚠️ WARNING: Translation suspiciously large!")
        """Create the final HTML output"""
        # Check if the translation is primarily a URL (only a URL and nothing else)
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+(?:\.[^\s<>"{}|\\^`\[\]]+)*'
        
        # Check if the entire content is just a URL
        url_match = re.match(r'^\s*' + url_pattern + r'\s*$', translated_text.strip())
        is_only_url = bool(url_match)
        
        # Build the label HTML if needed
        if hide_label:
            label_html = ""
            # Remove URLs from the text, but keep other content
            cleaned_text = self._remove_http_links(translated_text)
            
            # If after removing URLs there's no content left, and original was only URL
            if not cleaned_text and is_only_url:
                translated_text = "[Image contains only URL]"
            else:
                # Use the cleaned text (URLs removed, other content preserved)
                translated_text = cleaned_text
        else:
            if was_stopped:
                label_html = f'<p><em>(partial)</em></p>\n'
            else:
                label_html = ""
        
        # Build the image HTML based on type - or skip it entirely if hide_label is enabled
        if hide_label:
            # Don't include the image at all when hide_label is enabled
            image_html = ""
            css_class = "translated-text-only"
        elif is_long_text:
            image_html = f"""<details>
                <summary>📖 View Original Image</summary>
                <img src="{img_rel_path}" alt="Original image" />
            </details>"""
            css_class = "image-with-translation webnovel-image"
        else:
            image_html = f'<img src="{img_rel_path}" alt="Original image" />'
            css_class = "image-with-translation"
        
        # Combine everything
        return f"""<div class="{css_class}">
            {image_html}
            <div class="image-translation">
                {label_html}{self._format_translation_as_html(translated_text)}
            </div>
        </div>"""
            
    def _api_delay_with_stop_check(self, check_stop_fn):
        """API delay with stop checking"""
        # Check for stop during delay (split into 0.1s intervals)
        for i in range(int(self.api_delay * 10)):
            if check_stop_fn and check_stop_fn():
                return True
            time.sleep(0.1)
        return False
    
    def _format_translation_as_html(self, text: str) -> str:
        """Format translated text as HTML paragraphs"""
        # Convert to string and strip whitespace
        text = str(text).strip()
        
        # Remove various tuple wrapping patterns
        # Handle complete tuple wrapping
        if text.startswith('("') and text.endswith('")'):
            text = text[2:-2]
        elif text.startswith("('") and text.endswith("')"):
            text = text[2:-2]
        # Handle incomplete tuple wrapping (like just (" at the start)
        elif text.startswith('("'):
            text = text[2:]
        elif text.startswith("('"):
            text = text[2:]
        elif text.startswith('('):
            # Check if it looks like a tuple representation
            if len(text) > 1 and text[1] in ['"', "'"]:
                text = text[2:]  # Remove (" or ('
            else:
                text = text[1:]  # Just remove the (
        
        # Remove trailing tuple markers if present
        if text.endswith('")'):
            text = text[:-2]
        elif text.endswith("')"):
            text = text[:-2]
        elif text.endswith(')') and len(text) > 1 and text[-2] in ['"', "'"]:
            text = text[:-2]
        
        # Ensure we have actual newlines, not escaped ones
        if '\\n' in text:
            print(f"   🔧 Found literal \\n in text, converting to actual newlines")
            text = text.replace('\\n', '\n')
        
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
        
        # If no paragraphs were created (single line), wrap it
        if not html_parts and text.strip():
            html_parts.append(f'<p>{text.strip()}</p>')
        
        result = '\n'.join(html_parts)
        
        # Debug output
        print(f"   📝 Created {len(html_parts)} paragraphs from text")
        
        return result

    def _unescape_response_text(self, text):
        """Unescape text that comes back with literal \n characters"""
        if not text:
            return text
        
        # Convert to string if needed
        text = str(text)
        
        # Remove tuple wrapping if present (e.g., ('text') or ("text"))
        if text.startswith('("') and text.endswith('")'):
            text = text[2:-2]
        elif text.startswith("('") and text.endswith("')"):
            text = text[2:-2]
        elif text.startswith('(') and text.endswith(')') and len(text) > 2:
            # Check if it's a single-item tuple representation
            inner = text[1:-1].strip()
            if (inner.startswith('"') and inner.endswith('"')) or (inner.startswith("'") and inner.endswith("'")):
                text = inner[1:-1]
        
        # Handle escaped characters - convert literal \n to actual newlines
        text = text.replace('\\n', '\n')
        text = text.replace('\\t', '\t')
        text = text.replace('\\"', '"')
        text = text.replace("\\'", "'")
        text = text.replace('\\\\', '\\')
        
        return text
    
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
                    if header and ('(partial)' in header.text or '[Image text translation' in header.text):
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
