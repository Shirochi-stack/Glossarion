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
import queue
import threading
import cv2
import numpy as np
from unified_api_client import UnifiedClientError

logger = logging.getLogger(__name__)

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
    
    # Use chunk timeout if provided, otherwise use default
    timeout = chunk_timeout if chunk_timeout else 300
    check_interval = 0.5
    elapsed = 0
    
    while elapsed < timeout:
        try:
            result = result_queue.get(timeout=check_interval)
            if isinstance(result, Exception):
                raise result
            if isinstance(result, tuple):
                api_result, api_time = result
                # Check if it took too long
                if chunk_timeout and api_time > chunk_timeout:
                    raise UnifiedClientError(f"Image API call took {api_time:.1f}s (timeout: {chunk_timeout}s)")
                return api_result
            return result
        except queue.Empty:
            if stop_check_fn and stop_check_fn():
                raise UnifiedClientError("Image translation stopped by user")
            elapsed += check_interval
    
    raise UnifiedClientError(f"Image API call timed out after {timeout} seconds")

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
        self.translated_images_dir = os.path.join(output_dir, "translated_images")
        os.makedirs(self.translated_images_dir, exist_ok=True)
        self.api_delay = float(os.getenv("SEND_INTERVAL_SECONDS", "2"))
        
        # Track processed images to avoid duplicates
        self.processed_images = {}
        self.image_translations = {}
        
        # Configuration from environment
        self.process_webnovel = os.getenv("PROCESS_WEBNOVEL_IMAGES", "1") == "1"
        self.webnovel_min_height = int(os.getenv("WEBNOVEL_MIN_HEIGHT", "1000"))
        self.image_max_tokens = int(os.getenv("MAX_OUTPUT_TOKENS", "8192"))
        self.chunk_height = int(os.getenv("IMAGE_CHUNK_HEIGHT", "2000"))
        
        # Add context tracking for image chunks
        self.contextual_enabled = os.getenv("CONTEXTUAL", "1") == "1"
        self.history_manager = history_manager
        self.chunk_context_manager = chunk_context_manager

        
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
    
    def load_progress(self):
        """Load progress tracking for image chunks"""
        if self.progress_manager:
            # Use the shared progress manager's data
            prog = self.progress_manager.prog.copy()
            # Ensure image_chunks key exists
            if "image_chunks" not in prog:
                prog["image_chunks"] = {}
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
        if self.progress_manager:
            # Update the shared progress manager's data
            self.progress_manager.prog["image_chunks"] = prog.get("image_chunks", {})
            # Save through the progress manager
            self.progress_manager.save()
        else:
            # Fallback to original behavior if no progress manager provided
            progress_file = os.path.join(self.output_dir, "translation_progress.json")
            try:
                # Write to a temporary file first
                temp_file = progress_file + '.tmp'
                with open(temp_file, 'w', encoding='utf-8') as f:
                    json.dump(prog, f, ensure_ascii=False, indent=2)
                
                # If successful, replace the original file
                if os.path.exists(progress_file):
                    os.remove(progress_file)
                os.rename(temp_file, progress_file)
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
            
            print(f"   🧹 Preprocessing image for watermark removal...")
            
            # Open image
            img = Image.open(image_path)
            
            # Convert to RGB if necessary
            if img.mode not in ('RGB', 'RGBA'):
                img = img.convert('RGB')
            
            # Check if advanced watermark removal is enabled
            if os.getenv("ADVANCED_WATERMARK_REMOVAL", "0") == "1":
                print(f"   🔬 Using advanced watermark removal...")
                
                # Convert to numpy array for advanced processing
                img_array = np.array(img)
                
                # Step 1: Detect if image has repeating patterns
                has_pattern, pattern_mask = self._detect_watermark_pattern(img_array)
                if has_pattern:
                    print(f"   🔍 Detected watermark pattern in image")
                    img_array = self._remove_periodic_watermark(img_array, pattern_mask)
                
                # Step 2: Apply adaptive histogram equalization
                img_array = self._adaptive_histogram_equalization(img_array)
                
                # Step 3: Edge-preserving denoising
                img_array = self._bilateral_filter(img_array)
                
                # Step 4: Text-specific enhancement
                img_array = self._enhance_text_regions(img_array)
                
                # Convert back to PIL Image
                img = Image.fromarray(img_array)
            
            # Apply basic PIL enhancements (always)
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.5)  # Increase contrast
            
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(1.1)  # Slight brightness increase
            
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
                    print(f"   📝 Created temp cleaned image")
                    return tmp.name  # Return temp path
            
        except Exception as e:
            logger.warning(f"Could not preprocess image: {e}")
            return image_path  # Return original on error
    
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
        
        try:
            self.current_image_path = image_path
            print(f"   🔍 translate_image called for: {image_path}")
            
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
            
            # Apply watermark preprocessing
            processed_path = self.preprocess_image_for_watermarks(image_path)
            
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
                
                # Process chunks if image is too tall
                if height > self.chunk_height:
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
            logger.error(f"Error translating image {image_path}: {e}")
            print(f"   ❌ Exception in translate_image: {e}")
            import traceback
            traceback.print_exc()
            return None
            
        finally:
            # Clean up temp file if it was created
            if processed_path and processed_path != image_path:
                # Only delete if we're not saving cleaned images
                if not os.getenv("SAVE_CLEANED_IMAGES", "1") == "1":
                    try:
                        if os.path.exists(processed_path):
                            os.unlink(processed_path)
                            print(f"   🧹 Cleaned up temp file")
                    except Exception as e:
                        logger.warning(f"Could not delete temp file: {e}")

    def _process_image_chunks(self, img, width, height, context, check_stop_fn):
        """Process a tall image by splitting it into chunks with contextual support"""
        num_chunks = (height + self.chunk_height - 1) // self.chunk_height
        overlap = 100  # Pixels of overlap between chunks
        
        print(f"   ✂️ Image too tall ({height}px), splitting into {num_chunks} chunks of {self.chunk_height}px...")
        
        # Clear context for new image
        self.image_chunk_context = []
        
        # Add retry info if enabled
        if os.getenv("RETRY_TIMEOUT", "1") == "1":
            timeout_seconds = int(os.getenv("CHUNK_TIMEOUT", "180"))
            print(f"   ⏱️ Auto-retry enabled: Will retry if chunks take > {timeout_seconds}s")
        
        print(f"   ⏳ This may take {num_chunks * 30}-{num_chunks * 60} seconds to complete")
        print(f"   ℹ️ Stop will take effect after current chunk completes")
        
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
        
        # Rest of the method remains the same...
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
            chunk_bytes = self._image_to_bytes(chunk)
            
            # Get custom image chunk prompt template from environment
            image_chunk_prompt_template = os.getenv("IMAGE_CHUNK_PROMPT", "This is part {chunk_idx} of {total_chunks} of a longer image. {context}")
            
            # Build context for this chunk
            chunk_context = image_chunk_prompt_template.format(
                chunk_idx=i+1,
                total_chunks=num_chunks,
                context=context
            )
            
            # Translate chunk WITH CONTEXT
            translation = self._call_vision_api(chunk_bytes, chunk_context, check_stop_fn)
            
            if translation:
                # Clean AI artifacts from chunk
                chunk_text = self._clean_translation_response(translation)
                all_translations.append(chunk_text)
                print(f"   🔍 DEBUG: Chunk {i+1} length: {len(chunk_text)} chars")
                if len(chunk_text) > 10000:  # Flag suspiciously large chunks
                    print(f"   ⚠️ WARNING: Chunk unusually large!")
                    print(f"   First 500 chars: {chunk_text[:500]}")
                    print(f"   Last 500 chars: {chunk_text[-500:]}")
                
                # Store context for next chunks
                if self.contextual_enabled:
                    self.image_chunk_context.append({
                        "user": chunk_context,
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
    
    def _process_single_image(self, img, context, check_stop_fn):
        """Process a single image that doesn't need chunking"""
        
        # Clear any previous context
        self.image_chunk_context = []
        
        print(f"   👍 Image height OK ({img.height}px), processing as single image...")
        
        # Check for stop before processing
        if check_stop_fn and check_stop_fn():
            print("   ❌ Image translation stopped by user")
            return None
        
        # Convert image to bytes
        image_bytes = self._image_to_bytes(img)
        
        # Call API
        translation = self._call_vision_api(image_bytes, context, check_stop_fn)
        
        if translation:
            return self._clean_translation_response(translation)
        else:
            print(f"   ❌ Translation returned empty result")
            return None

    def _image_to_bytes(self, img):
        """Convert PIL Image to bytes"""
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG', optimize=False)
        img_bytes.seek(0)
        return img_bytes.read()

    def _call_vision_api(self, image_data, context, check_stop_fn):
        """Make the actual API call for vision translation with retry support"""
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
                    messages.extend([
                        {"role": "user", "content": ctx["user"]},
                        {"role": "assistant", "content": ctx["assistant"]}
                    ])
        
        # Add current chunk (this already exists)
        messages.append({
            "role": "user", 
            "content": context if context else ""
        })
        
        # Rest of the method stays EXACTLY the same...
        retry_timeout_enabled = os.getenv("RETRY_TIMEOUT", "1") == "1"
        chunk_timeout = int(os.getenv("CHUNK_TIMEOUT", "180")) if retry_timeout_enabled else None
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
        """Clean AI artifacts from translation response"""
        if not response or not response.strip():
            return response
        
        # Remove common AI prefixes
        if response.startswith(('Sure', 'Here', "I'll translate", 'Certainly')):
            lines = response.split('\n')
            if len(lines) > 1:
                response = '\n'.join(lines[1:]).strip()
        
        return response

    def _save_translation_debug(self, image_path, translated_text):
        """Save translation to file for debugging"""
        trans_filename = f"translated_{os.path.basename(image_path)}.txt"
        trans_filepath = os.path.join(self.translated_images_dir, trans_filename)
        
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
