# ocr_manager.py
"""
OCR Manager for handling multiple OCR providers
Handles installation, model downloading, and OCR processing
Updated with HuggingFace donut model and proper bubble detection integration
"""

import os
import sys
import json
import subprocess
import threading
import traceback
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
from dataclasses import dataclass
from PIL import Image
import logging

try:
    import gptqmodel
    HAS_GPTQ = True
except ImportError:
    try:
        import auto_gptq
        HAS_GPTQ = True
    except ImportError:
        HAS_GPTQ = False

try:
    import optimum
    HAS_OPTIMUM = True
except ImportError:
    HAS_OPTIMUM = False

try:
    import accelerate
    HAS_ACCELERATE = True
except ImportError:
    HAS_ACCELERATE = False

logger = logging.getLogger(__name__)

@dataclass
class OCRResult:
    """Unified OCR result format"""
    text: str
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    confidence: float
    vertices: Optional[List[Tuple[int, int]]] = None
    
class OCRProvider:
    """Base class for OCR providers"""
    
    def __init__(self, log_callback=None):
        self.log_callback = log_callback
        self.is_installed = False
        self.is_loaded = False
        self.model = None
        
    def _log(self, message: str, level: str = "info"):
        """Log message"""
        if self.log_callback:
            self.log_callback(message, level)
        else:
            print(f"[{level.upper()}] {message}")
    
    def check_installation(self) -> bool:
        """Check if provider is installed"""
        raise NotImplementedError
    
    def install(self, progress_callback=None) -> bool:
        """Install the provider"""
        raise NotImplementedError
    
    def load_model(self) -> bool:
        """Load the OCR model"""
        raise NotImplementedError
    
    def detect_text(self, image: np.ndarray, **kwargs) -> List[OCRResult]:
        """Detect text in image"""
        raise NotImplementedError


class MangaOCRProvider(OCRProvider):
    """Manga OCR provider using HuggingFace model directly"""
    
    def __init__(self, log_callback=None):
        super().__init__(log_callback)
        self.processor = None
        self.model = None
        self.tokenizer = None
        
    def check_installation(self) -> bool:
        """Check if transformers is installed"""
        try:
            import transformers
            import torch
            self.is_installed = True
            return True
        except ImportError:
            return False 
    
    def install(self, progress_callback=None) -> bool:
        """Install transformers and torch"""
        pass
    
    def load_model(self) -> bool:
        """Load the manga-ocr model from HuggingFace"""
        try:
            if not self.is_installed and not self.check_installation():
                self._log("❌ Transformers not installed", "error")
                return False
            
            self._log("🔥 Loading manga-ocr model from HuggingFace...")
            
            from transformers import VisionEncoderDecoderModel, AutoTokenizer, AutoImageProcessor
            import torch
            
            # Load the model directly from HuggingFace
            model_name = "kha-white/manga-ocr-base"
            
            self._log(f"   Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            self._log(f"   Loading image processor...")
            self.processor = AutoImageProcessor.from_pretrained(model_name)
            
            self._log(f"   Loading model...")
            self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
            
            # Set to eval mode
            self.model.eval()
            
            # Move to GPU if available
            if torch.cuda.is_available():
                self.model = self.model.cuda()
                self._log("   ✅ Model loaded on GPU")
            else:
                self._log("   ✅ Model loaded on CPU")
            
            self.is_loaded = True
            self._log("✅ Manga OCR model ready")
            return True
            
        except Exception as e:
            self._log(f"❌ Failed to load manga-ocr model: {str(e)}", "error")
            import traceback
            self._log(traceback.format_exc(), "error")
            return False
    
    def _run_ocr(self, pil_image):
        """Run OCR on a PIL image using the HuggingFace model"""
        import torch
        
        # Process image
        pixel_values = self.processor(pil_image, return_tensors="pt").pixel_values
        
        # Move to same device as model
        if next(self.model.parameters()).is_cuda:
            pixel_values = pixel_values.cuda()
        
        # Generate text
        with torch.no_grad():
            generated_ids = self.model.generate(pixel_values)
        
        # Decode
        generated_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return generated_text
    
    def detect_text(self, image: np.ndarray, **kwargs) -> List[OCRResult]:
        """
        Process the image region passed to it.
        This could be a bubble region or the full image.
        """
        results = []
        
        try:
            if not self.is_loaded:
                if not self.load_model():
                    return results
            
            import cv2
            from PIL import Image
            
            # Get confidence from kwargs
            confidence = kwargs.get('confidence', 0.7)
            
            # Convert numpy array to PIL
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            h, w = image.shape[:2]
            
            self._log("🔍 Processing region with manga-ocr...")
            
            # Run OCR on the image region
            text = self._run_ocr(pil_image)
            
            if text and text.strip():
                # Return result for this region with its actual bbox
                results.append(OCRResult(
                    text=text.strip(),
                    bbox=(0, 0, w, h),  # Relative to the region passed in
                    confidence=confidence,
                    vertices=[(0, 0), (w, 0), (w, h), (0, h)]
                ))
                self._log(f"✅ Detected text: {text[:50]}...")
            
        except Exception as e:
            self._log(f"❌ Error in manga-ocr: {str(e)}", "error")
            
        return results

class Qwen2VL(OCRProvider):
    """OCR using Qwen2-VL - Vision Language Model that can read Korean text"""
    
    def __init__(self, log_callback=None):
        super().__init__(log_callback)
        self.processor = None
        self.model = None
        self.tokenizer = None
        
    def check_installation(self) -> bool:
        """Check if required packages are installed"""
        try:
            import transformers
            import torch
            self.is_installed = True
            return True
        except ImportError:
            return False
    
    def install(self, progress_callback=None) -> bool:
        """Install requirements for Qwen2-VL"""
        pass
    
    def load_model(self, model_size=None) -> bool:
        """Load Qwen2-VL model with size selection"""
        self._log(f"DEBUG: load_model called with model_size={model_size}")  # Add this
        try:
            if not self.is_installed and not self.check_installation():
                self._log("❌ Not installed", "error")
                return False
            
            self._log("🔥 Loading Qwen2-VL for Advanced OCR...")


            
            from transformers import AutoProcessor, AutoTokenizer
            import torch
            
            # Model options
            model_options = {
                "1": "Qwen/Qwen2-VL-2B-Instruct",
                "2": "Qwen/Qwen2-VL-7B-Instruct",
                "3": "Qwen/Qwen2-VL-72B-Instruct",
                "4": "custom"
            }
            
            # CHANGE: Default to 7B instead of 2B
            # Check for saved preference first
            if model_size is None:
                # Try to get from environment or config
                import os
                model_size = os.environ.get('QWEN2VL_MODEL_SIZE', '1')  # Default to '2' which is 7B
            
            # Determine which model to load
            if model_size and str(model_size).startswith("custom:"):
                # Custom model passed with ID
                model_id = str(model_size).replace("custom:", "")
                self.loaded_model_size = "Custom"
                self._log(f"Loading custom model: {model_id}")
            elif model_size == "4":
                # Custom option selected but no ID - shouldn't happen
                self._log("❌ Custom model selected but no ID provided", "error")
                return False
            elif model_size and str(model_size) in model_options:
                # Standard model option
                option = model_options[str(model_size)]
                if option == "custom":
                    self._log("❌ Custom model needs an ID", "error")
                    return False
                model_id = option
                # Set loaded_model_size for status display
                if model_size == "1":
                    self.loaded_model_size = "2B"
                elif model_size == "2":
                    self.loaded_model_size = "7B"
                elif model_size == "3":
                    self.loaded_model_size = "72B"
            else:
                # CHANGE: Default to 7B (option "2") instead of 2B
                model_id = model_options["1"]  # Changed from "1" to "2"
                self.loaded_model_size = "2B"   # Changed from "2B" to "7B"
                self._log("No model size specified, defaulting to 2B")  # Changed message
            
            self._log(f"Loading model: {model_id}")
            
            # Load processor and tokenizer
            self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            
            # Load the model - let it figure out the class dynamically
            if torch.cuda.is_available():
                self._log(f"GPU: {torch.cuda.get_device_name(0)}")
                # Use auto model class
                from transformers import AutoModelForVision2Seq
                self.model = AutoModelForVision2Seq.from_pretrained(
                    model_id,
                    dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True
                )
                self._log("✅ Model loaded on GPU")
            else:
                self._log("Loading on CPU...")
                from transformers import AutoModelForVision2Seq
                self.model = AutoModelForVision2Seq.from_pretrained(
                    model_id,
                    dtype=torch.float32,
                    trust_remote_code=True
                )
                self._log("✅ Model loaded on CPU")
            
            self.model.eval()
            self.is_loaded = True
            self._log("✅ Qwen2-VL ready for Advanced OCR!")
            return True
            
        except Exception as e:
            self._log(f"❌ Failed to load: {str(e)}", "error")
            import traceback
            self._log(traceback.format_exc(), "debug")
            return False       

    def detect_text(self, image: np.ndarray, **kwargs) -> List[OCRResult]:
        """Process image with Qwen2-VL for Korean text extraction"""
        results = []
        
        try:
            if not self.is_loaded:
                if not self.load_model():
                    return results
            
            import cv2
            from PIL import Image
            import torch
            
            # Convert to PIL
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            h, w = image.shape[:2]
            
            self._log(f"🔍 Processing with Qwen2-VL ({w}x{h} pixels)...")
            
            # FIXED: Clear, unambiguous prompt that only asks for text extraction
            # Don't ask for transcription, translation, or explanation
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": pil_image,
                        },
                        {
                            "type": "text", 
                            # CHANGED: More explicit prompt to prevent translation
                            "text": (
                                "Output ONLY the exact raw text you see in this image. "
                                "Do not translate. Do not explain. Do not add any commentary. "
                                "If you see Korean text, output the Korean text exactly as written. "
                                "If you see Japanese text, output the Japanese text exactly as written. "
                                "If you see Chinese text, output the Chinese text exactly as written. "
                                "If there is no text, output nothing."
                            )
                        }
                    ]
                }
            ]
            
            # Alternative simpler prompt if the above still causes issues:
            # "text": "OCR: Extract text as-is"
            
            # Process with Qwen2-VL
            text = self.processor.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )

            inputs = self.processor(
                text=[text],
                images=[pil_image],
                padding=True,
                return_tensors="pt"
            )

            # Get the device the model is currently on
            model_device = next(self.model.parameters()).device

            # Move inputs to the same device as the model
            inputs = inputs.to(model_device)

            # Generate text with stricter parameters to avoid creative responses
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=128,      # Reduced from 512 - manga bubbles are typically short
                    do_sample=False,        # Keep deterministic
                    temperature=0.01,       # Keep your very low temperature
                    top_p=1.0,             # Keep no nucleus sampling
                    repetition_penalty=1.0, # Keep no repetition penalty
                    num_beams=1,           # Ensure greedy decoding (faster than beam search)
                    use_cache=True,        # Enable KV cache for speed
                    early_stopping=True,   # Stop at EOS token
                    pad_token_id=self.tokenizer.pad_token_id,      # Proper padding
                    eos_token_id=self.tokenizer.eos_token_id,      # Proper stopping
                )
            
            # Decode the output
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )[0]
            
            if output_text and output_text.strip():
                text = output_text.strip()
                
                # ADDED: Filter out any response that looks like an explanation or apology
                # Common patterns that indicate the model is being "helpful" instead of just extracting
                unwanted_patterns = [
                    "죄송합니다",  # "I apologize"
                    "sorry",
                    "apologize",
                    "이미지에는",  # "in this image"
                    "텍스트가 없습니다",  # "there is no text"
                    "I cannot",
                    "I don't see",
                    "There is no",
                    "질문이 있으시면",  # "if you have questions"
                ]
                
                # Check if response contains unwanted patterns
                text_lower = text.lower()
                is_explanation = any(pattern.lower() in text_lower for pattern in unwanted_patterns)
                
                # Also check if the response is suspiciously long for a bubble
                # Most manga bubbles are short, if we get 50+ chars it might be an explanation
                is_too_long = len(text) > 100 and ('.' in text or ',' in text or '!' in text)
                
                if is_explanation or is_too_long:
                    self._log(f"⚠️ Model returned explanation instead of text, ignoring", "warning")
                    # Return empty result or just skip this region
                    return results
                
                # Check language
                has_korean = any('\uAC00' <= c <= '\uD7AF' for c in text)
                has_japanese = any('\u3040' <= c <= '\u309F' or '\u30A0' <= c <= '\u30FF' for c in text)
                has_chinese = any('\u4E00' <= c <= '\u9FFF' for c in text)
                
                if has_korean:
                    self._log(f"✅ Korean detected: {text[:50]}...")
                elif has_japanese:
                    self._log(f"✅ Japanese detected: {text[:50]}...")
                elif has_chinese:
                    self._log(f"✅ Chinese detected: {text[:50]}...")
                else:
                    self._log(f"✅ Text: {text[:50]}...")
                
                results.append(OCRResult(
                    text=text,
                    bbox=(0, 0, w, h),
                    confidence=0.9,
                    vertices=[(0, 0), (w, 0), (w, h), (0, h)]
                ))
            else:
                self._log("⚠️ No text detected", "warning")
        
        except Exception as e:
            self._log(f"❌ Error: {str(e)}", "error")
            import traceback
            self._log(traceback.format_exc(), "debug")
        
        return results
    
class EasyOCRProvider(OCRProvider):
    """EasyOCR provider for multiple languages"""
    
    def __init__(self, log_callback=None, languages=None):
        super().__init__(log_callback)
        # Default to safe language combination
        self.languages = languages or ['ja', 'en']  # Safe default
        self._validate_language_combination()

    def _validate_language_combination(self):
        """Validate and fix EasyOCR language combinations"""
        # EasyOCR language compatibility rules
        incompatible_pairs = [
            (['ja', 'ko'], 'Japanese and Korean cannot be used together'),
            (['ja', 'zh'], 'Japanese and Chinese cannot be used together'),
            (['ko', 'zh'], 'Korean and Chinese cannot be used together')
        ]
        
        for incompatible, reason in incompatible_pairs:
            if all(lang in self.languages for lang in incompatible):
                self._log(f"⚠️ EasyOCR: {reason}", "warning")
                # Keep first language + English
                self.languages = [self.languages[0], 'en']
                self._log(f"🔧 Auto-adjusted to: {self.languages}", "info")
                break
    
    def check_installation(self) -> bool:
        """Check if easyocr is installed"""
        try:
            import easyocr
            self.is_installed = True
            return True
        except ImportError:
            return False
    
    def install(self, progress_callback=None) -> bool:
        """Install easyocr"""
        pass
    
    def load_model(self) -> bool:
        """Load easyocr model"""
        try:
            if not self.is_installed and not self.check_installation():
                self._log("❌ easyocr not installed", "error")
                return False
            
            self._log(f"🔥 Loading easyocr model for languages: {self.languages}...")
            import easyocr
            
            # This will download models on first run
            self.model = easyocr.Reader(self.languages, gpu=True)
            self.is_loaded = True
            
            self._log("✅ easyocr model loaded successfully")
            return True
            
        except Exception as e:
            self._log(f"❌ Failed to load easyocr: {str(e)}", "error")
            # Try CPU mode if GPU fails
            try:
                import easyocr
                self.model = easyocr.Reader(self.languages, gpu=False)
                self.is_loaded = True
                self._log("✅ easyocr loaded in CPU mode")
                return True
            except:
                return False
    
    def detect_text(self, image: np.ndarray, **kwargs) -> List[OCRResult]:
        """Detect text using easyocr"""
        results = []
        
        try:
            if not self.is_loaded:
                if not self.load_model():
                    return results
            
            # EasyOCR can work directly with numpy arrays
            ocr_results = self.model.readtext(image, detail=1)
            
            # Parse results
            for (bbox, text, confidence) in ocr_results:
                # bbox is a list of 4 points
                xs = [point[0] for point in bbox]
                ys = [point[1] for point in bbox]
                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)
                
                results.append(OCRResult(
                    text=text,
                    bbox=(int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)),
                    confidence=confidence,
                    vertices=[(int(p[0]), int(p[1])) for p in bbox]
                ))
            
            self._log(f"✅ Detected {len(results)} text regions")
            
        except Exception as e:
            self._log(f"❌ Error in easyocr detection: {str(e)}", "error")
        
        return results


class PaddleOCRProvider(OCRProvider):
    """PaddleOCR provider"""
    
    def check_installation(self) -> bool:
        """Check if paddleocr is installed"""
        try:
            from paddleocr import PaddleOCR
            self.is_installed = True
            return True
        except ImportError:
            return False
    
    def install(self, progress_callback=None) -> bool:
        """Install paddleocr"""
        pass
    
    def load_model(self) -> bool:
        """Load paddleocr model"""
        try:
            if not self.is_installed and not self.check_installation():
                self._log("❌ paddleocr not installed", "error")
                return False
            
            self._log("🔥 Loading PaddleOCR model...")
            from paddleocr import PaddleOCR
            
            # Updated API - minimal parameters only
            self.model = PaddleOCR(
                use_angle_cls=True,
                lang='ch'  # Supports japan, korean, ch, en
                # Removed use_gpu and show_log - not supported anymore
            )
            self.is_loaded = True
            
            self._log("✅ PaddleOCR model loaded successfully")
            return True
            
        except Exception as e:
            self._log(f"❌ Failed to load paddleocr: {str(e)}", "error")
            
            # Try with even fewer parameters
            try:
                from paddleocr import PaddleOCR
                self.model = PaddleOCR(lang='japan')  # Just language
                self.is_loaded = True
                self._log("✅ PaddleOCR loaded with minimal config")
                return True
            except:
                # Final fallback - absolute minimal
                try:
                    from paddleocr import PaddleOCR
                    self.model = PaddleOCR()  # Default everything
                    self.is_loaded = True
                    self._log("✅ PaddleOCR loaded with defaults")
                    return True
                except Exception as e:
                    self._log(f"❌ PaddleOCR completely failed: {str(e)}", "error")
                    return False
    
    def detect_text(self, image: np.ndarray, **kwargs) -> List[OCRResult]:
        """Detect text using paddleocr with better error handling"""
        results = []
        
        try:
            if not self.is_loaded:
                if not self.load_model():
                    return results
            
            # Ensure image is in the correct format for PaddleOCR
            import cv2
            if len(image.shape) == 2:  # Grayscale
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            
            # Run OCR
            try:
                ocr_results = self.model.ocr(image)
            except Exception as ocr_error:
                self._log(f"PaddleOCR failed on image: {ocr_error}", "error")
                return results
            
            # Debug: Check what PaddleOCR actually returned
            self._log(f"PaddleOCR returned type: {type(ocr_results)}", "debug")
            
            # Handle various return types from PaddleOCR
            if ocr_results is None:
                self._log("PaddleOCR returned None", "debug")
                return results
            
            if isinstance(ocr_results, int):
                # Sometimes returns status code
                if ocr_results == 0:
                    self._log("PaddleOCR returned status 0 (no text found)", "debug")
                else:
                    self._log(f"PaddleOCR returned status code: {ocr_results}", "debug")
                return results
            
            if not isinstance(ocr_results, list):
                self._log(f"Unexpected PaddleOCR result type: {type(ocr_results)}", "warning")
                return results
            
            # Handle empty list
            if len(ocr_results) == 0:
                self._log("No text detected by PaddleOCR", "debug")
                return results
            
            # Check first element to determine format
            first_element = ocr_results[0] if ocr_results else None
            
            # If first element is None or empty, skip
            if first_element is None or (isinstance(first_element, list) and len(first_element) == 0):
                self._log("PaddleOCR returned empty results", "debug")
                return results
            
            # Process results based on format
            if isinstance(first_element, list):
                # Standard format: list of detection results
                for item in ocr_results:
                    if not item:  # Skip None or empty
                        continue
                        
                    # Each item should be a list of detections for this image
                    if isinstance(item, list):
                        for detection in item:
                            if not detection or len(detection) < 2:
                                continue
                            
                            try:
                                # detection[0] = bbox points
                                # detection[1] = (text, confidence)
                                bbox_points = detection[0]
                                text_data = detection[1]
                                
                                if not isinstance(text_data, tuple) or len(text_data) < 2:
                                    continue
                                
                                text = str(text_data[0])  # Ensure it's a string
                                confidence = float(text_data[1])  # Ensure it's a float
                                
                                if not text.strip():
                                    continue
                                
                                # Calculate bounding box
                                xs = [float(p[0]) for p in bbox_points]
                                ys = [float(p[1]) for p in bbox_points]
                                x_min, x_max = min(xs), max(xs)
                                y_min, y_max = min(ys), max(ys)
                                
                                results.append(OCRResult(
                                    text=text,
                                    bbox=(int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)),
                                    confidence=confidence,
                                    vertices=[(int(p[0]), int(p[1])) for p in bbox_points]
                                ))
                                
                            except Exception as e:
                                self._log(f"Failed to parse detection: {e}", "debug")
                                continue
            else:
                self._log(f"Unknown PaddleOCR result format: {type(first_element)}", "warning")
            
            if results:
                self._log(f"✅ Detected {len(results)} text regions", "info")
            else:
                self._log("No valid text regions found", "debug")
            
        except Exception as e:
            # Better error message
            error_msg = str(e) if str(e) else type(e).__name__
            self._log(f"❌ Error in paddleocr detection: {error_msg}", "error")
            import traceback
            self._log(traceback.format_exc(), "debug")
        
        return results


class DocTROCRProvider(OCRProvider):
    """DocTR OCR provider"""
    
    def check_installation(self) -> bool:
        """Check if doctr is installed"""
        try:
            from doctr.models import ocr_predictor
            self.is_installed = True
            return True
        except ImportError:
            return False
    
    def install(self, progress_callback=None) -> bool:
        """Install doctr"""
        pass
    
    def load_model(self) -> bool:
        """Load doctr model"""
        try:
            if not self.is_installed and not self.check_installation():
                self._log("❌ doctr not installed", "error")
                return False
            
            self._log("🔥 Loading DocTR model...")
            from doctr.models import ocr_predictor
            
            # Load pretrained model
            self.model = ocr_predictor(pretrained=True)
            self.is_loaded = True
            
            self._log("✅ DocTR model loaded successfully")
            return True
            
        except Exception as e:
            self._log(f"❌ Failed to load doctr: {str(e)}", "error")
            return False
    
    def detect_text(self, image: np.ndarray, **kwargs) -> List[OCRResult]:
        """Detect text using doctr"""
        results = []
        
        try:
            if not self.is_loaded:
                if not self.load_model():
                    return results
            
            from doctr.io import DocumentFile
            
            # DocTR expects document format
            # Convert numpy array to PIL and save temporarily
            import tempfile
            import cv2
            
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                cv2.imwrite(tmp.name, image)
                doc = DocumentFile.from_images(tmp.name)
            
            # Run OCR
            result = self.model(doc)
            
            # Parse results
            h, w = image.shape[:2]
            for page in result.pages:
                for block in page.blocks:
                    for line in block.lines:
                        for word in line.words:
                            # Handle different geometry formats
                            geometry = word.geometry
                            
                            if len(geometry) == 4:
                                # Standard format: (x1, y1, x2, y2)
                                x1, y1, x2, y2 = geometry
                            elif len(geometry) == 2:
                                # Alternative format: ((x1, y1), (x2, y2))
                                (x1, y1), (x2, y2) = geometry
                            else:
                                self._log(f"Unexpected geometry format: {geometry}", "warning")
                                continue
                            
                            # Convert relative coordinates to absolute
                            x1, x2 = int(x1 * w), int(x2 * w)
                            y1, y2 = int(y1 * h), int(y2 * h)
                            
                            results.append(OCRResult(
                                text=word.value,
                                bbox=(x1, y1, x2 - x1, y2 - y1),
                                confidence=word.confidence,
                                vertices=[(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
                            ))
            
            # Clean up temp file
            try:
                os.unlink(tmp.name)
            except:
                pass
            
            self._log(f"DocTR detected {len(results)} text regions")
            
        except Exception as e:
            self._log(f"Error in doctr detection: {str(e)}", "error")
            import traceback
            self._log(traceback.format_exc(), "error")
        
        return results


class OCRManager:
    """Manager for multiple OCR providers"""
    
    def __init__(self, log_callback=None):
        self.log_callback = log_callback
        self.providers = {
            'manga-ocr': MangaOCRProvider(log_callback),
            'easyocr': EasyOCRProvider(log_callback),
            'paddleocr': PaddleOCRProvider(log_callback),
            'doctr': DocTROCRProvider(log_callback),
            'Qwen2-VL': Qwen2VL(log_callback)
        }
        self.current_provider = None
        
    def get_provider(self, name: str) -> Optional[OCRProvider]:
        """Get OCR provider by name"""
        return self.providers.get(name)
    
    def set_current_provider(self, name: str):
        """Set current active provider"""
        if name in self.providers:
            self.current_provider = name
            return True
        return False
    
    def check_provider_status(self, name: str) -> Dict[str, bool]:
        """Check installation and loading status of provider"""
        provider = self.providers.get(name)
        if not provider:
            return {'installed': False, 'loaded': False}
        
        return {
            'installed': provider.check_installation(),
            'loaded': provider.is_loaded
        }
    
    def install_provider(self, name: str, progress_callback=None) -> bool:
        """Install a provider"""
        provider = self.providers.get(name)
        if not provider:
            return False
        
        return provider.install(progress_callback)
    
    def load_provider(self, name: str, **kwargs) -> bool:
        """Load a provider's model with optional parameters"""
        provider = self.providers.get(name)
        if not provider:
            return False
        
        return provider.load_model(**kwargs)  # <-- Passes model_size and any other kwargs
    
    def detect_text(self, image: np.ndarray, provider_name: str = None, **kwargs) -> List[OCRResult]:
        """Detect text using specified or current provider"""
        provider_name = provider_name or self.current_provider
        if not provider_name:
            return []
        
        provider = self.providers.get(provider_name)
        if not provider:
            return []
        
        return provider.detect_text(image, **kwargs)
