# ocr_manager.py
"""
OCR Manager for handling multiple OCR providers
Handles installation, model downloading, and OCR processing
Updated with HuggingFace Pororo model and proper bubble detection integration
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
        try:
            self._log("📦 Installing transformers for manga OCR...")
            if progress_callback:
                progress_callback("Installing transformers...", 0)
            
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "transformers", "torch", "pillow", "--upgrade"
            ])
            
            if progress_callback:
                progress_callback("Transformers installed successfully!", 100)
            
            self.is_installed = True
            self._log("✅ Transformers installed successfully")
            return True
            
        except Exception as e:
            self._log(f"❌ Failed to install transformers: {str(e)}", "error")
            return False
    
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

class PororoOCRProvider(OCRProvider):
    """Pororo OCR provider using HuggingFace model"""
    
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
            from huggingface_hub import hf_hub_download
            self.is_installed = True
            return True
        except ImportError:
            return False
    
    def install(self, progress_callback=None) -> bool:
        """Install required packages for HuggingFace Pororo"""
        try:
            self._log("📦 Installing packages for Pororo OCR...")
            if progress_callback:
                progress_callback("Installing required packages...", 0)
            
            # Install dependencies
            subprocess.check_call([
                sys.executable, "-m", "pip", "install",
                "transformers", "torch", "torchvision", 
                "huggingface-hub", "pillow", "--upgrade"
            ])
            
            if progress_callback:
                progress_callback("Packages installed successfully!", 100)
            
            self.is_installed = True
            self._log("✅ Packages installed successfully")
            return True
            
        except Exception as e:
            self._log(f"❌ Failed to install packages: {str(e)}", "error")
            return False
    
    def load_model(self) -> bool:
        """Load Pororo OCR model from HuggingFace"""
        try:
            if not self.is_installed and not self.check_installation():
                self._log("❌ Required packages not installed", "error")
                return False
            
            self._log("🔥 Loading Pororo OCR model from HuggingFace...")
            
            from transformers import (
                TrOCRProcessor, 
                VisionEncoderDecoderModel,
                AutoTokenizer,
                AutoImageProcessor
            )
            from huggingface_hub import hf_hub_download
            import torch
            
            # The HuggingFace model ID
            model_id = "ogkalu/pororo"
            
            try:
                # Try to load as a standard transformers model first
                self._log(f"   Loading from HuggingFace: {model_id}")
                
                # Download model files
                self._log("   Downloading model files...")
                
                # List files in the repository
                from huggingface_hub import list_repo_files
                files = list_repo_files(model_id)
                self._log(f"   Available files: {files[:5]}...")  # Show first 5 files
                
                # Check what type of model this is
                if any('pytorch_model' in f for f in files) or any('model.safetensors' in f for f in files):
                    # Standard transformers model
                    self._log("   Detected transformers model format")
                    
                    # Try loading as VisionEncoderDecoder model (similar to manga-ocr)
                    try:
                        self._log("   Loading as VisionEncoderDecoder model...")
                        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
                        self.processor = AutoImageProcessor.from_pretrained(model_id)
                        self.model = VisionEncoderDecoderModel.from_pretrained(model_id)
                    except Exception as e:
                        self._log(f"   Failed as VisionEncoderDecoder: {e}", "warning")
                        
                        # Try TrOCR format
                        self._log("   Trying TrOCR format...")
                        self.processor = TrOCRProcessor.from_pretrained(model_id)
                        self.model = VisionEncoderDecoderModel.from_pretrained(model_id)
                        self.tokenizer = self.processor.tokenizer
                        
                elif any('.pt' in f or '.pth' in f for f in files):
                    # PyTorch checkpoint format
                    self._log("   Detected PyTorch checkpoint format")
                    
                    # Download the model file
                    model_file = [f for f in files if '.pt' in f or '.pth' in f][0]
                    local_path = hf_hub_download(repo_id=model_id, filename=model_file)
                    self._log(f"   Downloaded model to: {local_path}")
                    
                    # Load as a generic torch model
                    # Note: This requires knowing the architecture
                    # For now, we'll try to load it as a VisionEncoderDecoder
                    from transformers import VisionEncoderDecoderConfig
                    
                    # Try to load config
                    try:
                        config = VisionEncoderDecoderConfig.from_pretrained(model_id)
                        self.model = VisionEncoderDecoderModel(config)
                        state_dict = torch.load(local_path, map_location='cpu')
                        self.model.load_state_dict(state_dict)
                    except:
                        # Fallback: use a default Korean OCR model
                        self._log("   Using fallback Korean OCR model...", "warning")
                        fallback_model = "microsoft/trocr-base-handwritten"
                        self.processor = TrOCRProcessor.from_pretrained(fallback_model)
                        self.model = VisionEncoderDecoderModel.from_pretrained(fallback_model)
                        self.tokenizer = self.processor.tokenizer
                else:
                    raise ValueError(f"Unknown model format in {model_id}")
                
                # Set model to eval mode
                self.model.eval()
                
                # Move to GPU if available
                if torch.cuda.is_available():
                    self.model = self.model.cuda()
                    self._log("   ✅ Model loaded on GPU")
                else:
                    self._log("   ✅ Model loaded on CPU")
                
                self.is_loaded = True
                self._log("✅ Pororo OCR model loaded successfully")
                return True
                
            except Exception as e:
                self._log(f"⚠️ Failed to load from HuggingFace: {e}", "warning")
                self._log("   Falling back to default Korean OCR model...", "warning")
                
                # Fallback to a working Korean OCR model
                fallback_model = "microsoft/trocr-base-handwritten"
                self.processor = TrOCRProcessor.from_pretrained(fallback_model)
                self.model = VisionEncoderDecoderModel.from_pretrained(fallback_model)
                self.tokenizer = self.processor.tokenizer
                
                self.model.eval()
                if torch.cuda.is_available():
                    self.model = self.model.cuda()
                
                self.is_loaded = True
                self._log("✅ Loaded fallback OCR model")
                return True
                
        except Exception as e:
            self._log(f"❌ Failed to load Pororo model: {str(e)}", "error")
            import traceback
            self._log(traceback.format_exc(), "error")
            return False
        
    def _run_ocr(self, pil_image):
        """Run OCR on a PIL image using the loaded model"""
        import torch
        
        try:
            if hasattr(self, 'processor') and self.processor:
                # Use processor if available (TrOCR style)
                pixel_values = self.processor(pil_image, return_tensors="pt").pixel_values
            else:
                # Manual preprocessing
                from torchvision import transforms
                transform = transforms.Compose([
                    transforms.Resize((384, 384)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                pixel_values = transform(pil_image).unsqueeze(0)
            
            # Move to same device as model
            if next(self.model.parameters()).is_cuda:
                pixel_values = pixel_values.cuda()
            
            # Generate text
            with torch.no_grad():
                generated_ids = self.model.generate(pixel_values, max_length=512)
            
            # Decode
            if self.tokenizer:
                generated_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            elif hasattr(self.processor, 'batch_decode'):
                generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            else:
                # Fallback to basic decoding
                generated_text = ""
                self._log("⚠️ No tokenizer available for decoding", "warning")
            
            # IMPORTANT: Clean and normalize the text to prevent encoding issues
            if generated_text:
                # Remove any control characters and normalize unicode
                import unicodedata
                import re
                
                # Normalize unicode (NFC form for Korean)
                generated_text = unicodedata.normalize('NFC', generated_text)
                
                # Remove control characters except normal whitespace
                generated_text = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]', '', generated_text)
                
                # Clean up excessive whitespace
                generated_text = ' '.join(generated_text.split())
                
                # Ensure it's properly encoded as UTF-8
                generated_text = generated_text.encode('utf-8', errors='ignore').decode('utf-8')
            
            return generated_text
            
        except Exception as e:
            self._log(f"❌ OCR inference failed: {str(e)}", "error")
            return ""
    
    def detect_text(self, image: np.ndarray, **kwargs) -> List[OCRResult]:
        """
        Process the image region with proper encoding handling.
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
            
            self._log("🔍 Processing region with Pororo OCR...")
            
            # Run OCR on the image
            try:
                text = self._run_ocr(pil_image)
                
                if text and text.strip():
                    # Ensure text is clean before returning
                    clean_text = text.strip()
                    
                    # Additional validation for Korean text
                    if clean_text:
                        # Check if text contains valid Korean characters
                        has_korean = any('\uAC00' <= c <= '\uD7AF' for c in clean_text)
                        if has_korean:
                            self._log(f"✅ Detected Korean text: {clean_text[:30]}...")
                        
                        results.append(OCRResult(
                            text=clean_text,
                            bbox=(0, 0, w, h),
                            confidence=confidence,
                            vertices=[(0, 0), (w, 0), (w, h), (0, h)]
                        ))
                else:
                    self._log("⚠️ No text detected in region", "warning")
                    
            except Exception as e:
                self._log(f"❌ OCR failed: {str(e)}", "error")
                import traceback
                self._log(traceback.format_exc(), "error")
            
        except Exception as e:
            self._log(f"❌ Error in Pororo OCR: {str(e)}", "error")
            import traceback
            self._log(traceback.format_exc(), "error")
        
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
        try:
            self._log("📦 Installing easyocr...")
            if progress_callback:
                progress_callback("Installing easyocr package...", 0)
            
            # Install easyocr
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "easyocr", "--upgrade"
            ])
            
            if progress_callback:
                progress_callback("easyocr installed successfully!", 100)
            
            self.is_installed = True
            self._log("✅ easyocr installed successfully")
            return True
            
        except Exception as e:
            self._log(f"❌ Failed to install easyocr: {str(e)}", "error")
            return False
    
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
        try:
            self._log("📦 Installing paddlepaddle and paddleocr...")
            if progress_callback:
                progress_callback("Installing PaddleOCR...", 0)
            
            # Install paddlepaddle first
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "paddlepaddle", "--upgrade"
            ])
            
            # Install paddleocr
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "paddleocr", "--upgrade"
            ])
            
            if progress_callback:
                progress_callback("PaddleOCR installed successfully!", 100)
            
            self.is_installed = True
            self._log("✅ paddleocr installed successfully")
            return True
            
        except Exception as e:
            self._log(f"❌ Failed to install paddleocr: {str(e)}", "error")
            return False
    
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
        try:
            self._log("📦 Installing doctr...")
            if progress_callback:
                progress_callback("Installing DocTR...", 0)
            
            # Install doctr
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "python-doctr[torch]", "--upgrade"
            ])
            
            if progress_callback:
                progress_callback("DocTR installed successfully!", 100)
            
            self.is_installed = True
            self._log("✅ doctr installed successfully")
            return True
            
        except Exception as e:
            self._log(f"❌ Failed to install doctr: {str(e)}", "error")
            return False
    
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
            'pororo': PororoOCRProvider(log_callback),
            'easyocr': EasyOCRProvider(log_callback),
            'paddleocr': PaddleOCRProvider(log_callback),
            'doctr': DocTROCRProvider(log_callback)
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
    
    def load_provider(self, name: str) -> bool:
        """Load a provider's model"""
        provider = self.providers.get(name)
        if not provider:
            return False
        
        return provider.load_model()
    
    def detect_text(self, image: np.ndarray, provider_name: str = None, **kwargs) -> List[OCRResult]:
        """Detect text using specified or current provider"""
        provider_name = provider_name or self.current_provider
        if not provider_name:
            return []
        
        provider = self.providers.get(provider_name)
        if not provider:
            return []
        
        return provider.detect_text(image, **kwargs)
