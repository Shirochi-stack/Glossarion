# ocr_manager.py
"""
OCR Manager for handling multiple OCR providers
Handles installation, model downloading, and OCR processing
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
            
            self._log("📥 Loading manga-ocr model from HuggingFace...")
            
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
        Process the ENTIRE image as a single region.
        Uses confidence from kwargs or defaults to settings value.
        """
        results = []
        
        try:
            if not self.is_loaded:
                if not self.load_model():
                    return results
            
            import cv2
            from PIL import Image
            
            # Get confidence from kwargs (passed from manga_translator.py)
            # This should be the actual confidence_threshold from manga_settings
            confidence = kwargs.get('confidence', 0.7)  # Default matches manga_settings_dialog.py
            
            # Convert numpy array to PIL
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            h, w = image.shape[:2]
            
            self._log("📝 Processing full image with manga-ocr...")
            
            # Run OCR on the entire image
            try:
                text = self._run_ocr(pil_image)
                
                if text and text.strip():
                    # Use the actual confidence value from settings
                    results.append(OCRResult(
                        text=text.strip(),
                        bbox=(0, 0, w, h),
                        confidence=confidence,  # Use the actual value
                        vertices=[(0, 0), (w, 0), (w, h), (0, h)]
                    ))
                    self._log(f"✅ Detected text with confidence {confidence:.2f}: {text[:50]}...")
                else:
                    self._log("⚠️ No text detected in image", "warning")
                    
            except Exception as e:
                self._log(f"❌ OCR failed: {str(e)}", "error")
            
        except Exception as e:
            self._log(f"❌ Error in manga-ocr: {str(e)}", "error")
            import traceback
            self._log(traceback.format_exc(), "error")
        
        return results
        
class PororoOCRProvider(OCRProvider):
    """Pororo OCR provider for Korean text"""
    
    def check_installation(self) -> bool:
        """Check if pororo is installed"""
        try:
            import pororo
            self.is_installed = True
            return True
        except ImportError:
            return False
    
    def install(self, progress_callback=None) -> bool:
        """Install pororo"""
        try:
            self._log("📦 Installing pororo...")
            if progress_callback:
                progress_callback("Installing pororo package...", 0)
            
            # Install dependencies first
            subprocess.check_call([
                sys.executable, "-m", "pip", "install",
                "torch", "torchvision", "--upgrade"
            ])
            
            # Install pororo
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "pororo", "--upgrade"
            ])
            
            if progress_callback:
                progress_callback("pororo installed successfully!", 100)
            
            self.is_installed = True
            self._log("✅ pororo installed successfully")
            return True
            
        except Exception as e:
            self._log(f"❌ Failed to install pororo: {str(e)}", "error")
            return False
    
    def load_model(self) -> bool:
        """Load pororo OCR model"""
        try:
            if not self.is_installed and not self.check_installation():
                self._log("❌ pororo not installed", "error")
                return False
            
            self._log("📥 Loading pororo OCR model...")
            from pororo import Pororo
            
            # Load OCR model for Korean
            self.model = Pororo(task="ocr", lang="ko")
            self.is_loaded = True
            
            self._log("✅ pororo OCR model loaded successfully")
            return True
            
        except Exception as e:
            self._log(f"❌ Failed to load pororo: {str(e)}", "error")
            return False
    
    def detect_text(self, image: np.ndarray, **kwargs) -> List[OCRResult]:
        """Detect Korean text using pororo"""
        results = []
        
        try:
            if not self.is_loaded:
                if not self.load_model():
                    return results
            
            # Pororo expects PIL Image or path
            if isinstance(image, np.ndarray):
                import cv2
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(image_rgb)
            else:
                pil_image = image
            
            # Run OCR
            ocr_results = self.model(pil_image, detail=True)
            
            # Parse results
            for item in ocr_results:
                if 'description' in item:
                    text = item['description']
                    bbox = item.get('bounding_poly', {}).get('vertices', [])
                    
                    if bbox:
                        xs = [v.get('x', 0) for v in bbox]
                        ys = [v.get('y', 0) for v in bbox]
                        x_min, x_max = min(xs), max(xs)
                        y_min, y_max = min(ys), max(ys)
                        
                        results.append(OCRResult(
                            text=text,
                            bbox=(x_min, y_min, x_max - x_min, y_max - y_min),
                            confidence=0.9,  # Pororo doesn't provide confidence
                            vertices=[(v.get('x', 0), v.get('y', 0)) for v in bbox]
                        ))
            
            self._log(f"✅ Detected {len(results)} text regions")
            
        except Exception as e:
            self._log(f"❌ Error in pororo detection: {str(e)}", "error")
        
        return results


class EasyOCRProvider(OCRProvider):
    """EasyOCR provider for multiple languages"""
    
    def __init__(self, log_callback=None, languages=['ja', 'ko', 'en']):
        super().__init__(log_callback)
        self.languages = languages
    
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
            
            self._log(f"📥 Loading easyocr model for languages: {self.languages}...")
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
            
            self._log("📥 Loading PaddleOCR model...")
            from paddleocr import PaddleOCR
            
            # Load model with auto-download
            self.model = PaddleOCR(
                use_angle_cls=True,
                lang='japan',  # Supports japan, korean, ch, en
                use_gpu=True,
                show_log=False
            )
            self.is_loaded = True
            
            self._log("✅ PaddleOCR model loaded successfully")
            return True
            
        except Exception as e:
            self._log(f"❌ Failed to load paddleocr: {str(e)}", "error")
            # Try CPU mode
            try:
                from paddleocr import PaddleOCR
                self.model = PaddleOCR(
                    use_angle_cls=True,
                    lang='japan',
                    use_gpu=False,
                    show_log=False
                )
                self.is_loaded = True
                self._log("✅ PaddleOCR loaded in CPU mode")
                return True
            except:
                return False
    
    def detect_text(self, image: np.ndarray, **kwargs) -> List[OCRResult]:
        """Detect text using paddleocr"""
        results = []
        
        try:
            if not self.is_loaded:
                if not self.load_model():
                    return results
            
            # Run OCR
            ocr_results = self.model.ocr(image)
            
            if ocr_results and ocr_results[0]:
                for line in ocr_results[0]:
                    bbox_points = line[0]
                    text = line[1][0]
                    confidence = line[1][1]
                    
                    # Convert points to bbox
                    xs = [p[0] for p in bbox_points]
                    ys = [p[1] for p in bbox_points]
                    x_min, x_max = min(xs), max(xs)
                    y_min, y_max = min(ys), max(ys)
                    
                    results.append(OCRResult(
                        text=text,
                        bbox=(int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)),
                        confidence=confidence,
                        vertices=[(int(p[0]), int(p[1])) for p in bbox_points]
                    ))
            
            self._log(f"✅ Detected {len(results)} text regions")
            
        except Exception as e:
            self._log(f"❌ Error in paddleocr detection: {str(e)}", "error")
        
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
            
            self._log("📥 Loading DocTR model...")
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
                            # Convert relative coordinates to absolute
                            x1, y1, x2, y2 = word.geometry
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
            
            self._log(f"✅ Detected {len(results)} text regions")
            
        except Exception as e:
            self._log(f"❌ Error in doctr detection: {str(e)}", "error")
        
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
