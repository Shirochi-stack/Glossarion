# ocr_manager.py
"""
OCR Manager for handling multiple OCR providers
Handles installation, model downloading, and OCR processing
Updated with HuggingFace donut model and proper bubble detection integration
"""
import os
import sys
import cv2
import json
import subprocess
import threading
import traceback
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
from dataclasses import dataclass
from PIL import Image
import logging
import time
import random
import base64
import io
import requests

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
    """Unified OCR result format with built-in sanitization to prevent data corruption."""
    text: str
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    confidence: float
    vertices: Optional[List[Tuple[int, int]]] = None

    def __post_init__(self):
        """
        This special method is called automatically after the object is created.
        It acts as a final safeguard to ensure the 'text' attribute is ALWAYS a clean string.
        """
        # --- THIS IS THE DEFINITIVE FIX ---
        # If the text we received is a tuple, we extract the first element.
        # This makes it impossible for a tuple to exist in a finished object.
        if isinstance(self.text, tuple):
            # Log that we are fixing a critical data error.
            print(f"CRITICAL WARNING: Corrupted tuple detected in OCRResult. Sanitizing '{self.text}' to '{self.text[0]}'.")
            self.text = self.text[0]
        
        # Ensure the final result is always a stripped string.
        self.text = str(self.text).strip()
    
class OCRProvider:
    """Base class for OCR providers"""
    
    def __init__(self, log_callback=None):
        self.log_callback = log_callback
        self.is_installed = False
        self.is_loaded = False
        self.model = None
        self.stop_flag = None
        self._stopped = False
        
    def _log(self, message: str, level: str = "info"):
        """Log message with stop suppression"""
        # Suppress logs when stopped (allow only essential stop confirmation messages)
        if self._check_stop():
            essential_stop_keywords = [
                "⏹️ Translation stopped by user",
                "⏹️ OCR processing stopped",
                "cleanup", "🧹"
            ]
            if not any(keyword in message for keyword in essential_stop_keywords):
                return
        
        if self.log_callback:
            self.log_callback(message, level)
        else:
            print(f"[{level.upper()}] {message}")
    
    def set_stop_flag(self, stop_flag):
        """Set the stop flag for checking interruptions"""
        self.stop_flag = stop_flag
        self._stopped = False
    
    def _check_stop(self) -> bool:
        """Check if stop has been requested"""
        if self._stopped:
            return True
        if self.stop_flag and self.stop_flag.is_set():
            self._stopped = True
            return True
        # Check global manga translator cancellation
        try:
            from manga_translator import MangaTranslator
            if MangaTranslator.is_globally_cancelled():
                self._stopped = True
                return True
        except Exception:
            pass
        return False
    
    def reset_stop_flags(self):
        """Reset stop flags when starting new processing"""
        self._stopped = False
    
    def check_installation(self) -> bool:
        """Check if provider is installed"""
        raise NotImplementedError
    
    def install(self, progress_callback=None) -> bool:
        """Install the provider"""
        raise NotImplementedError
    
    def load_model(self, **kwargs) -> bool:
        """Load the OCR model"""
        raise NotImplementedError
    
    def detect_text(self, image: np.ndarray, **kwargs) -> List[OCRResult]:
        """Detect text in image"""
        raise NotImplementedError

class CustomAPIProvider(OCRProvider):
    """Custom API OCR provider that uses existing GUI variables"""
    
    def __init__(self, log_callback=None):
        super().__init__(log_callback)
        
        # Use EXISTING environment variables from TranslatorGUI
        self.api_url = os.environ.get('OPENAI_CUSTOM_BASE_URL', '')
        self.api_key = os.environ.get('API_KEY', '') or os.environ.get('OPENAI_API_KEY', '')
        self.model_name = os.environ.get('MODEL', 'gpt-4o-mini')
        
        # OCR prompt - use system prompt or a dedicated OCR prompt variable
        self.ocr_prompt = os.environ.get('OCR_SYSTEM_PROMPT', 
            os.environ.get('SYSTEM_PROMPT', 
            "YOU ARE AN OCR SYSTEM. YOUR ONLY JOB IS TEXT EXTRACTION.\n\n"
            "CRITICAL RULES:\n"
            "1. DO NOT TRANSLATE ANYTHING\n"
            "2. DO NOT MODIFY THE TEXT\n"
            "3. DO NOT EXPLAIN OR COMMENT\n"
            "4. ONLY OUTPUT THE EXACT TEXT YOU SEE\n"
            "5. PRESERVE NATURAL TEXT FLOW - DO NOT ADD UNNECESSARY LINE BREAKS\n\n"
            "If you see Korean text, output it in Korean.\n"
            "If you see Japanese text, output it in Japanese.\n"
            "If you see Chinese text, output it in Chinese.\n"
            "If you see English text, output it in English.\n\n"
            "IMPORTANT: Only use line breaks where they naturally occur in the original text "
            "(e.g., between dialogue lines or paragraphs). Do not break text mid-sentence or "
            "between every word/character.\n\n"
            "For vertical text common in manga/comics, transcribe it as a continuous line unless "
            "there are clear visual breaks.\n\n"
            "NEVER translate. ONLY extract exactly what is written.\n"
            "Output ONLY the raw text, nothing else."
            ))
        
        # Use existing temperature and token settings  
        self.temperature = float(os.environ.get('TRANSLATION_TEMPERATURE', '0.01'))
        # Don't hardcode to 8192 - get fresh value when actually used
        self.max_tokens = int(os.environ.get('MAX_OUTPUT_TOKENS', '4096'))
        
        # Image settings from existing compression variables
        self.image_format = 'jpeg' if os.environ.get('IMAGE_COMPRESSION_FORMAT', 'auto') != 'png' else 'png'
        self.image_quality = int(os.environ.get('JPEG_QUALITY', '100'))
        
        # Simple defaults
        self.api_format = 'openai'  # Most custom endpoints are OpenAI-compatible
        self.timeout = int(os.environ.get('CHUNK_TIMEOUT', '30'))
        self.api_headers = {}  # Additional custom headers
        
        # Retry configuration for Custom API OCR calls
        self.max_retries = int(os.environ.get('CUSTOM_OCR_MAX_RETRIES', '3'))
        self.retry_initial_delay = float(os.environ.get('CUSTOM_OCR_RETRY_INITIAL_DELAY', '0.8'))
        self.retry_backoff = float(os.environ.get('CUSTOM_OCR_RETRY_BACKOFF', '1.8'))
        self.retry_jitter = float(os.environ.get('CUSTOM_OCR_RETRY_JITTER', '0.4'))
        self.retry_on_empty = os.environ.get('CUSTOM_OCR_RETRY_ON_EMPTY', '1') == '1'
        
    def check_installation(self) -> bool:
        """Always installed - uses UnifiedClient"""
        self.is_installed = True
        return True
    
    def install(self, progress_callback=None) -> bool:
        """No installation needed for API-based provider"""
        return self.check_installation()
    
    def load_model(self, **kwargs) -> bool:
        """Initialize UnifiedClient with current settings"""
        try:
            from unified_api_client import UnifiedClient
            
            # Support passing API key from GUI if available
            if 'api_key' in kwargs:
                api_key = kwargs['api_key']
            else:
                api_key = os.environ.get('API_KEY', '') or os.environ.get('OPENAI_API_KEY', '')
            
            if 'model' in kwargs:
                model = kwargs['model']
            else:
                model = os.environ.get('MODEL', 'gpt-4o-mini')
            
            if not api_key:
                self._log("❌ No API key configured", "error")
                return False
            
            # Create UnifiedClient just like translations do
            self.client = UnifiedClient(model=model, api_key=api_key)
            
            #self._log(f"✅ Using {model} for OCR via UnifiedClient")
            self.is_loaded = True
            return True
            
        except Exception as e:
            self._log(f"❌ Failed to initialize UnifiedClient: {str(e)}", "error")
            return False
    
    def _test_connection(self) -> bool:
        """Test API connection with a simple request"""
        try:
            # Create a small test image
            test_image = np.ones((100, 100, 3), dtype=np.uint8) * 255
            cv2.putText(test_image, "TEST", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            
            # Encode image
            image_base64 = self._encode_image(test_image)
            
            # Prepare test request based on API format
            if self.api_format == 'openai':
                test_payload = {
                    "model": self.model_name,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "What text do you see?"},
                                {"type": "image_url", "image_url": {"url": f"data:image/{self.image_format};base64,{image_base64}"}}
                            ]
                        }
                    ],
                    "max_tokens": 50
                }
            else:
                # For other formats, just try a basic health check
                return True
            
            headers = self._prepare_headers()
            response = requests.post(
                self.api_url,
                headers=headers,
                json=test_payload,
                timeout=10
            )
            
            return response.status_code == 200
            
        except Exception:
            return False
    
    def _encode_image(self, image: np.ndarray) -> str:
        """Encode numpy array to base64 string"""
        # Convert BGR to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        # Convert to PIL Image
        pil_image = Image.fromarray(image_rgb)
        
        # Save to bytes buffer
        buffer = io.BytesIO()
        if self.image_format.lower() == 'png':
            pil_image.save(buffer, format='PNG')
        else:
            pil_image.save(buffer, format='JPEG', quality=self.image_quality)
        
        # Encode to base64
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        
        return image_base64
    
    def _prepare_headers(self) -> dict:
        """Prepare request headers"""
        headers = {
            "Content-Type": "application/json"
        }
        
        # Add API key if configured
        if self.api_key:
            if self.api_format == 'anthropic':
                headers["x-api-key"] = self.api_key
            else:
                headers["Authorization"] = f"Bearer {self.api_key}"
        
        # Add any custom headers
        headers.update(self.api_headers)
        
        return headers
    
    def _prepare_request_payload(self, image_base64: str) -> dict:
        """Prepare request payload based on API format"""
        if self.api_format == 'openai':
            return {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": self.ocr_prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/{self.image_format};base64,{image_base64}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": self.max_tokens,
                "temperature": self.temperature
            }
        
        elif self.api_format == 'anthropic':
            return {
                "model": self.model_name,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": self.ocr_prompt
                            },
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": f"image/{self.image_format}",
                                    "data": image_base64
                                }
                            }
                        ]
                    }
                ]
            }
        
        else:
            # Custom format - use environment variable for template
            template = os.environ.get('CUSTOM_OCR_REQUEST_TEMPLATE', '{}')
            payload = json.loads(template)
            
            # Replace placeholders
            payload_str = json.dumps(payload)
            payload_str = payload_str.replace('{{IMAGE_BASE64}}', image_base64)
            payload_str = payload_str.replace('{{PROMPT}}', self.ocr_prompt)
            payload_str = payload_str.replace('{{MODEL}}', self.model_name)
            payload_str = payload_str.replace('{{MAX_TOKENS}}', str(self.max_tokens))
            payload_str = payload_str.replace('{{TEMPERATURE}}', str(self.temperature))
            
            return json.loads(payload_str)
    
    def _extract_text_from_response(self, response_data: dict) -> str:
        """Extract text from API response based on format"""
        try:
            if self.api_format == 'openai':
                # OpenAI format: response.choices[0].message.content
                return response_data.get('choices', [{}])[0].get('message', {}).get('content', '')
            
            elif self.api_format == 'anthropic':
                # Anthropic format: response.content[0].text
                content = response_data.get('content', [])
                if content and isinstance(content, list):
                    return content[0].get('text', '')
                return ''
            
            else:
                # Custom format - use environment variable for path
                response_path = os.environ.get('CUSTOM_OCR_RESPONSE_PATH', 'text')
                
                # Navigate through the response using the path
                result = response_data
                for key in response_path.split('.'):
                    if isinstance(result, dict):
                        result = result.get(key, '')
                    elif isinstance(result, list) and key.isdigit():
                        idx = int(key)
                        result = result[idx] if idx < len(result) else ''
                    else:
                        result = ''
                        break
                
                return str(result)
                
        except Exception as e:
            self._log(f"Failed to extract text from response: {e}", "error")
            return ''
    
    def detect_text(self, image: np.ndarray, **kwargs) -> List[OCRResult]:
        """Process image using UnifiedClient.send_image()"""
        results = []
        
        try:
            # Get fresh max_tokens from environment - GUI will have set this
            max_tokens = int(os.environ.get('MAX_OUTPUT_TOKENS', '4096'))
            if not self.is_loaded:
                if not self.load_model():
                    return results
            
            import cv2
            from PIL import Image
            import base64
            import io
            
            # Convert numpy array to PIL Image
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            h, w = image.shape[:2]
            
            # Convert PIL Image to base64 string
            buffer = io.BytesIO()
            
            # Use the image format from settings
            if self.image_format.lower() == 'png':
                pil_image.save(buffer, format='PNG')
            else:
                pil_image.save(buffer, format='JPEG', quality=self.image_quality)
            
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
            
            # For OpenAI vision models, we need BOTH:
            # 1. System prompt with instructions
            # 2. User message that includes the image
            messages = [
                {
                    "role": "system",
                    "content": self.ocr_prompt  # The OCR instruction as system prompt
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Image:"  # Minimal text, just to have something
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ]
            
            # Now send this properly formatted message
            # The UnifiedClient should handle this correctly
            # But we're NOT using send_image, we're using regular send

            # Retry-aware call
            from unified_api_client import UnifiedClientError  # local import to avoid hard dependency at module import time
            max_attempts = max(1, self.max_retries)
            attempt = 0
            last_error = None

            # Common refusal/error phrases that indicate a non-OCR response
            refusal_phrases = [
                "I can't extract", "I cannot extract",
                "I'm sorry", "I am sorry",
                "I'm unable", "I am unable",
                "cannot process images",
                "I can't help with that",
                "cannot view images",
                "no text in the image"
            ]

            while attempt < max_attempts:
                # Check for stop before each attempt
                if self._check_stop():
                    self._log("⏹️ OCR processing stopped by user", "warning")
                    return results
                
                try:
                    response = self.client.send(
                        messages=messages,
                        temperature=self.temperature,
                        max_tokens=max_tokens
                    )

                    # Extract content from response object
                    content, finish_reason = response

                    # Validate content
                    has_content = bool(content and str(content).strip())
                    refused = False
                    if has_content:
                        # Filter out explicit failure markers
                        if "[" in content and "FAILED]" in content:
                            refused = True
                        elif any(phrase.lower() in content.lower() for phrase in refusal_phrases):
                            refused = True

                    # Decide success or retry
                    if has_content and not refused:
                        text = str(content).strip()
                        results.append(OCRResult(
                            text=text,
                            bbox=(0, 0, w, h),
                            confidence=kwargs.get('confidence', 0.85),
                            vertices=[(0, 0), (w, 0), (w, h), (0, h)]
                        ))
                        self._log(f"✅ Detected: {text[:50]}...")
                        break  # success
                    else:
                        reason = "empty result" if not has_content else "refusal/non-OCR response"
                        last_error = f"{reason} (finish_reason: {finish_reason})"
                        # Check if we should retry on empty or refusal
                        should_retry = (not has_content and self.retry_on_empty) or refused
                        attempt += 1
                        if attempt >= max_attempts or not should_retry:
                            # No more retries or shouldn't retry
                            if not has_content:
                                self._log(f"⚠️ No text detected (finish_reason: {finish_reason})")
                            else:
                                self._log(f"❌ Model returned non-OCR response: {str(content)[:120]}", "warning")
                            break
                        # Backoff before retrying
                        delay = self.retry_initial_delay * (self.retry_backoff ** (attempt - 1)) + random.uniform(0, self.retry_jitter)
                        self._log(f"🔄 Retry {attempt}/{max_attempts - 1} after {delay:.1f}s due to {reason}...", "warning")
                        time.sleep(delay)
                        time.sleep(0.1)  # Brief pause for stability
                        self._log("💤 OCR retry pausing briefly for stability", "debug")
                        continue

                except UnifiedClientError as ue:
                    msg = str(ue)
                    last_error = msg
                    # Do not retry on explicit user cancellation
                    if 'cancelled' in msg.lower() or 'stopped by user' in msg.lower():
                        self._log(f"❌ OCR cancelled: {msg}", "error")
                        break
                    attempt += 1
                    if attempt >= max_attempts:
                        self._log(f"❌ OCR failed after {attempt} attempts: {msg}", "error")
                        break
                    delay = self.retry_initial_delay * (self.retry_backoff ** (attempt - 1)) + random.uniform(0, self.retry_jitter)
                    self._log(f"🔄 API error, retry {attempt}/{max_attempts - 1} after {delay:.1f}s: {msg}", "warning")
                    time.sleep(delay)
                    time.sleep(0.1)  # Brief pause for stability
                    self._log("💤 OCR API error retry pausing briefly for stability", "debug")
                    continue
                except Exception as e_inner:
                    last_error = str(e_inner)
                    attempt += 1
                    if attempt >= max_attempts:
                        self._log(f"❌ OCR exception after {attempt} attempts: {last_error}", "error")
                        break
                    delay = self.retry_initial_delay * (self.retry_backoff ** (attempt - 1)) + random.uniform(0, self.retry_jitter)
                    self._log(f"🔄 Exception, retry {attempt}/{max_attempts - 1} after {delay:.1f}s: {last_error}", "warning")
                    time.sleep(delay)
                    time.sleep(0.1)  # Brief pause for stability
                    self._log("💤 OCR exception retry pausing briefly for stability", "debug")
                    continue
        
        except Exception as e:
            self._log(f"❌ Error: {str(e)}", "error")
            import traceback
            self._log(traceback.format_exc(), "debug")
        
        return results

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
    
    def _is_valid_local_model_dir(self, path: str) -> bool:
        """Check that a local HF model directory has required files."""
        try:
            if not path or not os.path.isdir(path):
                return False
            needed_any_weights = any(
                os.path.exists(os.path.join(path, name)) for name in (
                    'pytorch_model.bin',
                    'model.safetensors'
                )
            )
            has_config = os.path.exists(os.path.join(path, 'config.json'))
            has_processor = (
                os.path.exists(os.path.join(path, 'preprocessor_config.json')) or
                os.path.exists(os.path.join(path, 'processor_config.json'))
            )
            has_tokenizer = (
                os.path.exists(os.path.join(path, 'tokenizer.json')) or
                os.path.exists(os.path.join(path, 'tokenizer_config.json'))
            )
            return has_config and needed_any_weights and has_processor and has_tokenizer
        except Exception:
            return False
    
    def load_model(self, **kwargs) -> bool:
        """Load the manga-ocr model, preferring a local directory to avoid re-downloading"""
        print("\n>>> MangaOCRProvider.load_model() called")
        try:
            if not self.is_installed and not self.check_installation():
                print("ERROR: Transformers not installed")
                self._log("❌ Transformers not installed", "error")
                return False

            # Always disable progress bars to avoid tqdm issues in some environments
            import os
            os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

            from transformers import VisionEncoderDecoderModel, AutoTokenizer, AutoImageProcessor
            import torch

            # Prefer a local model directory if present to avoid any Hub access
            candidates = []
            env_local = os.environ.get("MANGA_OCR_LOCAL_DIR")
            if env_local:
                candidates.append(env_local)

            # Project root one level up from this file
            root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
            candidates.append(os.path.join(root_dir, 'models', 'manga-ocr-base'))
            candidates.append(os.path.join(root_dir, 'models', 'kha-white', 'manga-ocr-base'))

            model_source = None
            local_only = False
            # Find a valid local dir
            for cand in candidates:
                if self._is_valid_local_model_dir(cand):
                    model_source = cand
                    local_only = True
                    break

            # If no valid local dir, use Hub
            if not model_source:
                model_source = "kha-white/manga-ocr-base"
                # Make sure we are not forcing offline mode
                if os.environ.get("HF_HUB_OFFLINE") == "1":
                    try:
                        del os.environ["HF_HUB_OFFLINE"]
                    except Exception:
                        pass
                self._log("🔥 Loading manga-ocr model from Hugging Face Hub")
                self._log(f"   Repo: {model_source}")
            else:
                # Only set offline when local dir is fully valid
                os.environ.setdefault("HF_HUB_OFFLINE", "1")
                self._log("🔥 Loading manga-ocr model from local directory")
                self._log(f"   Local path: {model_source}")

            # Decide target device once; we will move after full CPU load to avoid meta tensors
            use_cuda = torch.cuda.is_available()

            # Try loading components, falling back to Hub if local-only fails
            def _load_components(source: str, local_flag: bool):
                self._log("   Loading tokenizer...")
                tok = AutoTokenizer.from_pretrained(source, local_files_only=local_flag)

                self._log("   Loading image processor...")
                try:
                    from transformers import AutoProcessor
                except Exception:
                    AutoProcessor = None
                try:
                    proc = AutoImageProcessor.from_pretrained(source, local_files_only=local_flag)
                except Exception as e_proc:
                    if AutoProcessor is not None:
                        self._log(f"   ⚠️ AutoImageProcessor failed: {e_proc}. Trying AutoProcessor...", "warning")
                        proc = AutoProcessor.from_pretrained(source, local_files_only=local_flag)
                    else:
                        raise

                self._log("   Loading model...")
                # Prevent meta tensors by forcing full materialization on CPU at load time
                os.environ.setdefault('TORCHDYNAMO_DISABLE', '1')
                mdl = VisionEncoderDecoderModel.from_pretrained(
                    source,
                    local_files_only=local_flag,
                    low_cpu_mem_usage=False,
                    device_map=None,
                    torch_dtype=torch.float32  # Use torch_dtype instead of dtype
                )
                return tok, proc, mdl

            try:
                self.tokenizer, self.processor, self.model = _load_components(model_source, local_only)
            except Exception as e_local:
                if local_only:
                    # Fallback to Hub once if local fails
                    self._log(f"   ⚠️ Local model load failed: {e_local}", "warning")
                    try:
                        if os.environ.get("HF_HUB_OFFLINE") == "1":
                            del os.environ["HF_HUB_OFFLINE"]
                    except Exception:
                        pass
                    model_source = "kha-white/manga-ocr-base"
                    local_only = False
                    self._log("   Retrying from Hugging Face Hub...")
                    self.tokenizer, self.processor, self.model = _load_components(model_source, local_only)
                else:
                    raise

            # Move to CUDA only after full CPU materialization
            target_device = 'cpu'
            if use_cuda:
                try:
                    self.model = self.model.to('cuda')
                    target_device = 'cuda'
                except Exception as move_err:
                    self._log(f"   ⚠️ Could not move model to CUDA: {move_err}", "warning")
                    target_device = 'cpu'

            # Finalize eval mode
            self.model.eval()

            # Sanity-check: ensure no parameter remains on 'meta' device
            try:
                for n, p in self.model.named_parameters():
                    dev = getattr(p, 'device', None)
                    if dev is not None and getattr(dev, 'type', '') == 'meta':
                        raise RuntimeError(f"Parameter {n} is on 'meta' after load")
            except Exception as sanity_err:
                self._log(f"❌ Manga-OCR model load sanity check failed: {sanity_err}", "error")
                return False

            print(f"SUCCESS: Model loaded on {target_device.upper()}")
            self._log(f"   ✅ Model loaded on {target_device.upper()}")
            self.is_loaded = True
            self._log("✅ Manga OCR model ready")
            print(">>> Returning True from load_model()")
            return True

        except Exception as e:
            print(f"\nEXCEPTION in load_model: {e}")
            import traceback
            print(traceback.format_exc())
            self._log(f"❌ Failed to load manga-ocr model: {str(e)}", "error")
            self._log(traceback.format_exc(), "error")
            try:
                if 'local_only' in locals() and local_only:
                    self._log("Hint: Local load failed. Ensure your models/manga-ocr-base contains required files (config.json, preprocessor_config.json, tokenizer.json or tokenizer_config.json, and model weights).", "warning")
            except Exception:
                pass
            return False
    
    def _run_ocr(self, pil_image):
        """Run OCR on a PIL image using the HuggingFace model"""
        import torch
        
        # Process image (keyword arg for broader compatibility across transformers versions)
        inputs = self.processor(images=pil_image, return_tensors="pt")
        pixel_values = inputs["pixel_values"]
        
        # Move to same device as model
        try:
            model_device = next(self.model.parameters()).device
        except StopIteration:
            model_device = torch.device('cpu')
        pixel_values = pixel_values.to(model_device)
        
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
        
        # Check for stop at start
        if self._check_stop():
            self._log("⏹️ Manga-OCR processing stopped by user", "warning")
            return results
        
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
            
            # Check for stop before inference
            if self._check_stop():
                self._log("⏹️ Manga-OCR inference stopped by user", "warning")
                return results
            
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
        
        # Get OCR prompt from environment or use default
        self.ocr_prompt = os.environ.get('OCR_SYSTEM_PROMPT', 
            "YOU ARE AN OCR SYSTEM. YOUR ONLY JOB IS TEXT EXTRACTION.\n\n"
            "CRITICAL RULES:\n"
            "1. DO NOT TRANSLATE ANYTHING\n"
            "2. DO NOT MODIFY THE TEXT\n"
            "3. DO NOT EXPLAIN OR COMMENT\n"
            "4. ONLY OUTPUT THE EXACT TEXT YOU SEE\n"
            "5. PRESERVE NATURAL TEXT FLOW - DO NOT ADD UNNECESSARY LINE BREAKS\n\n"
            "If you see Korean text, output it in Korean.\n"
            "If you see Japanese text, output it in Japanese.\n"
            "If you see Chinese text, output it in Chinese.\n"
            "If you see English text, output it in English.\n\n"
            "IMPORTANT: Only use line breaks where they naturally occur in the original text "
            "(e.g., between dialogue lines or paragraphs). Do not break text mid-sentence or "
            "between every word/character.\n\n"
            "For vertical text common in manga/comics, transcribe it as a continuous line unless "
            "there are clear visual breaks.\n\n"
            "NEVER translate. ONLY extract exactly what is written.\n"
            "Output ONLY the raw text, nothing else."
        )
    
    def set_ocr_prompt(self, prompt: str):
        """Allow setting the OCR prompt dynamically"""
        self.ocr_prompt = prompt
        
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
    
    def load_model(self, model_size=None, **kwargs) -> bool:
        """Load Qwen2-VL model with size selection"""
        self._log(f"DEBUG: load_model called with model_size={model_size}")

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
                model_size = os.environ.get('QWEN2VL_MODEL_SIZE', '1')
            
            # Determine which model to load
            if model_size and str(model_size).startswith("custom:"):
                # Custom model passed with ID
                model_id = str(model_size).replace("custom:", "")
                self.loaded_model_size = "Custom"
                self.model_id = model_id
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
        if hasattr(self, 'model_id'):
            self._log(f"DEBUG: Using model: {self.model_id}", "debug")
            
        # Check if OCR prompt was passed in kwargs (for dynamic updates)
        if 'ocr_prompt' in kwargs:
            self.ocr_prompt = kwargs['ocr_prompt']
        
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
            
            # Use the configurable OCR prompt
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
                            "text": self.ocr_prompt  # Use the configurable prompt
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

            # Get the device and dtype the model is currently on
            model_device = next(self.model.parameters()).device
            model_dtype = next(self.model.parameters()).dtype

            # Move inputs to the same device as the model and cast float tensors to model dtype
            try:
                # Move first
                inputs = inputs.to(model_device)
                # Then align dtypes only for floating tensors (e.g., pixel_values)
                for k, v in inputs.items():
                    if isinstance(v, torch.Tensor) and torch.is_floating_point(v):
                        inputs[k] = v.to(model_dtype)
            except Exception:
                # Fallback: ensure at least pixel_values is correct if present
                try:
                    if isinstance(inputs, dict) and "pixel_values" in inputs:
                        pv = inputs["pixel_values"].to(model_device)
                        if torch.is_floating_point(pv):
                            inputs["pixel_values"] = pv.to(model_dtype)
                except Exception:
                    pass

            # Ensure pixel_values explicitly matches model dtype if present
            try:
                if isinstance(inputs, dict) and "pixel_values" in inputs:
                    inputs["pixel_values"] = inputs["pixel_values"].to(device=model_device, dtype=model_dtype)
            except Exception:
                pass

            # Generate text with stricter parameters to avoid creative responses
            use_amp = (hasattr(torch, 'cuda') and model_device.type == 'cuda' and model_dtype in (torch.float16, torch.bfloat16))
            autocast_dev = 'cuda' if model_device.type == 'cuda' else 'cpu'
            autocast_dtype = model_dtype if model_dtype in (torch.float16, torch.bfloat16) else None

            with torch.no_grad():
                if use_amp and autocast_dtype is not None:
                    with torch.autocast(autocast_dev, dtype=autocast_dtype):
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
                else:
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
    
    def load_model(self, **kwargs) -> bool:
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
    """PaddleOCR provider with memory safety measures"""
    
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
    
    def load_model(self, **kwargs) -> bool:
        """Load paddleocr model with memory-safe configurations"""
        try:
            if not self.is_installed and not self.check_installation():
                self._log("❌ paddleocr not installed", "error")
                return False
            
            self._log("🔥 Loading PaddleOCR model...")
            
            # Set memory-safe environment variables BEFORE importing
            import os
            os.environ['OMP_NUM_THREADS'] = '1'  # Prevent OpenMP conflicts
            os.environ['MKL_NUM_THREADS'] = '1'  # Prevent MKL conflicts
            os.environ['OPENBLAS_NUM_THREADS'] = '1'  # Prevent OpenBLAS conflicts
            os.environ['FLAGS_use_mkldnn'] = '0'  # Disable MKL-DNN
            
            from paddleocr import PaddleOCR
            
            # Try memory-safe configurations
            configs_to_try = [
                # Config 1: Most memory-safe configuration
                {
                    'use_angle_cls': False,  # Disable angle to save memory
                    'lang': 'ch',
                    'rec_batch_num': 1,  # Process one at a time
                    'max_text_length': 100,  # Limit text length
                    'drop_score': 0.5,  # Higher threshold to reduce detections
                    'cpu_threads': 1,  # Single thread to avoid conflicts
                },
                # Config 2: Minimal memory footprint
                {
                    'lang': 'ch',
                    'rec_batch_num': 1,
                    'cpu_threads': 1,
                },
                # Config 3: Absolute minimal
                {
                    'lang': 'ch'
                },
                # Config 4: Empty config
                {}
            ]
            
            for i, config in enumerate(configs_to_try):
                try:
                    self._log(f"   Trying configuration {i+1}/{len(configs_to_try)}: {config}")
                    
                    # Force garbage collection before loading
                    import gc
                    gc.collect()
                    
                    self.model = PaddleOCR(**config)
                    self.is_loaded = True
                    self.current_config = config
                    self._log(f"✅ PaddleOCR loaded successfully with config: {config}")
                    return True
                except Exception as e:
                    error_str = str(e)
                    self._log(f"   Config {i+1} failed: {error_str}", "debug")
                    
                    # Clean up on failure
                    if hasattr(self, 'model'):
                        del self.model
                    gc.collect()
                    continue
            
            self._log(f"❌ PaddleOCR failed to load with any configuration", "error")
            return False
            
        except Exception as e:
            self._log(f"❌ Failed to load paddleocr: {str(e)}", "error")
            import traceback
            self._log(traceback.format_exc(), "debug")
            return False
    
    def detect_text(self, image: np.ndarray, **kwargs) -> List[OCRResult]:
        """Detect text with memory safety measures"""
        results = []
        
        try:
            if not self.is_loaded:
                if not self.load_model():
                    return results
            
            import cv2
            import numpy as np
            import gc
            
            # Memory safety: Ensure image isn't too large
            h, w = image.shape[:2] if len(image.shape) >= 2 else (0, 0)
            
            # Limit image size to prevent memory issues
            MAX_DIMENSION = 1500
            if h > MAX_DIMENSION or w > MAX_DIMENSION:
                scale = min(MAX_DIMENSION/h, MAX_DIMENSION/w)
                new_h, new_w = int(h*scale), int(w*scale)
                self._log(f"⚠️ Resizing large image from {w}x{h} to {new_w}x{new_h} for memory safety", "warning")
                image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
                scale_factor = 1/scale
            else:
                scale_factor = 1.0
            
            # Ensure correct format
            if len(image.shape) == 2:  # Grayscale
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif len(image.shape) == 4:  # Batch
                image = image[0]
            
            # Ensure uint8 type
            if image.dtype != np.uint8:
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
                else:
                    image = image.astype(np.uint8)
            
            # Make a copy to avoid memory corruption
            image_copy = image.copy()
            
            # Force garbage collection before OCR
            gc.collect()
            
            # Process with timeout protection
            import signal
            import threading
            
            ocr_results = None
            ocr_error = None
            
            def run_ocr():
                nonlocal ocr_results, ocr_error
                try:
                    ocr_results = self.model.ocr(image_copy)
                except Exception as e:
                    ocr_error = e
            
            # Run OCR in a separate thread with timeout
            ocr_thread = threading.Thread(target=run_ocr)
            ocr_thread.daemon = True
            ocr_thread.start()
            ocr_thread.join(timeout=30)  # 30 second timeout
            
            if ocr_thread.is_alive():
                self._log("❌ PaddleOCR timeout - taking too long", "error")
                return results
            
            if ocr_error:
                raise ocr_error
            
            # Parse results
            results = self._parse_ocr_results(ocr_results)
            
            # Scale coordinates back if image was resized
            if scale_factor != 1.0 and results:
                for r in results:
                    x, y, width, height = r.bbox
                    r.bbox = (int(x*scale_factor), int(y*scale_factor), 
                            int(width*scale_factor), int(height*scale_factor))
                    r.vertices = [(int(v[0]*scale_factor), int(v[1]*scale_factor)) 
                                for v in r.vertices]
            
            if results:
                self._log(f"✅ Detected {len(results)} text regions", "info")
            else:
                self._log("No text regions found", "debug")
            
            # Clean up
            del image_copy
            gc.collect()
            
        except Exception as e:
            error_msg = str(e) if str(e) else type(e).__name__
            
            if "memory" in error_msg.lower() or "0x" in error_msg:
                self._log("❌ Memory access violation in PaddleOCR", "error")
                self._log("   This is a known Windows issue with PaddleOCR", "info")
                self._log("   Please switch to EasyOCR or manga-ocr instead", "warning")
            elif "trace_order.size()" in error_msg:
                self._log("❌ PaddleOCR internal error", "error")
                self._log("   Please switch to EasyOCR or manga-ocr", "warning")
            else:
                self._log(f"❌ Error in paddleocr detection: {error_msg}", "error")
            
            import traceback
            self._log(traceback.format_exc(), "debug")
        
        return results
    
    def _parse_ocr_results(self, ocr_results) -> List[OCRResult]:
        """Parse OCR results safely"""
        results = []
        
        if isinstance(ocr_results, bool) and ocr_results == False:
            return results
        
        if ocr_results is None or not isinstance(ocr_results, list):
            return results
        
        if len(ocr_results) == 0:
            return results
        
        # Handle batch format
        if isinstance(ocr_results[0], list) and len(ocr_results[0]) > 0:
            first_item = ocr_results[0][0]
            if isinstance(first_item, list) and len(first_item) > 0:
                if isinstance(first_item[0], (list, tuple)) and len(first_item[0]) == 2:
                    ocr_results = ocr_results[0]
        
        # Parse detections
        for detection in ocr_results:
            if not detection or isinstance(detection, bool):
                continue
            
            if not isinstance(detection, (list, tuple)) or len(detection) < 2:
                continue
            
            try:
                bbox_points = detection[0]
                text_data = detection[1]
                
                if not isinstance(bbox_points, (list, tuple)) or len(bbox_points) != 4:
                    continue
                
                if not isinstance(text_data, (tuple, list)) or len(text_data) < 2:
                    continue
                
                text = str(text_data[0]).strip()
                confidence = float(text_data[1])
                
                if not text or confidence < 0.3:
                    continue
                
                xs = [float(p[0]) for p in bbox_points]
                ys = [float(p[1]) for p in bbox_points]
                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)
                
                if (x_max - x_min) < 5 or (y_max - y_min) < 5:
                    continue
                
                results.append(OCRResult(
                    text=text,
                    bbox=(int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)),
                    confidence=confidence,
                    vertices=[(int(p[0]), int(p[1])) for p in bbox_points]
                ))
                
            except Exception:
                continue
        
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
    
    def load_model(self, **kwargs) -> bool:
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


class RapidOCRProvider(OCRProvider):
    """RapidOCR provider for fast local OCR"""
    
    def check_installation(self) -> bool:
        """Check if rapidocr is installed"""
        try:
            import rapidocr_onnxruntime
            self.is_installed = True
            return True
        except ImportError:
            return False
    
    def install(self, progress_callback=None) -> bool:
        """Install rapidocr (requires manual pip install)"""
        # RapidOCR requires manual installation
        if progress_callback:
            progress_callback("RapidOCR requires manual pip installation")
        self._log("Run: pip install rapidocr-onnxruntime", "info")
        return False  # Always return False since we can't auto-install
    
    def load_model(self, **kwargs) -> bool:
        """Load RapidOCR model"""
        try:
            if not self.is_installed and not self.check_installation():
                self._log("RapidOCR not installed", "error")
                return False
            
            self._log("Loading RapidOCR...")
            from rapidocr_onnxruntime import RapidOCR
            
            self.model = RapidOCR()
            self.is_loaded = True
            
            self._log("RapidOCR model loaded successfully")
            return True
            
        except Exception as e:
            self._log(f"Failed to load RapidOCR: {str(e)}", "error")
            return False
    
    def detect_text(self, image: np.ndarray, **kwargs) -> List[OCRResult]:
        """Detect text using RapidOCR"""
        if not self.is_loaded:
            self._log("RapidOCR model not loaded", "error")
            return []
        
        results = []
        
        try:
            # Convert numpy array to PIL Image for RapidOCR
            if len(image.shape) == 3:
                # BGR to RGB
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image
            
            # RapidOCR expects PIL Image or numpy array
            ocr_results, _ = self.model(image_rgb)
            
            if ocr_results:
                for result in ocr_results:
                    # RapidOCR returns [bbox, text, confidence]
                    bbox_points = result[0]  # 4 corner points
                    text = result[1]
                    confidence = float(result[2])
                    
                    if not text or not text.strip():
                        continue
                    
                    # Convert 4-point bbox to x,y,w,h format
                    xs = [point[0] for point in bbox_points]
                    ys = [point[1] for point in bbox_points]
                    x_min, x_max = min(xs), max(xs)
                    y_min, y_max = min(ys), max(ys)
                    
                    results.append(OCRResult(
                        text=text.strip(),
                        bbox=(int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)),
                        confidence=confidence,
                        vertices=[(int(p[0]), int(p[1])) for p in bbox_points]
                    ))
            
            self._log(f"Detected {len(results)} text regions")
            
        except Exception as e:
            self._log(f"Error in RapidOCR detection: {str(e)}", "error")
        
        return results

class OCRManager:
    """Manager for multiple OCR providers"""
    
    def __init__(self, log_callback=None):
        self.log_callback = log_callback
        self.providers = {
            'custom-api': CustomAPIProvider(log_callback) ,
            'manga-ocr': MangaOCRProvider(log_callback),
            'easyocr': EasyOCRProvider(log_callback),
            'paddleocr': PaddleOCRProvider(log_callback),
            'doctr': DocTROCRProvider(log_callback),
            'rapidocr': RapidOCRProvider(log_callback),
            'Qwen2-VL': Qwen2VL(log_callback)
        }
        self.current_provider = None
        self.stop_flag = None
        
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
        
        result = {
            'installed': provider.check_installation(),
            'loaded': provider.is_loaded
        }
        if self.log_callback:
            self.log_callback(f"DEBUG: check_provider_status({name}) returning loaded={result['loaded']}", "debug")
        return result
    
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
    
    def shutdown(self):
        """Release models/processors/tokenizers for all providers and clear caches."""
        try:
            import gc
            for name, provider in list(self.providers.items()):
                try:
                    if hasattr(provider, 'model'):
                        provider.model = None
                    if hasattr(provider, 'processor'):
                        provider.processor = None
                    if hasattr(provider, 'tokenizer'):
                        provider.tokenizer = None
                    if hasattr(provider, 'reader'):
                        provider.reader = None
                    if hasattr(provider, 'is_loaded'):
                        provider.is_loaded = False
                except Exception:
                    pass
            gc.collect()
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
        except Exception:
            pass

    def detect_text(self, image: np.ndarray, provider_name: str = None, **kwargs) -> List[OCRResult]:
        """Detect text using specified or current provider"""
        provider_name = provider_name or self.current_provider
        if not provider_name:
            return []
        
        provider = self.providers.get(provider_name)
        if not provider:
            return []
        
        return provider.detect_text(image, **kwargs)
    
    def set_stop_flag(self, stop_flag):
        """Set stop flag for all providers"""
        self.stop_flag = stop_flag
        for provider in self.providers.values():
            if hasattr(provider, 'set_stop_flag'):
                provider.set_stop_flag(stop_flag)
    
    def reset_stop_flags(self):
        """Reset stop flags for all providers"""
        for provider in self.providers.values():
            if hasattr(provider, 'reset_stop_flags'):
                provider.reset_stop_flags()
