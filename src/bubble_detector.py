"""
bubble_detector.py - Modified version that works in frozen PyInstaller executables
Replace your bubble_detector.py with this version
"""
import os
import sys
import json
import numpy as np
import cv2
from typing import List, Tuple, Optional, Dict, Any
import logging
import traceback
import hashlib
from pathlib import Path
import threading
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check if we're running in a frozen environment
IS_FROZEN = getattr(sys, 'frozen', False)
if IS_FROZEN:
    # In frozen environment, set proper paths for ML libraries
    MEIPASS = sys._MEIPASS
    os.environ['TORCH_HOME'] = MEIPASS
    os.environ['TRANSFORMERS_CACHE'] = os.path.join(MEIPASS, 'transformers')
    os.environ['HF_HOME'] = os.path.join(MEIPASS, 'huggingface')
    logger.info(f"Running in frozen environment: {MEIPASS}")

# Modified import checks for frozen environment
YOLO_AVAILABLE = False
YOLO = None
torch = None
TORCH_AVAILABLE = False
ONNX_AVAILABLE = False
TRANSFORMERS_AVAILABLE = False
RTDetrForObjectDetection = None
RTDetrImageProcessor = None
PIL_AVAILABLE = False

# Try to import YOLO dependencies with better error handling
if IS_FROZEN:
    # In frozen environment, try harder to import
    try:
        # First try to import torch components individually
        import torch
        import torch.nn
        import torch.cuda
        TORCH_AVAILABLE = True
        logger.info("✓ PyTorch loaded in frozen environment")
    except Exception as e:
        logger.warning(f"PyTorch not available in frozen environment: {e}")
        TORCH_AVAILABLE = False
        torch = None
    
    # Try ultralytics after torch
    if TORCH_AVAILABLE:
        try:
            from ultralytics import YOLO
            YOLO_AVAILABLE = True
            logger.info("✓ Ultralytics YOLO loaded in frozen environment")
        except Exception as e:
            logger.warning(f"Ultralytics not available in frozen environment: {e}")
            YOLO_AVAILABLE = False
    
    # Try transformers
    try:
        import transformers
        # Try specific imports
        try:
            from transformers import RTDetrV2ForObjectDetection, RTDetrImageProcessor
            TRANSFORMERS_AVAILABLE = True
            logger.info("Transformers RT-DETR V2 available")
        except ImportError:
            # Try alternative import
            try:
                from transformers import AutoModel, AutoImageProcessor
                RTDetrForObjectDetection = AutoModel
                RTDetrImageProcessor = AutoImageProcessor
                TRANSFORMERS_AVAILABLE = True
                logger.info("✓ Transformers loaded with AutoModel fallback")
            except:
                TRANSFORMERS_AVAILABLE = False
                logger.warning("Transformers RT-DETR not available in frozen environment")
    except Exception as e:
        logger.warning(f"Transformers not available in frozen environment: {e}")
        TRANSFORMERS_AVAILABLE = False
else:
    # Normal environment - original import logic
    try:
        from ultralytics import YOLO
        YOLO_AVAILABLE = True
    except:
        YOLO_AVAILABLE = False
        logger.warning("Ultralytics YOLO not available")

    try:
        import torch
        # Test if cuda attribute exists
        _ = torch.cuda
        TORCH_AVAILABLE = True
    except (ImportError, AttributeError):
        TORCH_AVAILABLE = False
        torch = None
        logger.warning("PyTorch not available or incomplete")

    try:
        from transformers import RTDetrForObjectDetection, RTDetrImageProcessor
        try:
            from transformers import RTDetrV2ForObjectDetection
            RTDetrForObjectDetection = RTDetrV2ForObjectDetection
        except ImportError:
            pass
        TRANSFORMERS_AVAILABLE = True
    except:
        TRANSFORMERS_AVAILABLE = False
        logger.info("Transformers not available for RT-DETR")

# Configure ORT memory behavior before importing
try:
    os.environ.setdefault('ORT_DISABLE_MEMORY_ARENA', '1')
except Exception:
    pass
# ONNX Runtime - works well in frozen environments
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
    logger.info("✓ ONNX Runtime available")
except ImportError:
    ONNX_AVAILABLE = False
    logger.warning("ONNX Runtime not available")

# PIL
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logger.info("PIL not available")


class BubbleDetector:
    """
    Combined YOLOv8 and RT-DETR speech bubble detector for comics and manga.
    Supports multiple model formats and provides configurable detection.
    Backward compatible with existing code while adding RT-DETR support.
    """
    
    # VRAM monitoring settings
    _VRAM_WARNING_THRESHOLD = 0.90  # Warn when VRAM usage is above 90%
    _last_vram_warning = 0          # Track last warning time to avoid spam
    _VRAM_WARNING_COOLDOWN = 30     # Seconds between warnings
    
    @classmethod
    def _check_vram_usage(cls, device_or_id=0):
        """Monitor VRAM usage and warn if getting close to capacity.
        Accepts either a CUDA device index (int) or a torch.device.
        """
        try:
            if not torch.cuda.is_available():
                return

            # Normalize device to an index
            device_index = None
            try:
                # If int, treat as index
                if isinstance(device_or_id, int):
                    device_index = device_or_id
                # If torch.device with cuda type
                elif hasattr(device_or_id, 'type') and getattr(device_or_id, 'type', None) == 'cuda':
                    device_index = getattr(device_or_id, 'index', None)
                # Fallback to current device
                if device_index is None:
                    device_index = torch.cuda.current_device()
            except Exception:
                device_index = 0

            import time
            current_time = time.time()
            if current_time - cls._last_vram_warning < cls._VRAM_WARNING_COOLDOWN:
                return

            total = torch.cuda.get_device_properties(device_index).total_memory
            reserved = torch.cuda.memory_reserved(device_index)
            allocated = torch.cuda.memory_allocated(device_index)
            used = max(reserved, allocated)
            usage_percent = used / total if total > 0 else 0.0

            if usage_percent >= cls._VRAM_WARNING_THRESHOLD:
                free_mb = (total - used) / (1024 * 1024)
                total_mb = total / (1024 * 1024)
                logger.warning(
                    f"VRAM usage high ({usage_percent:.1%})! Only {free_mb:.0f}MB free out of {total_mb:.0f}MB. "
                    "Consider reducing batch size, lowering precision, or staggering model loads."
                )
                cls._last_vram_warning = current_time
        except Exception as e:
            logger.debug(f"VRAM check failed: {e}")
    
    def _resolve_precision_dtype(self, device):
        """Return torch.dtype based on advanced.precision toggle and device.
        Values accepted: 'auto', 'fp32'/'float32', 'fp16'/'float16', 'bf16'/'bfloat16'.
        auto => fp16 on CUDA, fp32 on CPU.
        fp16/bf16 on CPU are coerced to fp32 for safety.
        """
        try:
            if not TORCH_AVAILABLE or torch is None:
                return None
            prec = None
            # Prefer explicit setting if present
            # self.quantize_dtype may be set; also advanced.torch_precision -> self.torch_precision may exist in other modules
            prec = str(getattr(self, 'quantize_dtype', 'auto') or 'auto').lower()
            if prec == 'auto':
                return torch.float16 if (device and getattr(device, 'type', 'cpu') == 'cuda') else torch.float32
            if prec in ('fp32', 'float32', '32', 'f32'):
                return torch.float32
            if prec in ('fp16', 'float16', '16', 'f16'):
                if device and getattr(device, 'type', 'cpu') == 'cpu':
                    return torch.float32
                return torch.float16
            if prec in ('bf16', 'bfloat16'):
                if device and getattr(device, 'type', 'cpu') == 'cpu':
                    return torch.float32
                # Prefer bfloat16 only if supported; otherwise fallback to float16
                try:
                    return torch.bfloat16
                except Exception:
                    return torch.float16
            # Fallback
            return torch.float32
        except Exception:
            return torch.float32
    
    # Process-wide shared RT-DETR to avoid concurrent meta-device loads
    _rtdetr_init_lock = threading.Lock()
    _rtdetr_shared_model = None
    _rtdetr_shared_processor = None
    _rtdetr_loaded = False
    _rtdetr_repo_id = 'ogkalu/comic-text-and-bubble-detector'
    
    # Shared RT-DETR (ONNX) across process to avoid device/context storms
    _rtdetr_onnx_init_lock = threading.Lock()
    _rtdetr_onnx_shared_session = None
    _rtdetr_onnx_loaded = False
    _rtdetr_onnx_providers = None
    _rtdetr_onnx_model_path = None
    # Limit concurrent runs to avoid device hangs. Defaults to 2 for better parallelism.
    # Can be overridden via env DML_MAX_CONCURRENT or config rtdetr_max_concurrency
    try:
        _rtdetr_onnx_max_concurrent = int(os.environ.get('DML_MAX_CONCURRENT', '2'))
    except Exception:
        _rtdetr_onnx_max_concurrent = 2
    _rtdetr_onnx_sema = threading.Semaphore(max(1, _rtdetr_onnx_max_concurrent))
    _rtdetr_onnx_sema_initialized = False
    
    def __init__(self, config_path: str = "config.json"):
        """
        Initialize the bubble detector.
        
        Args:
            config_path: Path to configuration file
        """
        # Set thread limits early if environment indicates single-threaded mode
        try:
            if os.environ.get('OMP_NUM_THREADS') == '1':
                # Already in single-threaded mode, ensure it's applied to this process
                # Check if torch is available at module level before trying to use it
                if TORCH_AVAILABLE and torch is not None:
                    try:
                        torch.set_num_threads(1)
                    except (RuntimeError, AttributeError):
                        pass
                try:
                    import cv2
                    cv2.setNumThreads(1)
                except (ImportError, AttributeError):
                    pass
        except Exception:
            pass
        
        self.config_path = config_path
        self.config = self._load_config()
        
        # YOLOv8 components (original)
        self.model = None
        self.model_loaded = False
        self.model_type = None  # 'yolo', 'onnx', or 'torch'
        self.onnx_session = None
        
        # RT-DETR components (new)
        self.rtdetr_model = None
        self.rtdetr_processor = None
        self.rtdetr_loaded = False
        self.rtdetr_repo = 'ogkalu/comic-text-and-bubble-detector'

        # RT-DETR (ONNX) backend components
        self.rtdetr_onnx_session = None
        self.rtdetr_onnx_loaded = False
        self.rtdetr_onnx_repo = 'ogkalu/comic-text-and-bubble-detector'
        
        # RT-DETR class definitions
        self.CLASS_BUBBLE = 0      # Empty speech bubble
        self.CLASS_TEXT_BUBBLE = 1 # Bubble with text
        self.CLASS_TEXT_FREE = 2   # Text without bubble
        
        # Detection settings
        self.default_confidence = 0.3
        self.default_iou_threshold = 0.45
        # Allow override from settings
        try:
            ocr_cfg = self.config.get('manga_settings', {}).get('ocr', {}) if isinstance(self.config, dict) else {}
            self.default_max_detections = int(ocr_cfg.get('bubble_max_detections', 100))
            self.max_det_yolo = int(ocr_cfg.get('bubble_max_detections_yolo', self.default_max_detections))
            self.max_det_rtdetr = int(ocr_cfg.get('bubble_max_detections_rtdetr', self.default_max_detections))
        except Exception:
            self.default_max_detections = 100
            self.max_det_yolo = 100
            self.max_det_rtdetr = 100
        
        # Cache directory for ONNX conversions
        self.cache_dir = os.environ.get('BUBBLE_CACHE_DIR', 'models')
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # RT-DETR concurrency setting from config
        try:
            rtdetr_max_conc = int(ocr_cfg.get('rtdetr_max_concurrency', 2))
            # Update class-level semaphore if not yet initialized or if value changed
            if not BubbleDetector._rtdetr_onnx_sema_initialized or rtdetr_max_conc != BubbleDetector._rtdetr_onnx_max_concurrent:
                BubbleDetector._rtdetr_onnx_max_concurrent = max(1, rtdetr_max_conc)
                BubbleDetector._rtdetr_onnx_sema = threading.Semaphore(BubbleDetector._rtdetr_onnx_max_concurrent)
                BubbleDetector._rtdetr_onnx_sema_initialized = True
                logger.info(f"RT-DETR concurrency set to: {BubbleDetector._rtdetr_onnx_max_concurrent}")
        except Exception as e:
            logger.warning(f"Failed to set RT-DETR concurrency: {e}")
        
        # GPU availability
        self.use_gpu = TORCH_AVAILABLE and torch.cuda.is_available()
        self.device = 'cuda' if self.use_gpu else 'cpu'
        
        # Quantization/precision settings
        adv_cfg = self.config.get('manga_settings', {}).get('advanced', {}) if isinstance(self.config, dict) else {}
        ocr_cfg = self.config.get('manga_settings', {}).get('ocr', {}) if isinstance(self.config, dict) else {}
        env_quant = os.environ.get('MODEL_QUANTIZE', 'false').lower() == 'true'
        self.quantize_enabled = bool(env_quant or adv_cfg.get('quantize_models', False) or ocr_cfg.get('quantize_bubble_detector', False))
        self.quantize_dtype = str(adv_cfg.get('torch_precision', os.environ.get('TORCH_PRECISION', 'auto'))).lower()
        # Prefer advanced.onnx_quantize; fall back to env or global quantize
        self.onnx_quantize_enabled = bool(adv_cfg.get('onnx_quantize', os.environ.get('ONNX_QUANTIZE', 'false').lower() == 'true' or self.quantize_enabled))
        
        # Stop flag support
        self.stop_flag = None
        self._stopped = False
        self.log_callback = None
        
        logger.info(f"🗨️ BubbleDetector initialized")
        logger.info(f"   GPU: {'Available' if self.use_gpu else 'Not available'}")
        logger.info(f"   YOLO: {'Available' if YOLO_AVAILABLE else 'Not installed'}")
        logger.info(f"   ONNX: {'Available' if ONNX_AVAILABLE else 'Not installed'}")
        logger.info(f"   RT-DETR: {'Available' if TRANSFORMERS_AVAILABLE else 'Not installed'}")
        logger.info(f"   Quantization: {'ENABLED' if self.quantize_enabled else 'disabled'} (torch_precision={self.quantize_dtype}, onnx_quantize={'on' if self.onnx_quantize_enabled else 'off'})" )
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file."""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load config: {e}")
        return {}
    
    def _save_config(self):
        """Save configuration to file."""
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
    
    def set_stop_flag(self, stop_flag):
        """Set the stop flag for checking interruptions"""
        self.stop_flag = stop_flag
        self._stopped = False
    
    def set_log_callback(self, log_callback):
        """Set log callback for GUI integration"""
        self.log_callback = log_callback
    
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
    
    def _log(self, message: str, level: str = "info"):
        """Log message with stop suppression"""
        # Suppress logs when stopped (allow only essential stop confirmation messages)
        if self._check_stop():
            essential_stop_keywords = [
                "⏹️ Translation stopped by user",
                "⏹️ Bubble detection stopped",
                "cleanup", "🧹"
            ]
            if not any(keyword in message for keyword in essential_stop_keywords):
                return
        
        if self.log_callback:
            self.log_callback(message, level)
        else:
            logger.info(message) if level == 'info' else getattr(logger, level, logger.info)(message)
    
    def reset_stop_flags(self):
        """Reset stop flags when starting new processing"""
        self._stopped = False
        
    def load_model(self, model_path: str, force_reload: bool = False) -> bool:
        """
        Load a YOLOv8 model for bubble detection.
        
        Args:
            model_path: Path to model file (.pt, .onnx, or .torchscript)
            force_reload: Force reload even if model is already loaded
            
        Returns:
            True if model loaded successfully, False otherwise
        """
        try:
            # If given a Hugging Face repo ID (e.g., 'owner/name'), fetch detector.onnx into models/
            if model_path and (('/' in model_path) and not os.path.exists(model_path)):
                try:
                    from huggingface_hub import hf_hub_download
                    os.makedirs(self.cache_dir, exist_ok=True)
                    logger.info(f"📥 Resolving repo '{model_path}' to detector.onnx in {self.cache_dir}...")
                    resolved = hf_hub_download(repo_id=model_path, filename='detector.onnx', cache_dir=self.cache_dir, local_dir=self.cache_dir, local_dir_use_symlinks=False)
                    if resolved and os.path.exists(resolved):
                        model_path = resolved
                        logger.info(f"✅ Downloaded detector.onnx to: {model_path}")
                except Exception as repo_err:
                    logger.error(f"Failed to download from repo '{model_path}': {repo_err}")
            if not os.path.exists(model_path):
                logger.error(f"Model file not found: {model_path}")
                return False
            
            # Check if it's the same model already loaded
            if self.model_loaded and not force_reload:
                last_path = self.config.get('last_model_path', '')
                if last_path == model_path:
                    logger.info("Model already loaded (same path)")
                    return True
                else:
                    logger.info(f"Model path changed from {last_path} to {model_path}, reloading...")
                    force_reload = True
            
            # Clear previous model if force reload
            if force_reload:
                logger.info("Force reloading model...")
                self.model = None
                self.onnx_session = None
                self.model_loaded = False
                self.model_type = None
            
            logger.info(f"📥 Loading bubble detection model: {model_path}")
            
            # Determine model type by extension
            ext = Path(model_path).suffix.lower()
            
            if ext in ['.pt', '.pth']:
                if not YOLO_AVAILABLE:
                    logger.warning("Ultralytics package not available in this build")
                    logger.info("Bubble detection will be disabled - this is normal for lightweight builds")
                    # Don't return False immediately, try other fallbacks
                    self.model_loaded = False
                    return False
                
                # Load YOLOv8 model
                try:
                    self.model = YOLO(model_path)
                    self.model_type = 'yolo'
                    
                    # Set to eval mode
                    if hasattr(self.model, 'model'):
                        self.model.model.eval()
                    
                    # Move to GPU if available
                    if self.use_gpu and TORCH_AVAILABLE:
                        try:
                            self.model.to('cuda')
                        except Exception as gpu_error:
                            logger.warning(f"Could not move model to GPU: {gpu_error}")
                            
                    logger.info("✅ YOLOv8 model loaded successfully")
                    # Apply optional FP16 precision to reduce VRAM if enabled
                    if self.quantize_enabled and self.use_gpu and TORCH_AVAILABLE:
                        try:
                            m = self.model.model if hasattr(self.model, 'model') else self.model
                            m.half()
                            logger.info("🔻 Applied FP16 precision to YOLO model (GPU)")
                        except Exception as _e:
                            logger.warning(f"Could not switch YOLO model to FP16: {_e}")
                    
                except Exception as yolo_error:
                    logger.error(f"Failed to load YOLO model: {yolo_error}")
                    return False
                    
            elif ext == '.onnx':
                if not ONNX_AVAILABLE:
                    logger.warning("ONNX Runtime not available in this build")
                    logger.info("ONNX model support disabled - this is normal for lightweight builds")
                    return False
                
                try:
                    # Load ONNX model
                    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.use_gpu else ['CPUExecutionProvider']
                    session_path = model_path
                    if self.quantize_enabled:
                        try:
                            from onnxruntime.quantization import quantize_dynamic, QuantType
                            quant_path = os.path.splitext(model_path)[0] + ".int8.onnx"
                            if not os.path.exists(quant_path) or os.environ.get('FORCE_ONNX_REBUILD', 'false').lower() == 'true':
                                logger.info("🔻 Quantizing ONNX model weights to INT8 (dynamic)...")
                                quantize_dynamic(model_input=model_path, model_output=quant_path, weight_type=QuantType.QInt8, op_types_to_quantize=['Conv', 'MatMul'])
                            session_path = quant_path
                            self.config['last_onnx_quantized_path'] = quant_path
                            self._save_config()
                            logger.info(f"✅ Using quantized ONNX model: {quant_path}")
                        except Exception as qe:
                            logger.warning(f"ONNX quantization not applied: {qe}")
                    # Use conservative ORT memory options to reduce RAM growth
                    so = ort.SessionOptions()
                    try:
                        so.enable_mem_pattern = False
                        so.enable_cpu_mem_arena = False
                    except Exception:
                        pass
                    self.onnx_session = ort.InferenceSession(session_path, sess_options=so, providers=providers)
                    self.model_type = 'onnx'
                    
                    logger.info("✅ ONNX model loaded successfully")
                    
                except Exception as onnx_error:
                    logger.error(f"Failed to load ONNX model: {onnx_error}")
                    return False
                    
            elif ext == '.torchscript':
                if not TORCH_AVAILABLE:
                    logger.warning("PyTorch not available in this build")
                    logger.info("TorchScript model support disabled - this is normal for lightweight builds")
                    return False
                
                try:
                    # Add safety check for torch being None
                    if torch is None:
                        logger.error("PyTorch module is None - cannot load TorchScript model")
                        return False
                        
                    # Load TorchScript model
                    self.model = torch.jit.load(model_path, map_location='cpu')
                    self.model.eval()
                    self.model_type = 'torch'
                    
                    if self.use_gpu:
                        try:
                            self.model = self.model.cuda()
                        except Exception as gpu_error:
                            logger.warning(f"Could not move TorchScript model to GPU: {gpu_error}")
                    
                    logger.info("✅ TorchScript model loaded successfully")
                    
                    # Optional FP16 precision on GPU
                    if self.quantize_enabled and self.use_gpu and TORCH_AVAILABLE:
                        try:
                            self.model = self.model.half()
                            logger.info("🔻 Applied FP16 precision to TorchScript model (GPU)")
                        except Exception as _e:
                            logger.warning(f"Could not switch TorchScript model to FP16: {_e}")
                    
                except Exception as torch_error:
                    logger.error(f"Failed to load TorchScript model: {torch_error}")
                    return False
                    
            else:
                logger.error(f"Unsupported model format: {ext}")
                logger.info("Supported formats: .pt/.pth (YOLOv8), .onnx (ONNX), .torchscript (TorchScript)")
                return False
            
            # Only set loaded if we actually succeeded
            self.model_loaded = True
            self.config['last_model_path'] = model_path
            self.config['model_type'] = self.model_type
            self._save_config()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            logger.error(traceback.format_exc())
            self.model_loaded = False
            
            # Provide helpful context for .exe users
            logger.info("Note: If running from .exe, some ML libraries may not be included")
            logger.info("This is normal for lightweight builds - bubble detection will be disabled")
            
            return False
    
    def load_rtdetr_model(self, model_path: str = None, model_id: str = None, force_reload: bool = False) -> bool:
        """
        Load RT-DETR model for advanced bubble and text detection.
        This implementation avoids the 'meta tensor' copy error by:
        - Serializing the entire load under a class lock (no concurrent loads)
        - Loading directly onto the target device (CUDA if available) via device_map='auto'
        - Avoiding .to() on a potentially-meta model; no device migration post-load
        
        Args:
            model_path: Optional path to local model
            model_id: Optional HuggingFace model ID (default: 'ogkalu/comic-text-and-bubble-detector')
            force_reload: Force reload even if already loaded
            
        Returns:
            True if successful, False otherwise
        """
        if not TRANSFORMERS_AVAILABLE:
            logger.error("Transformers library required for RT-DETR. Install with: pip install transformers")
            return False
        
        if not PIL_AVAILABLE:
            logger.error("PIL required for RT-DETR. Install with: pip install pillow")
            return False
        
        if self.rtdetr_loaded and not force_reload:
            logger.info("RT-DETR model already loaded")
            return True
        
        # Fast path: if shared already loaded and not forcing reload, attach
        if BubbleDetector._rtdetr_loaded and not force_reload:
            self.rtdetr_model = BubbleDetector._rtdetr_shared_model
            self.rtdetr_processor = BubbleDetector._rtdetr_shared_processor
            self.rtdetr_loaded = True
            logger.info("RT-DETR model attached from shared cache")
            return True
        
        # Serialize the ENTIRE loading sequence to avoid concurrent init issues
        with BubbleDetector._rtdetr_init_lock:
            try:
                # Re-check after acquiring lock
                if BubbleDetector._rtdetr_loaded and not force_reload:
                    self.rtdetr_model = BubbleDetector._rtdetr_shared_model
                    self.rtdetr_processor = BubbleDetector._rtdetr_shared_processor
                    self.rtdetr_loaded = True
                    logger.info("RT-DETR model attached from shared cache (post-lock)")
                    return True
                
                # Use custom model_id if provided, otherwise use default
                repo_id = model_id if model_id else self.rtdetr_repo
                logger.info(f"📥 Loading RT-DETR model from {repo_id}...")

                # Ensure TorchDynamo/compile doesn't interfere on some builds
                try:
                    os.environ.setdefault('TORCHDYNAMO_DISABLE', '1')
                except Exception:
                    pass
                
                # Decide device strategy
                gpu_available = bool(TORCH_AVAILABLE and hasattr(torch, 'cuda') and torch.cuda.is_available())
                device_map = 'auto' if gpu_available else None
                # Choose dtype
                dtype = None
                if TORCH_AVAILABLE:
                    try:
                        dtype = torch.float16 if gpu_available else torch.float32
                    except Exception:
                        dtype = None
                low_cpu = True if gpu_available else False
                
                # Load processor (once)
                self.rtdetr_processor = RTDetrImageProcessor.from_pretrained(
                    repo_id,
                    size={"width": 640, "height": 640},
                    cache_dir=self.cache_dir if not model_path else None
                )
                
                # Prepare kwargs for from_pretrained
                from_kwargs = {
                    'cache_dir': self.cache_dir if not model_path else None,
                    'low_cpu_mem_usage': low_cpu,
                    'device_map': device_map,
                }
                # Note: dtype is handled via torch_dtype parameter in newer transformers
                if dtype is not None:
                    from_kwargs['torch_dtype'] = dtype
                
                # First attempt: load directly to target (CUDA if available)
                try:
                        self.rtdetr_model = RTDetrV2ForObjectDetection.from_pretrained(
                        model_path if model_path else repo_id,
                        **from_kwargs,
                    )
                except Exception as primary_err:
                    # Fallback to a simple CPU load (no device move) if CUDA path fails
                    logger.warning(f"RT-DETR primary load failed ({primary_err}); retrying on CPU...")
                    from_kwargs_fallback = {
                        'cache_dir': self.cache_dir if not model_path else None,
                        'low_cpu_mem_usage': False,
                        'device_map': None,
                    }
                    if TORCH_AVAILABLE:
                        from_kwargs_fallback['torch_dtype'] = torch.float32
                    self.rtdetr_model = RTDetrForObjectDetection.from_pretrained(
                        model_path if model_path else repo_id,
                        **from_kwargs_fallback,
                    )
                
                # Optional dynamic quantization for linear layers (CPU only)
                if self.quantize_enabled and TORCH_AVAILABLE and (not gpu_available):
                    try:
                        try:
                            import torch.ao.quantization as tq
                            quantize_dynamic = tq.quantize_dynamic  # type: ignore
                        except Exception:
                            import torch.quantization as tq  # type: ignore
                            quantize_dynamic = tq.quantize_dynamic  # type: ignore
                        self.rtdetr_model = quantize_dynamic(self.rtdetr_model, {torch.nn.Linear}, dtype=torch.qint8)
                        logger.info("🔻 Applied dynamic INT8 quantization to RT-DETR linear layers (CPU)")
                    except Exception as qe:
                        logger.warning(f"RT-DETR dynamic quantization skipped: {qe}")
                
                # Finalize
                self.rtdetr_model.eval()

                # Sanity check: ensure no parameter is left on 'meta' device
                try:
                    for n, p in self.rtdetr_model.named_parameters():
                        dev = getattr(p, 'device', None)
                        if dev is not None and getattr(dev, 'type', '') == 'meta':
                            raise RuntimeError(f"Parameter {n} is on 'meta' device after load")
                except Exception as e:
                    logger.error(f"RT-DETR load sanity check failed: {e}")
                    self.rtdetr_loaded = False
                    return False

                # Publish shared cache
                BubbleDetector._rtdetr_shared_model = self.rtdetr_model
                BubbleDetector._rtdetr_shared_processor = self.rtdetr_processor
                BubbleDetector._rtdetr_loaded = True
                BubbleDetector._rtdetr_repo_id = repo_id

                self.rtdetr_loaded = True
                
                # Save the model ID that was used
                self.config['rtdetr_loaded'] = True
                self.config['rtdetr_model_id'] = repo_id
                self._save_config()
                
                loc = 'CUDA' if gpu_available else 'CPU'
                logger.info(f"✅ RT-DETR model loaded successfully ({loc})")
                logger.info("   Classes: Empty bubbles, Text bubbles, Free text")
                
                # Auto-convert to ONNX for RT-DETR only if explicitly enabled
                if os.environ.get('AUTO_CONVERT_RTDETR_ONNX', 'false').lower() == 'true':
                    onnx_path = os.path.join(self.cache_dir, 'rtdetr_comic.onnx')
                    if self.convert_to_onnx('rtdetr', onnx_path):
                        logger.info("🚀 RT-DETR converted to ONNX for faster inference")
                        # Store ONNX path for later use
                        self.config['rtdetr_onnx_path'] = onnx_path
                        self._save_config()
                        # Optionally quantize ONNX for reduced RAM
                        if self.onnx_quantize_enabled:
                            try:
                                from onnxruntime.quantization import quantize_dynamic, QuantType
                                quant_path = os.path.splitext(onnx_path)[0] + ".int8.onnx"
                                if not os.path.exists(quant_path) or os.environ.get('FORCE_ONNX_REBUILD', 'false').lower() == 'true':
                                    logger.info("🔻 Quantizing RT-DETR ONNX to INT8 (dynamic)...")
                                    quantize_dynamic(model_input=onnx_path, model_output=quant_path, weight_type=QuantType.QInt8, op_types_to_quantize=['Conv', 'MatMul'])
                                self.config['rtdetr_onnx_quantized_path'] = quant_path
                                self._save_config()
                                logger.info(f"✅ Quantized RT-DETR ONNX saved to: {quant_path}")
                            except Exception as qe:
                                logger.warning(f"ONNX quantization for RT-DETR skipped: {qe}")
                    else:
                        logger.info("ℹ️ Skipping RT-DETR ONNX export (converter not supported in current environment)")
                
                return True
            except Exception as e:
                logger.error(f"❌ Failed to load RT-DETR: {e}")
                self.rtdetr_loaded = False
                return False
 
    def check_rtdetr_available(self, model_id: str = None) -> bool:
        """
        Check if RT-DETR model is available (cached).
        
        Args:
            model_id: Optional HuggingFace model ID
            
        Returns:
            True if model is cached and available
        """
        try:
            from pathlib import Path
            
            # Use provided model_id or default
            repo_id = model_id if model_id else self.rtdetr_repo
            
            # Check HuggingFace cache
            cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
            model_id_formatted = repo_id.replace("/", "--")
            
            # Look for model folder
            model_folders = list(cache_dir.glob(f"models--{model_id_formatted}*"))
            
            if model_folders:
                for folder in model_folders:
                    if (folder / "snapshots").exists():
                        snapshots = list((folder / "snapshots").iterdir())
                        if snapshots:
                            return True
            
            return False
            
        except Exception:
            return False
        
    def detect_bubbles(self, 
                      image_path: str, 
                      confidence: float = None,
                      iou_threshold: float = None,
                      max_detections: int = None,
                      use_rtdetr: bool = None) -> List[Tuple[int, int, int, int]]:
        """
        Detect speech bubbles in an image (backward compatible method).
        
        Args:
            image_path: Path to image file
            confidence: Minimum confidence threshold (0-1)
            iou_threshold: IOU threshold for NMS (0-1)
            max_detections: Maximum number of detections to return
            use_rtdetr: If True, use RT-DETR instead of YOLOv8 (if available)
            
        Returns:
            List of bubble bounding boxes as (x, y, width, height) tuples
        """
        # Check for stop at start
        if self._check_stop():
            self._log("⏹️ Bubble detection stopped by user", "warning")
            return []
        
        # Decide which model to use
        if use_rtdetr is None:
            # Auto-select: prefer RT-DETR if available
            use_rtdetr = self.rtdetr_loaded
        
        if use_rtdetr:
            # Prefer ONNX backend if available, else PyTorch
            if getattr(self, 'rtdetr_onnx_loaded', False):
                results = self.detect_with_rtdetr_onnx(
                    image_path=image_path,
                    confidence=confidence,
                    return_all_bubbles=True
                )
                return results
            if self.rtdetr_loaded:
                results = self.detect_with_rtdetr(
                    image_path=image_path,
                    confidence=confidence,
                    return_all_bubbles=True
                )
                return results
        
        # Original YOLOv8 detection
        if not self.model_loaded:
            logger.error("No model loaded. Call load_model() first.")
            return []
        
        # Use defaults if not specified
        confidence = confidence or self.default_confidence
        iou_threshold = iou_threshold or self.default_iou_threshold
        max_detections = max_detections or self.default_max_detections
        
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Failed to load image: {image_path}")
                return []
            
            h, w = image.shape[:2]
            self._log(f"🔍 Detecting bubbles in {w}x{h} image")
            
            # Check for stop before inference
            if self._check_stop():
                self._log("⏹️ Bubble detection inference stopped by user", "warning")
                return []
            
            if self.model_type == 'yolo':
                # YOLOv8 inference
                results = self.model(
                    image_path,
                    conf=confidence,
                    iou=iou_threshold,
                    max_det=min(max_detections, getattr(self, 'max_det_yolo', max_detections)),
                    verbose=False
                )
                
                bubbles = []
                for r in results:
                    if r.boxes is not None:
                        for box in r.boxes:
                            # Get box coordinates
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            x, y = int(x1), int(y1)
                            width = int(x2 - x1)
                            height = int(y2 - y1)
                            
                            # Get confidence
                            conf = float(box.conf[0])
                            
                            # Add to list
                            if len(bubbles) < max_detections:
                                bubbles.append((x, y, width, height))
                            
                            logger.debug(f"   Bubble: ({x},{y}) {width}x{height} conf={conf:.2f}")
                
            elif self.model_type == 'onnx':
                # ONNX inference
                bubbles = self._detect_with_onnx(image, confidence, iou_threshold, max_detections)
                
            elif self.model_type == 'torch':
                # TorchScript inference
                bubbles = self._detect_with_torchscript(image, confidence, iou_threshold, max_detections)
            
            else:
                logger.error(f"Unknown model type: {self.model_type}")
                return []
            
            logger.info(f"✅ Detected {len(bubbles)} speech bubbles")
            time.sleep(0.1)  # Brief pause for stability
            logger.debug("💤 Bubble detection pausing briefly for stability")
            return bubbles
            
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            logger.error(traceback.format_exc())
            return []
    
    def detect_with_rtdetr(self,
                          image_path: str = None,
                          image: np.ndarray = None,
                          confidence: float = None,
                          return_all_bubbles: bool = False) -> Any:
        """
        Detect using RT-DETR model with 3-class detection (PyTorch backend).
        
        Args:
            image_path: Path to image file
            image: Image array (BGR format)
            confidence: Confidence threshold
            return_all_bubbles: If True, return list of bubble boxes (for compatibility)
                               If False, return dict with all classes
        
        Returns:
            List of bubbles if return_all_bubbles=True, else dict with classes
        """
        # Check for stop at start
        if self._check_stop():
            self._log("⏹️ RT-DETR detection stopped by user", "warning")
            if return_all_bubbles:
                return []
            return {'bubbles': [], 'text_bubbles': [], 'text_free': []}
        
        if not self.rtdetr_loaded:
            self._log("RT-DETR not loaded. Call load_rtdetr_model() first.", "warning")
            if return_all_bubbles:
                return []
            return {'bubbles': [], 'text_bubbles': [], 'text_free': []}
        
        confidence = confidence or self.default_confidence
        
        try:
            # Load image
            if image_path:
                image = cv2.imread(image_path)
            elif image is None:
                logger.error("No image provided")
                if return_all_bubbles:
                    return []
                return {'bubbles': [], 'text_bubbles': [], 'text_free': []}
            
            # Convert BGR to RGB for PIL
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            
            # Prepare image for model
            inputs = self.rtdetr_processor(images=pil_image, return_tensors="pt")
            
            if not TORCH_AVAILABLE or not torch:
                raise RuntimeError("PyTorch not available")

            # Ensure CUDA is properly initialized
            if self.use_gpu and torch.cuda.is_available():
                try:
                    # Clear CUDA cache first
                    torch.cuda.empty_cache()
                    device = torch.device('cuda')
                    
                    # Get model's current device
                    model_device = next(self.rtdetr_model.parameters()).device
                    
                    # If model isn't on GPU yet, move it
                    if model_device.type != 'cuda':
                        logger.info("Moving RT-DETR model to GPU...")
                        self.rtdetr_model = self.rtdetr_model.to(device)
                        
                    # Move inputs to GPU directly (already initialized)
                    for k, v in inputs.items():
                        if isinstance(v, torch.Tensor):
                            inputs[k] = v.to(device=device, dtype=self.rtdetr_model.dtype)
                    
                    logger.info("✓ Using GPU for RT-DETR detection")
                except Exception as e:
                    logger.error(f"GPU setup failed, falling back to CPU: {e}")
                    self.use_gpu = False  # Disable GPU for future calls
                    device = torch.device('cpu')
                    # Move everything to CPU
                    self.rtdetr_model = self.rtdetr_model.cpu()
                    for k, v in inputs.items():
                        if isinstance(v, torch.Tensor):
                            inputs[k] = v.cpu().float()
            else:
                device = torch.device('cpu')
                # Already on CPU, just normalize dtypes
                for k, v in inputs.items():
                    if isinstance(v, torch.Tensor):
                        inputs[k] = v.float()

            # Skip CUDA test - we already handled device setup

            # Model and inputs are already on the correct device from earlier setup

            logger.info(f"Using device: {device} (CUDA available: {torch.cuda.is_available()})")
            # Enforce precision policy from settings
            desired_dtype = self._resolve_precision_dtype(device)
            try:
                self.rtdetr_model = self.rtdetr_model.to(device=device, dtype=desired_dtype).eval()
            except Exception as _cast_err:
                # If dtype cast fails (e.g., bf16 unsupported), fallback to fp32
                logger.warning(f"Model dtype cast to {desired_dtype} failed: {_cast_err}; falling back to fp32")
                self.rtdetr_model = self.rtdetr_model.to(device=device, dtype=torch.float32).eval()
            
            # Use autocast for mixed precision on GPU
            # Run inference with proper CUDA memory management
            try:
                # Align floating inputs to model dtype; keep integer tensors as-is
                model_dtype = next(self.rtdetr_model.parameters()).dtype if TORCH_AVAILABLE and self.rtdetr_model is not None else None
                for k, v in inputs.items():
                    if isinstance(v, torch.Tensor):
                        if v.is_floating_point() and model_dtype is not None:
                            inputs[k] = v.to(device=device, dtype=model_dtype)
                        else:
                            inputs[k] = v.to(device)
                
                if device.type == "cuda":
                    # Clear cache before inference
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    # Check VRAM usage
                    self._check_vram_usage(getattr(self, 'device_id', device))
                    
                    # Use autocast matching model dtype when appropriate
                    use_amp = model_dtype in (getattr(torch, 'float16', None), getattr(torch, 'bfloat16', None))
                    if use_amp:
                        with torch.inference_mode(), torch.cuda.amp.autocast(dtype=model_dtype):
                            outputs = self.rtdetr_model(**inputs)
                            torch.cuda.synchronize()
                    else:
                        with torch.inference_mode():
                            outputs = self.rtdetr_model(**inputs)
                            torch.cuda.synchronize()
                else:
                    with torch.inference_mode():
                        outputs = self.rtdetr_model(**inputs)
            except Exception as e:
                logger.error(f"Inference failed on {device}: {e}")
                # If we're on GPU and hit an error, try one more time after cache clear
                if device.type == "cuda":
                    try:
                        logger.info("Retrying GPU inference after cache clear...")
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                        with torch.inference_mode(), torch.cuda.amp.autocast():
                            outputs = self.rtdetr_model(**inputs)
                            torch.cuda.synchronize()
                    except Exception as retry_err:
                        logger.error(f"GPU retry also failed: {retry_err}")
                        raise
                logger.warning(f"Autocast path failed ({e}); retrying on CPU")
                device = torch.device('cpu')
                self.rtdetr_model = self.rtdetr_model.to(device).eval()
                # Rebuild inputs on CPU FP32
                rebuilt = {}
                for k, v in inputs.items():
                    if isinstance(v, torch.Tensor):
                        rebuilt[k] = v.detach().cpu().float()
                    else:
                        rebuilt[k] = v
                inputs = rebuilt
                outputs = self.rtdetr_model(**inputs)
            else:
                outputs = self.rtdetr_model(**inputs)
            
            # Post-process results
            with torch.no_grad():
                target_sizes = torch.tensor([pil_image.size[::-1]], device=device) if TORCH_AVAILABLE else None
                
                results = self.rtdetr_processor.post_process_object_detection(
                    outputs,
                    target_sizes=target_sizes,
                    threshold=confidence
                )[0]
                if TORCH_AVAILABLE and device.type == "cuda":
                    target_sizes = target_sizes.to(device)
                else:
                    outputs = self.rtdetr_model(**inputs)
                
            # Brief pause for stability after inference
                time.sleep(0.1)
                logger.debug("💤 RT-DETR inference pausing briefly for stability")
            
            # Post-process results
            target_sizes = torch.tensor([pil_image.size[::-1]]) if TORCH_AVAILABLE else None
            if TORCH_AVAILABLE and device.type == "cuda":
                target_sizes = target_sizes.to(device)
            
            results = self.rtdetr_processor.post_process_object_detection(
                outputs,
                target_sizes=target_sizes,
                threshold=confidence
            )[0]
            
            # Apply per-detector cap if configured
            cap = getattr(self, 'max_det_rtdetr', self.default_max_detections)
            if cap and len(results['boxes']) > cap:
                # Keep top-scoring first
                scores = results['scores']
                top_idx = scores.topk(k=cap).indices if hasattr(scores, 'topk') else range(cap)
                results = {
                    'boxes': [results['boxes'][i] for i in top_idx],
                    'scores': [results['scores'][i] for i in top_idx],
                    'labels': [results['labels'][i] for i in top_idx]
                }
            
            logger.info(f"📊 RT-DETR found {len(results['boxes'])} detections above {confidence:.2f} confidence")

            # Apply NMS to remove duplicate detections
            # Group detections by class
            class_detections = {self.CLASS_BUBBLE: [], self.CLASS_TEXT_BUBBLE: [], self.CLASS_TEXT_FREE: []}
            
            for box, score, label in zip(results['boxes'], results['scores'], results['labels']):
                x1, y1, x2, y2 = map(float, box.tolist())
                label_id = label.item()
                if label_id in class_detections:
                    class_detections[label_id].append((x1, y1, x2, y2, float(score.item())))
            
            # Apply NMS per class to remove duplicates
            def compute_iou(box1, box2):
                """Compute IoU between two boxes (x1, y1, x2, y2)"""
                x1_1, y1_1, x2_1, y2_1 = box1[:4]
                x1_2, y1_2, x2_2, y2_2 = box2[:4]
                
                # Intersection
                x_left = max(x1_1, x1_2)
                y_top = max(y1_1, y1_2)
                x_right = min(x2_1, x2_2)
                y_bottom = min(y2_1, y2_2)
                
                if x_right < x_left or y_bottom < y_top:
                    return 0.0
                
                intersection = (x_right - x_left) * (y_bottom - y_top)
                
                # Union
                area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
                area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
                union = area1 + area2 - intersection
                
                return intersection / union if union > 0 else 0.0
            
            def apply_nms(boxes_with_scores, iou_threshold=0.45):
                """Apply Non-Maximum Suppression"""
                if not boxes_with_scores:
                    return []
                
                # Sort by score (descending)
                sorted_boxes = sorted(boxes_with_scores, key=lambda x: x[4], reverse=True)
                keep = []
                
                while sorted_boxes:
                    # Keep the box with highest score
                    current = sorted_boxes.pop(0)
                    keep.append(current)
                    
                    # Remove boxes with high IoU
                    sorted_boxes = [box for box in sorted_boxes if compute_iou(current, box) < iou_threshold]
                
                return keep
            
            # Apply NMS and organize by class
            detections = {
                'bubbles': [],       # Empty speech bubbles
                'text_bubbles': [],  # Bubbles with text
                'text_free': []      # Text without bubbles
            }
            
            for class_id, boxes_list in class_detections.items():
                nms_boxes = apply_nms(boxes_list, iou_threshold=self.default_iou_threshold)
                
                for x1, y1, x2, y2, scr in nms_boxes:
                    width = int(x2 - x1)
                    height = int(y2 - y1)
                    # Store as (x, y, width, height) to match YOLOv8 format
                    bbox = (int(x1), int(y1), width, height)
                    
                    if class_id == self.CLASS_BUBBLE:
                        detections['bubbles'].append(bbox)
                    elif class_id == self.CLASS_TEXT_BUBBLE:
                        detections['text_bubbles'].append(bbox)
                    elif class_id == self.CLASS_TEXT_FREE:
                        detections['text_free'].append(bbox)
                    
                    # Stop early if we hit the configured cap across all classes
                    total_count = len(detections['bubbles']) + len(detections['text_bubbles']) + len(detections['text_free'])
                    if total_count >= (self.config.get('manga_settings', {}).get('ocr', {}).get('bubble_max_detections', self.default_max_detections) if isinstance(self.config, dict) else self.default_max_detections):
                        break
            
            # Log results
            total = len(detections['bubbles']) + len(detections['text_bubbles']) + len(detections['text_free'])
            logger.info(f"✅ RT-DETR detected {total} objects:")
            logger.info(f"   - Empty bubbles: {len(detections['bubbles'])}")
            logger.info(f"   - Text bubbles: {len(detections['text_bubbles'])}")
            logger.info(f"   - Free text: {len(detections['text_free'])}")
            
            # Return format based on compatibility mode
            if return_all_bubbles:
                # Return all bubbles (empty + with text) for backward compatibility
                all_bubbles = detections['bubbles'] + detections['text_bubbles']
                return all_bubbles
            else:
                return detections
            
        except Exception as e:
            logger.error(f"RT-DETR detection failed: {e}")
            logger.error(traceback.format_exc())
            if return_all_bubbles:
                return []
            return {'bubbles': [], 'text_bubbles': [], 'text_free': []}
    
    def detect_all_text_regions(self, image_path: str = None, image: np.ndarray = None) -> List[Tuple[int, int, int, int]]:
        """
        Detect all text regions using RT-DETR (both in bubbles and free text).
        
        Returns:
            List of bounding boxes for all text regions
        """
        if not self.rtdetr_loaded:
            logger.warning("RT-DETR required for text detection")
            return []
        
        detections = self.detect_with_rtdetr(image_path=image_path, image=image, return_all_bubbles=False)
        
        # Combine text bubbles and free text
        all_text = detections['text_bubbles'] + detections['text_free']
        
        logger.info(f"📝 Found {len(all_text)} text regions total")
        return all_text
    
    def _detect_with_onnx(self, image: np.ndarray, confidence: float, 
                         iou_threshold: float, max_detections: int) -> List[Tuple[int, int, int, int]]:
        """Run detection using ONNX model."""
        # Preprocess image
        img_size = 640  # Standard YOLOv8 input size
        img_resized = cv2.resize(image, (img_size, img_size))
        img_norm = img_resized.astype(np.float32) / 255.0
        img_transposed = np.transpose(img_norm, (2, 0, 1))
        img_batch = np.expand_dims(img_transposed, axis=0)
        
        # Run inference
        input_name = self.onnx_session.get_inputs()[0].name
        outputs = self.onnx_session.run(None, {input_name: img_batch})
        
        # Process outputs (YOLOv8 format)
        predictions = outputs[0][0]  # Remove batch dimension
        
        # Filter by confidence and apply NMS
        bubbles = []
        boxes = []
        scores = []
        
        for pred in predictions.T:  # Transpose to get predictions per detection
            if len(pred) >= 5:
                x_center, y_center, width, height, obj_conf = pred[:5]
                
                if obj_conf >= confidence:
                    # Convert to corner coordinates
                    x1 = x_center - width / 2
                    y1 = y_center - height / 2
                    
                    # Scale to original image size
                    h, w = image.shape[:2]
                    x1 = int(x1 * w / img_size)
                    y1 = int(y1 * h / img_size)
                    width = int(width * w / img_size)
                    height = int(height * h / img_size)
                    
                    boxes.append([x1, y1, x1 + width, y1 + height])
                    scores.append(float(obj_conf))
        
        # Apply NMS
        if boxes:
            indices = cv2.dnn.NMSBoxes(boxes, scores, confidence, iou_threshold)
            if len(indices) > 0:
                indices = indices.flatten()[:max_detections]
                for i in indices:
                    x1, y1, x2, y2 = boxes[i]
                    bubbles.append((x1, y1, x2 - x1, y2 - y1))
        
        return bubbles
    
    def _detect_with_torchscript(self, image: np.ndarray, confidence: float,
                                 iou_threshold: float, max_detections: int) -> List[Tuple[int, int, int, int]]:
        """Run detection using TorchScript model."""
        # Similar to ONNX but using PyTorch tensors
        img_size = 640
        img_resized = cv2.resize(image, (img_size, img_size))
        img_norm = img_resized.astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0)
        
        if self.use_gpu:
            img_tensor = img_tensor.cuda()
        
        with torch.no_grad():
            outputs = self.model(img_tensor)
        
        # Process outputs similar to ONNX
        # Implementation depends on exact model output format
        # This is a placeholder - adjust based on your model
        return []
    
    def visualize_detections(self, image_path: str, bubbles: List[Tuple[int, int, int, int]] = None, 
                            output_path: str = None, use_rtdetr: bool = False) -> np.ndarray:
        """
        Visualize detected bubbles on the image.
        
        Args:
            image_path: Path to original image
            bubbles: List of bubble bounding boxes (if None, will detect)
            output_path: Optional path to save visualization
            use_rtdetr: Use RT-DETR for visualization with class colors
            
        Returns:
            Image with drawn bounding boxes
        """
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Failed to load image: {image_path}")
            return None
        
        vis_image = image.copy()
        
        if use_rtdetr and self.rtdetr_loaded:
            # RT-DETR visualization with different colors per class
            detections = self.detect_with_rtdetr(image_path=image_path, return_all_bubbles=False)
            
            # Colors for each class
            colors = {
                'bubbles': (0, 255, 0),       # Green for empty bubbles
                'text_bubbles': (255, 0, 0),  # Blue for text bubbles
                'text_free': (0, 0, 255)      # Red for free text
            }
            
            # Draw detections
            for class_name, bboxes in detections.items():
                color = colors[class_name]
                
                for i, (x, y, w, h) in enumerate(bboxes):
                    # Draw rectangle
                    cv2.rectangle(vis_image, (x, y), (x + w, y + h), color, 2)
                    
                    # Add label
                    label = f"{class_name.replace('_', ' ').title()} {i+1}"
                    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(vis_image, (x, y - label_size[1] - 4), 
                                (x + label_size[0], y), color, -1)
                    cv2.putText(vis_image, label, (x, y - 2), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        else:
            # Original YOLOv8 visualization
            if bubbles is None:
                bubbles = self.detect_bubbles(image_path)
            
            # Draw bounding boxes
            for i, (x, y, w, h) in enumerate(bubbles):
                # Draw rectangle
                color = (0, 255, 0)  # Green
                thickness = 2
                cv2.rectangle(vis_image, (x, y), (x + w, y + h), color, thickness)
                
                # Add label
                label = f"Bubble {i+1}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(vis_image, (x, y - label_size[1] - 4), (x + label_size[0], y), color, -1)
                cv2.putText(vis_image, label, (x, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Save if output path provided
        if output_path:
            cv2.imwrite(output_path, vis_image)
            logger.info(f"💾 Visualization saved to: {output_path}")
        
        return vis_image
    
    def convert_to_onnx(self, model_path: str, output_path: str = None) -> bool:
        """
        Convert a YOLOv8 or RT-DETR model to ONNX format.
        
        Args:
            model_path: Path to model file or 'rtdetr' for loaded RT-DETR
            output_path: Path for ONNX output (auto-generated if None)
            
        Returns:
            True if conversion successful, False otherwise
        """
        try:
            logger.info(f"🔄 Converting {model_path} to ONNX...")
            
            # Generate output path if not provided
            if output_path is None:
                if model_path == 'rtdetr' and self.rtdetr_loaded:
                    base_name = 'rtdetr_comic'
                else:
                    base_name = Path(model_path).stem
                output_path = os.path.join(self.cache_dir, f"{base_name}.onnx")
            
            # Check if already exists
            if os.path.exists(output_path) and not os.environ.get('FORCE_ONNX_REBUILD', 'false').lower() == 'true':
                logger.info(f"✅ ONNX model already exists: {output_path}")
                return True
            
            # Handle RT-DETR conversion
            if model_path == 'rtdetr' and self.rtdetr_loaded:
                if not TORCH_AVAILABLE:
                    logger.error("PyTorch required for RT-DETR ONNX conversion")
                    return False
                
                # RT-DETR specific conversion
                self.rtdetr_model.eval()
                
                # Create dummy input (pixel values): BxCxHxW
                dummy_input = torch.randn(1, 3, 640, 640)
                if self.device == 'cuda':
                    dummy_input = dummy_input.to('cuda')
                
                # Wrap the model to return only tensors (logits, pred_boxes)
                class _RTDetrExportWrapper(torch.nn.Module):
                    def __init__(self, mdl):
                        super().__init__()
                        self.mdl = mdl
                    def forward(self, images):
                        out = self.mdl(pixel_values=images)
                        # Handle dict/ModelOutput/tuple outputs
                        logits = None
                        boxes = None
                        try:
                            if isinstance(out, dict):
                                logits = out.get('logits', None)
                                boxes = out.get('pred_boxes', out.get('boxes', None))
                            else:
                                logits = getattr(out, 'logits', None)
                                boxes = getattr(out, 'pred_boxes', getattr(out, 'boxes', None))
                        except Exception:
                            pass
                        if (logits is None or boxes is None) and isinstance(out, (tuple, list)) and len(out) >= 2:
                            logits, boxes = out[0], out[1]
                        return logits, boxes
                
                wrapper = _RTDetrExportWrapper(self.rtdetr_model)
                if self.device == 'cuda':
                    wrapper = wrapper.to('cuda')
                
                # Try PyTorch 2.x dynamo_export first (more tolerant of newer aten ops)
                try:
                    success = False
                    try:
                        from torch.onnx import dynamo_export
                        try:
                            exp = dynamo_export(wrapper, dummy_input)
                        except TypeError:
                            # Older PyTorch dynamo_export may not support this calling convention
                            exp = dynamo_export(wrapper, dummy_input)
                        # exp may have save(); otherwise, it may expose model_proto
                        try:
                            exp.save(output_path)  # type: ignore
                            success = True
                        except Exception:
                            try:
                                import onnx as _onnx
                                _onnx.save(exp.model_proto, output_path)  # type: ignore
                                success = True
                            except Exception as _se:
                                logger.warning(f"dynamo_export produced model but could not save: {_se}")
                    except Exception as de:
                        logger.warning(f"dynamo_export failed; falling back to legacy exporter: {de}")
                    if success:
                        logger.info(f"✅ RT-DETR ONNX saved to: {output_path} (dynamo_export)")
                        return True
                except Exception as de2:
                    logger.warning(f"dynamo_export path error: {de2}")

                # Legacy exporter with opset fallback
                last_err = None
                for opset in [19, 18, 17, 16, 15, 14, 13]:
                    try:
                        torch.onnx.export(
                            wrapper,
                            dummy_input,
                            output_path,
                            export_params=True,
                            opset_version=opset,
                            do_constant_folding=True,
                            input_names=['pixel_values'],
                            output_names=['logits', 'boxes'],
                            dynamic_axes={
                                'pixel_values': {0: 'batch', 2: 'height', 3: 'width'},
                                'logits': {0: 'batch'},
                                'boxes': {0: 'batch'}
                            }
                        )
                        logger.info(f"✅ RT-DETR ONNX saved to: {output_path} (opset {opset})")
                        return True
                    except Exception as _e:
                        last_err = _e
                        try:
                            msg = str(_e)
                        except Exception:
                            msg = ''
                        logger.warning(f"RT-DETR ONNX export failed at opset {opset}: {msg}")
                        continue
                
                logger.error(f"All RT-DETR ONNX export attempts failed. Last error: {last_err}")
                return False
            
            # Handle YOLOv8 conversion - FIXED
            elif YOLO_AVAILABLE and os.path.exists(model_path):
                logger.info(f"Loading YOLOv8 model from: {model_path}")
                
                # Load model
                model = YOLO(model_path)
                
                # Export to ONNX - this returns the path to the exported model
                logger.info("Exporting to ONNX format...")
                exported_path = model.export(format='onnx', imgsz=640, simplify=True)
                
                # exported_path could be a string or Path object
                exported_path = str(exported_path) if exported_path else None
                
                if exported_path and os.path.exists(exported_path):
                    # Move to desired location if different
                    if exported_path != output_path:
                        import shutil
                        logger.info(f"Moving ONNX from {exported_path} to {output_path}")
                        shutil.move(exported_path, output_path)
                    
                    logger.info(f"✅ YOLOv8 ONNX saved to: {output_path}")
                    return True
                else:
                    # Fallback: check if it was created with expected name
                    expected_onnx = model_path.replace('.pt', '.onnx')
                    if os.path.exists(expected_onnx):
                        if expected_onnx != output_path:
                            import shutil
                            shutil.move(expected_onnx, output_path)
                        logger.info(f"✅ YOLOv8 ONNX saved to: {output_path}")
                        return True
                    else:
                        logger.error(f"ONNX export failed - no output file found")
                        return False
            
            else:
                logger.error(f"Cannot convert {model_path}: Model not found or dependencies missing")
                return False
                
        except Exception as e:
            logger.error(f"Conversion failed: {e}")
            # Avoid noisy full stack trace in production logs; return False gracefully
            return False
    
    def batch_detect(self, image_paths: List[str], **kwargs) -> Dict[str, List[Tuple[int, int, int, int]]]:
        """
        Detect bubbles in multiple images.
        
        Args:
            image_paths: List of image paths
            **kwargs: Detection parameters (confidence, iou_threshold, max_detections, use_rtdetr)
            
        Returns:
            Dictionary mapping image paths to bubble lists
        """
        results = {}
        
        for i, image_path in enumerate(image_paths):
            logger.info(f"Processing image {i+1}/{len(image_paths)}: {os.path.basename(image_path)}")
            bubbles = self.detect_bubbles(image_path, **kwargs)
            results[image_path] = bubbles
        
        return results
    
    def unload(self, release_shared: bool = False):
        """Release model resources held by this detector instance.
        Args:
            release_shared: If True, also clear class-level shared RT-DETR caches.
        """
        try:
            # Release instance-level models and sessions
            try:
                if getattr(self, 'onnx_session', None) is not None:
                    self.onnx_session = None
            except Exception:
                pass
            try:
                if getattr(self, 'rtdetr_onnx_session', None) is not None:
                    self.rtdetr_onnx_session = None
            except Exception:
                pass
            for attr in ['model', 'rtdetr_model', 'rtdetr_processor']:
                try:
                    if hasattr(self, attr):
                        setattr(self, attr, None)
                except Exception:
                    pass
            for flag in ['model_loaded', 'rtdetr_loaded', 'rtdetr_onnx_loaded']:
                try:
                    if hasattr(self, flag):
                        setattr(self, flag, False)
                except Exception:
                    pass

            # Optional: release shared caches
            if release_shared:
                try:
                    BubbleDetector._rtdetr_shared_model = None
                    BubbleDetector._rtdetr_shared_processor = None
                    BubbleDetector._rtdetr_loaded = False
                except Exception:
                    pass

            # Free CUDA cache and trigger GC
            try:
                if TORCH_AVAILABLE and torch is not None and torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            try:
                import gc
                gc.collect()
            except Exception:
                pass
        except Exception:
            # Best-effort only
            pass

    def get_bubble_masks(self, image_path: str, bubbles: List[Tuple[int, int, int, int]]) -> np.ndarray:
        """
        Create a mask image with bubble regions.
        
        Args:
            image_path: Path to original image
            bubbles: List of bubble bounding boxes
            
        Returns:
            Binary mask with bubble regions as white (255)
        """
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Fill bubble regions
        for x, y, bw, bh in bubbles:
            cv2.rectangle(mask, (x, y), (x + bw, y + bh), 255, -1)
        
        return mask
    
    def filter_bubbles_by_size(self, bubbles: List[Tuple[int, int, int, int]], 
                              min_area: int = 100, 
                              max_area: int = None) -> List[Tuple[int, int, int, int]]:
        """
        Filter bubbles by area.
        
        Args:
            bubbles: List of bubble bounding boxes
            min_area: Minimum area in pixels
            max_area: Maximum area in pixels (None for no limit)
            
        Returns:
            Filtered list of bubbles
        """
        filtered = []
        
        for x, y, w, h in bubbles:
            area = w * h
            if area >= min_area and (max_area is None or area <= max_area):
                filtered.append((x, y, w, h))
        
        return filtered
    
    def merge_overlapping_bubbles(self, bubbles: List[Tuple[int, int, int, int]], 
                                 overlap_threshold: float = 0.1) -> List[Tuple[int, int, int, int]]:
        """
        Merge overlapping bubble detections.
        
        Args:
            bubbles: List of bubble bounding boxes
            overlap_threshold: Minimum overlap ratio to merge
            
        Returns:
            Merged list of bubbles
        """
        if not bubbles:
            return []
        
        # Convert to numpy array for easier manipulation
        boxes = np.array([(x, y, x+w, y+h) for x, y, w, h in bubbles])
        
        merged = []
        used = set()
        
        for i, box1 in enumerate(boxes):
            if i in used:
                continue
            
            # Start with current box
            x1, y1, x2, y2 = box1
            
            # Check for overlaps with remaining boxes
            for j in range(i + 1, len(boxes)):
                if j in used:
                    continue
                
                box2 = boxes[j]
                
                # Calculate intersection
                ix1 = max(x1, box2[0])
                iy1 = max(y1, box2[1])
                ix2 = min(x2, box2[2])
                iy2 = min(y2, box2[3])
                
                if ix1 < ix2 and iy1 < iy2:
                    # Calculate overlap ratio
                    intersection = (ix2 - ix1) * (iy2 - iy1)
                    area1 = (x2 - x1) * (y2 - y1)
                    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
                    overlap = intersection / min(area1, area2)
                    
                    if overlap >= overlap_threshold:
                        # Merge boxes
                        x1 = min(x1, box2[0])
                        y1 = min(y1, box2[1])
                        x2 = max(x2, box2[2])
                        y2 = max(y2, box2[3])
                        used.add(j)
            
            merged.append((int(x1), int(y1), int(x2 - x1), int(y2 - y1)))
        
        return merged

    # ============================
    # RT-DETR (ONNX) BACKEND
    # ============================
    def load_rtdetr_onnx_model(self, model_id: str = None, force_reload: bool = False) -> bool:
        """
        Load RT-DETR ONNX model using onnxruntime. Downloads detector.onnx and config.json
        from the provided Hugging Face repo if not already cached.
        """
        if not ONNX_AVAILABLE:
            logger.error("ONNX Runtime not available for RT-DETR ONNX backend")
            return False
        try:
            # If singleton mode and already loaded, just attach shared session
            try:
                adv = (self.config or {}).get('manga_settings', {}).get('advanced', {}) if isinstance(self.config, dict) else {}
                singleton = bool(adv.get('use_singleton_models', True))
            except Exception:
                singleton = True
            if singleton and BubbleDetector._rtdetr_onnx_loaded and not force_reload and BubbleDetector._rtdetr_onnx_shared_session is not None:
                self.rtdetr_onnx_session = BubbleDetector._rtdetr_onnx_shared_session
                self.rtdetr_onnx_loaded = True
                return True

            repo = model_id or self.rtdetr_onnx_repo
            try:
                from huggingface_hub import hf_hub_download
            except Exception as e:
                logger.error(f"huggingface-hub required to fetch RT-DETR ONNX: {e}")
                return False

            # Ensure local models dir (use configured cache_dir directly: e.g., 'models')
            cache_dir = self.cache_dir
            os.makedirs(cache_dir, exist_ok=True)

            # Download files into models/ and avoid symlinks so the file is visible there
            try:
                _ = hf_hub_download(repo_id=repo, filename='config.json', cache_dir=cache_dir, local_dir=cache_dir, local_dir_use_symlinks=False)
            except Exception:
                pass
            onnx_fp = hf_hub_download(repo_id=repo, filename='detector.onnx', cache_dir=cache_dir, local_dir=cache_dir, local_dir_use_symlinks=False)
            BubbleDetector._rtdetr_onnx_model_path = onnx_fp

            # Pick providers: prefer CUDA if available; otherwise CPU. Do NOT use DML.
            providers = ['CPUExecutionProvider']
            try:
                avail = ort.get_available_providers() if ONNX_AVAILABLE else []
                if 'CUDAExecutionProvider' in avail:
                    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            except Exception:
                pass

            # Session options with reduced memory arena and optional thread limiting in singleton mode
            so = ort.SessionOptions()
            try:
                so.enable_mem_pattern = False
                so.enable_cpu_mem_arena = False
            except Exception:
                pass
            # If singleton models mode is enabled in config, limit ORT threading to reduce CPU spikes
            try:
                adv = (self.config or {}).get('manga_settings', {}).get('advanced', {}) if isinstance(self.config, dict) else {}
                if bool(adv.get('use_singleton_models', True)):
                    so.intra_op_num_threads = 1
                    so.inter_op_num_threads = 1
                    try:
                        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                    except Exception:
                        pass
                    try:
                        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
                    except Exception:
                        pass
            except Exception:
                pass

            # Create session (serialize creation in singleton mode to avoid device storms)
            if singleton:
                with BubbleDetector._rtdetr_onnx_init_lock:
                    # Re-check after acquiring lock
                    if BubbleDetector._rtdetr_onnx_loaded and BubbleDetector._rtdetr_onnx_shared_session is not None and not force_reload:
                        self.rtdetr_onnx_session = BubbleDetector._rtdetr_onnx_shared_session
                        self.rtdetr_onnx_loaded = True
                        return True
                    sess = ort.InferenceSession(onnx_fp, providers=providers, sess_options=so)
                    BubbleDetector._rtdetr_onnx_shared_session = sess
                    BubbleDetector._rtdetr_onnx_loaded = True
                    BubbleDetector._rtdetr_onnx_providers = providers
                    self.rtdetr_onnx_session = sess
                    self.rtdetr_onnx_loaded = True
            else:
                self.rtdetr_onnx_session = ort.InferenceSession(onnx_fp, providers=providers, sess_options=so)
                self.rtdetr_onnx_loaded = True
            logger.info("✅ RT-DETR (ONNX) model ready")
            return True
        except Exception as e:
            logger.error(f"Failed to load RT-DETR ONNX: {e}")
            self.rtdetr_onnx_session = None
            self.rtdetr_onnx_loaded = False
            return False

    def detect_with_rtdetr_onnx(self,
                                image_path: str = None,
                                image: np.ndarray = None,
                                confidence: float = 0.3,
                                return_all_bubbles: bool = False) -> Any:
        """Detect using RT-DETR ONNX backend.
        Returns bubbles list if return_all_bubbles else dict by classes similar to PyTorch path.
        """
        if not self.rtdetr_onnx_loaded or self.rtdetr_onnx_session is None:
            logger.warning("RT-DETR ONNX not loaded")
            return [] if return_all_bubbles else {'bubbles': [], 'text_bubbles': [], 'text_free': []}
        try:
            # Acquire image
            if image_path is not None:
                import cv2
                image = cv2.imread(image_path)
                if image is None:
                    raise RuntimeError(f"Failed to read image: {image_path}")
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                if image is None:
                    raise RuntimeError("No image provided")
                # Assume image is BGR np.ndarray if from OpenCV
                try:
                    import cv2
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                except Exception:
                    image_rgb = image

            # To PIL then resize 640x640 as in reference
            from PIL import Image as _PILImage
            pil_image = _PILImage.fromarray(image_rgb)
            im_resized = pil_image.resize((640, 640))
            arr = np.asarray(im_resized, dtype=np.float32) / 255.0
            arr = np.transpose(arr, (2, 0, 1))  # (3,H,W)
            im_data = arr[np.newaxis, ...]

            w, h = pil_image.size
            orig_size = np.array([[w, h]], dtype=np.int64)

            # Run with a concurrency guard to prevent device hangs and limit memory usage
            # Apply semaphore for ALL providers (not just DML) to control concurrency
            providers = BubbleDetector._rtdetr_onnx_providers or []
            def _do_run(session):
                return session.run(None, {
                    'images': im_data,
                    'orig_target_sizes': orig_size
                })
            
            # Always use semaphore to limit concurrent RT-DETR calls
            acquired = False
            try:
                BubbleDetector._rtdetr_onnx_sema.acquire()
                acquired = True
                
                # Special DML error handling
                if 'DmlExecutionProvider' in providers:
                    try:
                        outputs = _do_run(self.rtdetr_onnx_session)
                    except Exception as dml_err:
                        msg = str(dml_err)
                        if '887A0005' in msg or '887A0006' in msg or 'Dml' in msg:
                            # Rebuild CPU session and retry once
                            try:
                                base_path = BubbleDetector._rtdetr_onnx_model_path
                                if base_path:
                                    so = ort.SessionOptions()
                                    so.enable_mem_pattern = False
                                    so.enable_cpu_mem_arena = False
                                    cpu_providers = ['CPUExecutionProvider']
                                    # Serialize rebuild
                                    with BubbleDetector._rtdetr_onnx_init_lock:
                                        sess = ort.InferenceSession(base_path, providers=cpu_providers, sess_options=so)
                                        BubbleDetector._rtdetr_onnx_shared_session = sess
                                        BubbleDetector._rtdetr_onnx_providers = cpu_providers
                                        self.rtdetr_onnx_session = sess
                                    outputs = _do_run(self.rtdetr_onnx_session)
                                else:
                                    raise
                            except Exception:
                                raise
                        else:
                            raise
                else:
                    # Non-DML providers - just run directly
                    outputs = _do_run(self.rtdetr_onnx_session)
            finally:
                if acquired:
                    try:
                        BubbleDetector._rtdetr_onnx_sema.release()
                    except Exception:
                        pass

            # outputs expected: labels, boxes, scores
            labels, boxes, scores = outputs[:3]
            if labels.ndim == 2 and labels.shape[0] == 1:
                labels = labels[0]
            if scores.ndim == 2 and scores.shape[0] == 1:
                scores = scores[0]
            if boxes.ndim == 3 and boxes.shape[0] == 1:
                boxes = boxes[0]

            # Apply NMS to remove duplicate detections
            # Group detections by class and apply NMS per class
            class_detections = {self.CLASS_BUBBLE: [], self.CLASS_TEXT_BUBBLE: [], self.CLASS_TEXT_FREE: []}
            
            for lab, box, scr in zip(labels, boxes, scores):
                if float(scr) < float(confidence):
                    continue
                label_id = int(lab)
                if label_id in class_detections:
                    x1, y1, x2, y2 = map(float, box)
                    class_detections[label_id].append((x1, y1, x2, y2, float(scr)))
            
            # Apply NMS per class to remove duplicates
            def compute_iou(box1, box2):
                """Compute IoU between two boxes (x1, y1, x2, y2)"""
                x1_1, y1_1, x2_1, y2_1 = box1[:4]
                x1_2, y1_2, x2_2, y2_2 = box2[:4]
                
                # Intersection
                x_left = max(x1_1, x1_2)
                y_top = max(y1_1, y1_2)
                x_right = min(x2_1, x2_2)
                y_bottom = min(y2_1, y2_2)
                
                if x_right < x_left or y_bottom < y_top:
                    return 0.0
                
                intersection = (x_right - x_left) * (y_bottom - y_top)
                
                # Union
                area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
                area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
                union = area1 + area2 - intersection
                
                return intersection / union if union > 0 else 0.0
            
            def apply_nms(boxes_with_scores, iou_threshold=0.45):
                """Apply Non-Maximum Suppression"""
                if not boxes_with_scores:
                    return []
                
                # Sort by score (descending)
                sorted_boxes = sorted(boxes_with_scores, key=lambda x: x[4], reverse=True)
                keep = []
                
                while sorted_boxes:
                    # Keep the box with highest score
                    current = sorted_boxes.pop(0)
                    keep.append(current)
                    
                    # Remove boxes with high IoU
                    sorted_boxes = [box for box in sorted_boxes if compute_iou(current, box) < iou_threshold]
                
                return keep
            
            # Apply NMS and build final detections
            detections = {'bubbles': [], 'text_bubbles': [], 'text_free': []}
            bubbles_all = []
            
            for class_id, boxes_list in class_detections.items():
                nms_boxes = apply_nms(boxes_list, iou_threshold=self.default_iou_threshold)
                
                for x1, y1, x2, y2, scr in nms_boxes:
                    bbox = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
                    
                    if class_id == self.CLASS_BUBBLE:
                        detections['bubbles'].append(bbox)
                        bubbles_all.append(bbox)
                    elif class_id == self.CLASS_TEXT_BUBBLE:
                        detections['text_bubbles'].append(bbox)
                        bubbles_all.append(bbox)
                    elif class_id == self.CLASS_TEXT_FREE:
                        detections['text_free'].append(bbox)

            return bubbles_all if return_all_bubbles else detections
        except Exception as e:
            logger.error(f"RT-DETR ONNX detection failed: {e}")
            return [] if return_all_bubbles else {'bubbles': [], 'text_bubbles': [], 'text_free': []}


# Standalone utility functions
def download_model_from_huggingface(repo_id: str = "ogkalu/comic-speech-bubble-detector-yolov8m",
                                   filename: str = "comic-speech-bubble-detector-yolov8m.pt",
                                   cache_dir: str = "models") -> str:
    """
    Download model from Hugging Face Hub.
    
    Args:
        repo_id: Hugging Face repository ID
        filename: Model filename in the repository
        cache_dir: Local directory to cache the model
        
    Returns:
        Path to downloaded model file
    """
    try:
        from huggingface_hub import hf_hub_download
        
        os.makedirs(cache_dir, exist_ok=True)
        
        logger.info(f"📥 Downloading {filename} from {repo_id}...")
        
        model_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=cache_dir,
            local_dir=cache_dir
        )
        
        logger.info(f"✅ Model downloaded to: {model_path}")
        return model_path
        
    except ImportError:
        logger.error("huggingface-hub package required. Install with: pip install huggingface-hub")
        return None
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return None


def download_rtdetr_model(cache_dir: str = "models") -> bool:
    """
    Download RT-DETR model for advanced detection.
    
    Args:
        cache_dir: Directory to cache the model
        
    Returns:
        True if successful
    """
    if not TRANSFORMERS_AVAILABLE:
        logger.error("Transformers required. Install with: pip install transformers")
        return False
    
    try:
        logger.info("📥 Downloading RT-DETR model...")
        from transformers import RTDetrForObjectDetection, RTDetrImageProcessor
        
        # This will download and cache the model
        processor = RTDetrImageProcessor.from_pretrained(
            "ogkalu/comic-text-and-bubble-detector",
            cache_dir=cache_dir
        )
        model = RTDetrForObjectDetection.from_pretrained(
            "ogkalu/comic-text-and-bubble-detector",
            cache_dir=cache_dir
        )
        
        logger.info("✅ RT-DETR model downloaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return False


# Example usage and testing
if __name__ == "__main__":
    import sys
    
    # Create detector
    detector = BubbleDetector()
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "download":
            # Download model from Hugging Face
            model_path = download_model_from_huggingface()
            if model_path:
                print(f"YOLOv8 model downloaded to: {model_path}")
            
            # Also download RT-DETR
            if download_rtdetr_model():
                print("RT-DETR model downloaded")
        
        elif sys.argv[1] == "detect" and len(sys.argv) > 3:
            # Detect bubbles in an image
            model_path = sys.argv[2]
            image_path = sys.argv[3]
            
            # Load appropriate model
            if 'rtdetr' in model_path.lower():
                if detector.load_rtdetr_model():
                    # Use RT-DETR
                    results = detector.detect_with_rtdetr(image_path)
                    print(f"RT-DETR Detection:")
                    print(f"  Empty bubbles: {len(results['bubbles'])}")
                    print(f"  Text bubbles: {len(results['text_bubbles'])}")
                    print(f"  Free text: {len(results['text_free'])}")
            else:
                if detector.load_model(model_path):
                    bubbles = detector.detect_bubbles(image_path, confidence=0.5)
                    print(f"YOLOv8 detected {len(bubbles)} bubbles:")
                    for i, (x, y, w, h) in enumerate(bubbles):
                        print(f"  Bubble {i+1}: position=({x},{y}) size=({w}x{h})")
            
            # Optionally visualize
            if len(sys.argv) > 4:
                output_path = sys.argv[4]
                detector.visualize_detections(image_path, output_path=output_path, 
                                             use_rtdetr='rtdetr' in model_path.lower())
        
        elif sys.argv[1] == "test-both" and len(sys.argv) > 2:
            # Test both models
            image_path = sys.argv[2]
            
            # Load YOLOv8
            yolo_path = "models/comic-speech-bubble-detector-yolov8m.pt"
            if os.path.exists(yolo_path):
                detector.load_model(yolo_path)
                yolo_bubbles = detector.detect_bubbles(image_path, use_rtdetr=False)
                print(f"YOLOv8: {len(yolo_bubbles)} bubbles")
            
            # Load RT-DETR
            if detector.load_rtdetr_model():
                rtdetr_bubbles = detector.detect_bubbles(image_path, use_rtdetr=True)
                print(f"RT-DETR: {len(rtdetr_bubbles)} bubbles")
        
        else:
            print("Usage:")
            print("  python bubble_detector.py download")
            print("  python bubble_detector.py detect <model_path> <image_path> [output_path]")
            print("  python bubble_detector.py test-both <image_path>")
    
    else:
        print("Bubble Detector Module (YOLOv8 + RT-DETR)")
        print("Usage:")
        print("  python bubble_detector.py download")
        print("  python bubble_detector.py detect <model_path> <image_path> [output_path]")
        print("  python bubble_detector.py test-both <image_path>")
