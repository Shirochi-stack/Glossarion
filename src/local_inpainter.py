"""
Local inpainting implementation - COMPATIBLE VERSION WITH JIT SUPPORT
Maintains full backward compatibility while adding proper JIT model support
"""
import os
import sys
import json
import numpy as np
import cv2
from typing import Optional, List, Tuple, Dict, Any
import logging
import traceback
import re
import hashlib
import urllib.request
from pathlib import Path
import threading

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check if we're running in a frozen environment
IS_FROZEN = getattr(sys, 'frozen', False)
if IS_FROZEN:
    MEIPASS = sys._MEIPASS
    os.environ['TORCH_HOME'] = MEIPASS
    os.environ['TRANSFORMERS_CACHE'] = os.path.join(MEIPASS, 'transformers')
    os.environ['HF_HOME'] = os.path.join(MEIPASS, 'huggingface')
    logger.info(f"Running in frozen environment: {MEIPASS}")

# Environment variables for ONNX
ONNX_CACHE_DIR = os.environ.get('ONNX_CACHE_DIR', 'models')
AUTO_CONVERT_TO_ONNX = os.environ.get('AUTO_CONVERT_TO_ONNX', 'true').lower() == 'true'
SKIP_ONNX_FOR_CKPT = os.environ.get('SKIP_ONNX_FOR_CKPT', 'true').lower() == 'true'
FORCE_ONNX_REBUILD = os.environ.get('FORCE_ONNX_REBUILD', 'false').lower() == 'true'
CACHE_DIR = os.environ.get('MODEL_CACHE_DIR', os.path.expanduser('~/.cache/inpainting'))

# Modified import handling for frozen environment
TORCH_AVAILABLE = False
torch = None
nn = None
F = None
BaseModel = object

try:
    import onnxruntime_extensions
    ONNX_EXTENSIONS_AVAILABLE = True
except ImportError:
    ONNX_EXTENSIONS_AVAILABLE = False
    logger.info("ONNX Runtime Extensions not available - FFT models won't work in ONNX")

if IS_FROZEN:
    # In frozen environment, try harder to import
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        TORCH_AVAILABLE = True
        BaseModel = nn.Module
        logger.info("✓ PyTorch loaded in frozen environment")
    except Exception as e:
        logger.error(f"PyTorch not available in frozen environment: {e}")
        logger.error("❌ Inpainting disabled - PyTorch is required")
else:
    # Normal environment
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        TORCH_AVAILABLE = True
        BaseModel = nn.Module
    except ImportError:
        TORCH_AVAILABLE = False
        logger.error("PyTorch not available - inpainting disabled")

# Configure ORT memory behavior before importing
try:
    os.environ.setdefault('ORT_DISABLE_MEMORY_ARENA', '1')
except Exception:
    pass
# ONNX Runtime - usually works well in frozen environments
ONNX_AVAILABLE = False
try:
    import onnx
    import onnxruntime as ort
    ONNX_AVAILABLE = True
    logger.info("✓ ONNX Runtime available")
except ImportError:
    ONNX_AVAILABLE = False
    logger.warning("ONNX Runtime not available")

# Bubble detector - optional
BUBBLE_DETECTOR_AVAILABLE = False
try:
    from bubble_detector import BubbleDetector
    BUBBLE_DETECTOR_AVAILABLE = True
    logger.info("✓ Bubble detector available")
except ImportError:
    logger.info("Bubble detector not available - basic inpainting will be used")


# JIT Model URLs (for automatic download)
LAMA_JIT_MODELS = {
    'lama': {
        'url': 'https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt',
        'md5': 'e3aa4aaa15225a33ec84f9f4bc47e500',
        'name': 'BigLama'
    },
    'anime': {
        'url': 'https://github.com/Sanster/models/releases/download/AnimeMangaInpainting/anime-manga-big-lama.pt',
        'md5': '29f284f36a0a510bcacf39ecf4c4d54f',
        'name': 'Anime-Manga BigLama'
    },
    'lama_official': {
        'url': 'https://github.com/Sanster/models/releases/download/lama/lama.pt',
        'md5': '4b1a1de53b7a74e0ff9dd622834e8e1e',
        'name': 'LaMa Official'
    },
    'aot': {
        'url': 'https://huggingface.co/ogkalu/aot-inpainting-jit/resolve/main/aot_traced.pt',
        'md5': '5ecdac562c1d56267468fc4fbf80db27',
        'name': 'AOT GAN'
    },
    'lama_onnx': {
        'url': 'https://huggingface.co/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx',
        'md5': None,  # Add MD5 if you want to verify
        'name': 'LaMa ONNX (Carve)',
        'is_onnx': True  # Flag to indicate this is ONNX, not JIT
    }
}


def norm_img(img: np.ndarray) -> np.ndarray:
    """Normalize image to [0, 1] range"""
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    return img


def get_cache_path_by_url(url: str) -> str:
    """Get cache path for a model URL"""
    os.makedirs(CACHE_DIR, exist_ok=True)
    filename = os.path.basename(url)
    return os.path.join(CACHE_DIR, filename)


def download_model(url: str, md5: str = None) -> str:
    """Download model if not cached"""
    cache_path = get_cache_path_by_url(url)
    
    if os.path.exists(cache_path):
        logger.info(f"✅ Model already cached: {cache_path}")
        return cache_path
    
    logger.info(f"📥 Downloading model from {url}")
    
    try:
        urllib.request.urlretrieve(url, cache_path)
        logger.info(f"✅ Model downloaded to: {cache_path}")
        return cache_path
    except Exception as e:
        logger.error(f"❌ Download failed: {e}")
        if os.path.exists(cache_path):
            os.remove(cache_path)
        raise


class FFCInpaintModel(BaseModel):  # Use BaseModel instead of nn.Module
    """FFC model for LaMa inpainting - for checkpoint compatibility"""
    
    def __init__(self):
        if not TORCH_AVAILABLE:
            # Initialize as a simple object when PyTorch is not available
            super().__init__()
            logger.warning("PyTorch not available - FFCInpaintModel initialized as placeholder")
            self._pytorch_available = False
            return
            
        # Additional safety check for nn being None
        if nn is None:
            super().__init__()
            logger.error("Neural network modules not available - FFCInpaintModel disabled")
            self._pytorch_available = False
            return
            
        super().__init__()
        self._pytorch_available = True
        
        try:
            # Encoder
            self.model_1_ffc_convl2l = nn.Conv2d(4, 64, 7, padding=3)
            self.model_1_bn_l = nn.BatchNorm2d(64)
            
            self.model_2_ffc_convl2l = nn.Conv2d(64, 128, 3, padding=1)
            self.model_2_bn_l = nn.BatchNorm2d(128)
            
            self.model_3_ffc_convl2l = nn.Conv2d(128, 256, 3, padding=1)
            self.model_3_bn_l = nn.BatchNorm2d(256)
            
            self.model_4_ffc_convl2l = nn.Conv2d(256, 128, 3, padding=1)
            self.model_4_ffc_convl2g = nn.Conv2d(256, 384, 3, padding=1)
            self.model_4_bn_l = nn.BatchNorm2d(128)
            self.model_4_bn_g = nn.BatchNorm2d(384)
            
            # FFC blocks
            for i in range(5, 23):
                for conv_type in ['conv1', 'conv2']:
                    setattr(self, f'model_{i}_{conv_type}_ffc_convl2l', nn.Conv2d(128, 128, 3, padding=1))
                    setattr(self, f'model_{i}_{conv_type}_ffc_convl2g', nn.Conv2d(128, 384, 3, padding=1))
                    setattr(self, f'model_{i}_{conv_type}_ffc_convg2l', nn.Conv2d(384, 128, 3, padding=1))
                    setattr(self, f'model_{i}_{conv_type}_ffc_convg2g_conv1_0', nn.Conv2d(384, 192, 1))
                    setattr(self, f'model_{i}_{conv_type}_ffc_convg2g_conv1_1', nn.BatchNorm2d(192))
                    setattr(self, f'model_{i}_{conv_type}_ffc_convg2g_fu_conv_layer', nn.Conv2d(384, 384, 1))
                    setattr(self, f'model_{i}_{conv_type}_ffc_convg2g_fu_bn', nn.BatchNorm2d(384))
                    setattr(self, f'model_{i}_{conv_type}_ffc_convg2g_conv2', nn.Conv2d(192, 384, 1))
                    setattr(self, f'model_{i}_{conv_type}_bn_l', nn.BatchNorm2d(128))
                    setattr(self, f'model_{i}_{conv_type}_bn_g', nn.BatchNorm2d(384))
            
            # Decoder
            self.model_24 = nn.Conv2d(512, 256, 3, padding=1)
            self.model_25 = nn.BatchNorm2d(256)
            
            self.model_27 = nn.Conv2d(256, 128, 3, padding=1)
            self.model_28 = nn.BatchNorm2d(128)
            
            self.model_30 = nn.Conv2d(128, 64, 3, padding=1)
            self.model_31 = nn.BatchNorm2d(64)
            
            self.model_34 = nn.Conv2d(64, 3, 7, padding=3)
            
            # Activation functions
            self.relu = nn.ReLU(inplace=True)
            self.tanh = nn.Tanh()
            
            logger.info("FFCInpaintModel initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize FFCInpaintModel: {e}")
            self._pytorch_available = False
            raise
    
    def forward(self, image, mask):
        if not self._pytorch_available:
            logger.error("PyTorch not available for forward pass")
            raise RuntimeError("PyTorch not available for forward pass")
            
        if not TORCH_AVAILABLE or torch is None:
            logger.error("PyTorch not available for forward pass")
            raise RuntimeError("PyTorch not available for forward pass")
            
        try:
            x = torch.cat([image, mask], dim=1)
            
            x = self.relu(self.model_1_bn_l(self.model_1_ffc_convl2l(x)))
            x = self.relu(self.model_2_bn_l(self.model_2_ffc_convl2l(x)))
            x = self.relu(self.model_3_bn_l(self.model_3_ffc_convl2l(x)))
            
            x_l = self.relu(self.model_4_bn_l(self.model_4_ffc_convl2l(x)))
            x_g = self.relu(self.model_4_bn_g(self.model_4_ffc_convl2g(x)))
            
            for i in range(5, 23):
                identity_l, identity_g = x_l, x_g
                x_l, x_g = self._ffc_block(x_l, x_g, i, 'conv1')
                x_l, x_g = self._ffc_block(x_l, x_g, i, 'conv2')
                x_l = x_l + identity_l
                x_g = x_g + identity_g
            
            x = torch.cat([x_l, x_g], dim=1)
            x = self.relu(self.model_25(self.model_24(x)))
            x = self.relu(self.model_28(self.model_27(x)))
            x = self.relu(self.model_31(self.model_30(x)))
            x = self.tanh(self.model_34(x))
            
            mask_3ch = mask.repeat(1, 3, 1, 1)
            return x * mask_3ch + image * (1 - mask_3ch)
            
        except Exception as e:
            logger.error(f"Forward pass failed: {e}")
            raise RuntimeError(f"Forward pass failed: {e}")
    
    def _ffc_block(self, x_l, x_g, idx, conv_type):
        if not self._pytorch_available:
            raise RuntimeError("PyTorch not available for FFC block")
            
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available for FFC block")
            
        try:
            convl2l = getattr(self, f'model_{idx}_{conv_type}_ffc_convl2l')
            convl2g = getattr(self, f'model_{idx}_{conv_type}_ffc_convl2g')
            convg2l = getattr(self, f'model_{idx}_{conv_type}_ffc_convg2l')
            convg2g_conv1 = getattr(self, f'model_{idx}_{conv_type}_ffc_convg2g_conv1_0')
            convg2g_bn1 = getattr(self, f'model_{idx}_{conv_type}_ffc_convg2g_conv1_1')
            fu_conv = getattr(self, f'model_{idx}_{conv_type}_ffc_convg2g_fu_conv_layer')
            fu_bn = getattr(self, f'model_{idx}_{conv_type}_ffc_convg2g_fu_bn')
            convg2g_conv2 = getattr(self, f'model_{idx}_{conv_type}_ffc_convg2g_conv2')
            bn_l = getattr(self, f'model_{idx}_{conv_type}_bn_l')
            bn_g = getattr(self, f'model_{idx}_{conv_type}_bn_g')
            
            out_xl = convl2l(x_l) + convg2l(x_g)
            out_xg = convl2g(x_l) + convg2g_conv2(self.relu(convg2g_bn1(convg2g_conv1(x_g)))) + self.relu(fu_bn(fu_conv(x_g)))
            
            return self.relu(bn_l(out_xl)), self.relu(bn_g(out_xg))
            
        except Exception as e:
            logger.error(f"FFC block failed: {e}")
            raise RuntimeError(f"FFC block failed: {e}")


class LocalInpainter:
    """Local inpainter with full backward compatibility"""
    
    # MAINTAIN ORIGINAL SUPPORTED_METHODS for compatibility
    SUPPORTED_METHODS = {
        'lama': ('LaMa Inpainting', FFCInpaintModel),
        'mat': ('MAT Inpainting', FFCInpaintModel),
        'aot': ('AOT GAN Inpainting', FFCInpaintModel),
        'sd': ('Stable Diffusion Inpainting', FFCInpaintModel),
        'anime': ('Anime/Manga Inpainting', FFCInpaintModel),
        'lama_official': ('Official LaMa', FFCInpaintModel),
    }
    
    def __init__(self, config_path="config.json"):
        self.config_path = config_path
        self.config = self._load_config()
        self.model = None
        self.model_loaded = False
        self.current_method = None
        self.use_opencv_fallback = False  # FORCED DISABLED - No OpenCV fallback allowed
        self.onnx_session = None
        self.use_onnx = False
        self.is_jit_model = False
        self.pad_mod = 8
        
        # Default tiling settings - OFF by default for most models
        self.tiling_enabled = False
        self.tile_size = 512
        self.tile_overlap = 64
        
        # ONNX-specific settings
        self.onnx_model_loaded = False
        self.onnx_input_size = None  # Will be detected from model
        
        # Bubble detection
        self.bubble_detector = None
        self.bubble_model_loaded = False
        
        # Create directories
        os.makedirs(ONNX_CACHE_DIR, exist_ok=True)
        os.makedirs(CACHE_DIR, exist_ok=True)
        logger.info(f"📁 ONNX cache directory: {ONNX_CACHE_DIR}")
        logger.info(f"   Contents: {os.listdir(ONNX_CACHE_DIR) if os.path.exists(ONNX_CACHE_DIR) else 'Directory does not exist'}")

        
        # Check GPU availability safely
        self.use_gpu = False
        self.device = None
        
        if TORCH_AVAILABLE and torch is not None:
            try:
                self.use_gpu = torch.cuda.is_available()
                self.device = torch.device('cuda' if self.use_gpu else 'cpu')
                if self.use_gpu:
                    logger.info(f"🚀 GPU: {torch.cuda.get_device_name(0)}")
                else:
                    logger.info("💻 Using CPU")
            except AttributeError:
                # torch module exists but doesn't have cuda attribute
                self.use_gpu = False
                self.device = None
                logger.info("⚠️ PyTorch incomplete - inpainting disabled")
        else:
            logger.info("⚠️ PyTorch not available - inpainting disabled")
        
        # Quantization/precision toggle (off by default)
        try:
            adv_cfg = self.config.get('manga_settings', {}).get('advanced', {}) if isinstance(self.config, dict) else {}
            env_quant = os.environ.get('MODEL_QUANTIZE', 'false').lower() == 'true'
            self.quantize_enabled = bool(env_quant or adv_cfg.get('quantize_models', False))
            # ONNX quantization is now strictly opt-in (config or env), decoupled from general quantize_models
            self.onnx_quantize_enabled = bool(
                adv_cfg.get('onnx_quantize', os.environ.get('ONNX_QUANTIZE', 'false').lower() == 'true')
            )
            self.torch_precision = str(adv_cfg.get('torch_precision', os.environ.get('TORCH_PRECISION', 'auto'))).lower()
            logger.info(f"Quantization: {'ENABLED' if self.quantize_enabled else 'disabled'} for Local Inpainter; onnx_quantize={'on' if self.onnx_quantize_enabled else 'off'}; torch_precision={self.torch_precision}")
            self.int8_enabled = bool(
                adv_cfg.get('int8_quantize', False)
                or adv_cfg.get('quantize_int8', False)
                or os.environ.get('TORCH_INT8', 'false').lower() == 'true'
                or self.torch_precision in ('int8', 'int8_dynamic')
            )
            logger.info(
                f"Quantization: {'ENABLED' if self.quantize_enabled else 'disabled'} for Local Inpainter; "
                f"onnx_quantize={'on' if self.onnx_quantize_enabled else 'off'}; "
                f"torch_precision={self.torch_precision}; int8={'on' if self.int8_enabled else 'off'}"
            )
        except Exception:
            self.quantize_enabled = False
            self.onnx_quantize_enabled = False
            self.torch_precision = 'auto'
            self.int8_enabled = False
        
        # Stop flag support
        self.stop_flag = None
        self._stopped = False
        self.log_callback = None
        
        # Initialize bubble detector if available
        if BUBBLE_DETECTOR_AVAILABLE:
            try:
                self.bubble_detector = BubbleDetector()
                logger.info("🗨️ Bubble detection available")
            except:
                self.bubble_detector = None
                logger.info("🗨️ Bubble detection not available")
    
    def _load_config(self):
        try:
            if self.config_path and os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if not content:
                        return {}
                    try:
                        return json.loads(content)
                    except json.JSONDecodeError:
                        # Likely a concurrent write; retry once after a short delay
                        try:
                            import time
                            time.sleep(0.05)
                            with open(self.config_path, 'r', encoding='utf-8') as f2:
                                return json.load(f2)
                        except Exception:
                            return {}
        except Exception:
            return {}
        return {}
    
    def _save_config(self):
        # Don't save if config is empty (prevents purging)
        if not getattr(self, 'config', None):
            return
        try:
            # Load existing (best-effort)
            full_config = {}
            if self.config_path and os.path.exists(self.config_path):
                try:
                    with open(self.config_path, 'r', encoding='utf-8') as f:
                        full_config = json.load(f)
                except Exception:
                    full_config = {}
            # Update
            full_config.update(self.config)
            # Atomic write: write to temp then replace
            tmp_path = (self.config_path or 'config.json') + '.tmp'
            with open(tmp_path, 'w', encoding='utf-8') as f:
                json.dump(full_config, f, indent=2, ensure_ascii=False)
            try:
                os.replace(tmp_path, self.config_path or 'config.json')
            except Exception:
                # Fallback to direct write
                with open(self.config_path or 'config.json', 'w', encoding='utf-8') as f:
                    json.dump(full_config, f, indent=2, ensure_ascii=False)
        except Exception:
            # Never crash on config save
            pass
    
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
                "⏹️ Inpainting stopped",
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

    def convert_to_onnx(self, model_path: str, method: str) -> Optional[str]:
        """Convert a PyTorch model to ONNX format with FFT handling via custom operators"""
        if not ONNX_AVAILABLE:
            logger.warning("ONNX not available, skipping conversion")
            return None
        
        try:
            # Generate ONNX path
            model_name = os.path.basename(model_path).replace('.pt', '')
            onnx_path = os.path.join(ONNX_CACHE_DIR, f"{model_name}_{method}.onnx")
            
            # Check if ONNX already exists
            if os.path.exists(onnx_path) and not FORCE_ONNX_REBUILD:
                logger.info(f"✅ ONNX model already exists: {onnx_path}")
                return onnx_path
            
            logger.info(f"🔄 Converting {method} model to ONNX...")
            
            # The model should already be loaded at this point
            if not self.model_loaded or self.current_method != method:
                logger.error("Model not loaded for ONNX conversion")
                return None
            
            # Create dummy inputs
            dummy_image = torch.randn(1, 3, 512, 512).to(self.device)
            dummy_mask = torch.randn(1, 1, 512, 512).to(self.device)
            
            # For FFT models, we can't convert directly
            fft_models = ['lama', 'anime', 'lama_official']
            if method in fft_models:
                logger.warning(f"⚠️ {method.upper()} uses FFT operations that cannot be exported")
                return None  # Just return None, don't suggest Carve
            
            # Standard export for non-FFT models
            try:
                torch.onnx.export(
                    self.model,
                    (dummy_image, dummy_mask),
                    onnx_path,
                    export_params=True,
                    opset_version=13,
                    do_constant_folding=True,
                    input_names=['image', 'mask'],
                    output_names=['output'],
                    dynamic_axes={
                        'image': {0: 'batch', 2: 'height', 3: 'width'},
                        'mask': {0: 'batch', 2: 'height', 3: 'width'},
                        'output': {0: 'batch', 2: 'height', 3: 'width'}
                    }
                )
                logger.info(f"✅ ONNX model saved to: {onnx_path}")
                return onnx_path
                
            except torch.onnx.errors.UnsupportedOperatorError as e:
                logger.error(f"❌ Unsupported operator: {e}")
                return None
            
        except Exception as e:
            logger.error(f"❌ ONNX conversion failed: {e}")
            logger.error(traceback.format_exc())
            return None

    def load_onnx_model(self, onnx_path: str) -> bool:
        """Load an ONNX model with custom operator support"""
        if not ONNX_AVAILABLE:
            logger.error("ONNX Runtime not available")
            return False
        
        try:
            logger.info(f"📥 Loading ONNX model: {onnx_path}")
            
            # Store the path for later checking
            self.current_onnx_path = onnx_path

            # Check if this is a Carve model (fixed 512x512)
            is_carve_model = "lama_fp32" in onnx_path or "carve" in onnx_path.lower()
            if is_carve_model:
                logger.info("📦 Detected Carve ONNX model (fixed 512x512 input)")
                self.onnx_fixed_size = (512, 512)
            else:
                self.onnx_fixed_size = None
            
            # Standard ONNX loading, device-aware transformations for specific models
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.use_gpu else ['CPUExecutionProvider']
            session_path = onnx_path
            try:
                fname_lower = os.path.basename(onnx_path).lower()
            except Exception:
                fname_lower = str(onnx_path).lower()

            # Device-aware policy for LaMa-type ONNX (Carve or contains 'lama')
            is_lama_model = is_carve_model or ('lama' in fname_lower)
            if is_lama_model:
                base = os.path.splitext(onnx_path)[0]
                if self.use_gpu:
                    # Prefer FP16 on CUDA
                    fp16_path = base + '.fp16.onnx'
                    if (not os.path.exists(fp16_path)) or FORCE_ONNX_REBUILD:
                        try:
                            import onnx as _onnx
                            try:
                                from onnxruntime_tools.transformers.float16 import convert_float_to_float16 as _to_fp16
                            except Exception:
                                try:
                                    from onnxconverter_common import float16
                                    def _to_fp16(m, keep_io_types=True):
                                        return float16.convert_float_to_float16(m, keep_io_types=keep_io_types)
                                except Exception:
                                    _to_fp16 = None
                            if _to_fp16 is not None:
                                m = _onnx.load(onnx_path)
                                m_fp16 = _to_fp16(m, keep_io_types=True)
                                _onnx.save(m_fp16, fp16_path)
                                logger.info(f"✅ Generated FP16 ONNX for LaMa: {fp16_path}")
                        except Exception as e:
                            logger.warning(f"FP16 conversion for LaMa failed: {e}")
                    if os.path.exists(fp16_path):
                        session_path = fp16_path
                else:
                    # CPU path for LaMa: quantize only if enabled, and MatMul-only to avoid artifacts
                    if self.onnx_quantize_enabled:
                        try:
                            from onnxruntime.quantization import quantize_dynamic, QuantType
                            quant_path = base + '.matmul.int8.onnx'
                            if (not os.path.exists(quant_path)) or FORCE_ONNX_REBUILD:
                                logger.info("🔻 LaMa: Quantizing ONNX weights to INT8 (dynamic, ops=['MatMul'])...")
                                quantize_dynamic(
                                    model_input=onnx_path,
                                    model_output=quant_path,
                                    weight_type=QuantType.QInt8,
                                    op_types_to_quantize=['MatMul']
                                )
                            # Validate dynamic quant result
                            try:
                                import onnx as _onnx
                                _m_q = _onnx.load(quant_path)
                                _onnx.checker.check_model(_m_q)
                            except Exception as _qchk:
                                logger.warning(f"LaMa dynamic quant model invalid; deleting and falling back: {_qchk}")
                                try:
                                    os.remove(quant_path)
                                except Exception:
                                    pass
                                quant_path = None
                        except Exception as dy_err:
                            logger.warning(f"LaMa dynamic quantization failed: {dy_err}")
                            quant_path = None
                        # Fallback: static QDQ MatMul-only with zero data reader
                        if quant_path is None:
                            try:
                                import onnx as _onnx
                                from onnxruntime.quantization import (
                                    CalibrationDataReader, quantize_static,
                                    QuantFormat, QuantType, CalibrationMethod
                                )
                                m = _onnx.load(onnx_path)
                                shapes = {}
                                for inp in m.graph.input:
                                    dims = []
                                    for d in inp.type.tensor_type.shape.dim:
                                        dims.append(d.dim_value if d.dim_value > 0 else 1)
                                    shapes[inp.name] = dims
                                class _ZeroReader(CalibrationDataReader):
                                    def __init__(self, shapes):
                                        self.shapes = shapes
                                        self.done = False
                                    def get_next(self):
                                        if self.done:
                                            return None
                                        feed = {}
                                        for name, s in self.shapes.items():
                                            ss = list(s)
                                            if len(ss) == 4:
                                                if ss[2] <= 1: ss[2] = 512
                                                if ss[3] <= 1: ss[3] = 512
                                                if ss[1] <= 1 and 'mask' not in name.lower():
                                                    ss[1] = 3
                                            feed[name] = np.zeros(ss, dtype=np.float32)
                                        self.done = True
                                        return feed
                                dr = _ZeroReader(shapes)
                                quant_path = base + '.matmul.int8.onnx'
                                quantize_static(
                                    model_input=onnx_path,
                                    model_output=quant_path,
                                    calibration_data_reader=dr,
                                    quant_format=QuantFormat.QDQ,
                                    activation_type=QuantType.QUInt8,
                                    weight_type=QuantType.QInt8,
                                    per_channel=False,
                                    calibrate_method=CalibrationMethod.MinMax,
                                    op_types_to_quantize=['MatMul']
                                )
                                # Validate
                                try:
                                    _m_q = _onnx.load(quant_path)
                                    _onnx.checker.check_model(_m_q)
                                except Exception as _qchk2:
                                    logger.warning(f"LaMa static MatMul-only quant model invalid; deleting: {_qchk2}")
                                    try:
                                        os.remove(quant_path)
                                    except Exception:
                                        pass
                                    quant_path = None
                                else:
                                    logger.info(f"✅ Generated MatMul-only INT8 ONNX for LaMa: {quant_path}")
                            except Exception as st_err:
                                logger.warning(f"LaMa static MatMul-only quantization failed: {st_err}")
                                quant_path = None
                        # Use the quantized model if valid
                        if quant_path and os.path.exists(quant_path):
                            session_path = quant_path
                            logger.info(f"✅ Using LaMa quantized ONNX model: {quant_path}")
                    # If quantization not enabled or failed, session_path remains onnx_path (FP32)

            # Optional dynamic/static quantization for other models (opt-in)
            if (not is_lama_model) and self.onnx_quantize_enabled:
                base = os.path.splitext(onnx_path)[0]
                fname = os.path.basename(onnx_path).lower()
                is_aot = 'aot' in fname
                # For AOT: ignore any MatMul-only file and prefer Conv+MatMul
                if is_aot:
                    try:
                        ignored_matmul = base + ".matmul.int8.onnx"
                        if os.path.exists(ignored_matmul):
                            logger.info(f"⏭️ Ignoring MatMul-only quantized file for AOT: {ignored_matmul}")
                    except Exception:
                        pass
                # Choose target quant file and ops
                if is_aot:
                    quant_path = base + ".int8.onnx"
                    ops_to_quant = ['MatMul']
                    # Use MatMul-only for safer quantization across models
                    ops_for_static = ['MatMul']
                    # Try to simplify AOT graph prior to quantization
                    quant_input_path = onnx_path
                    try:
                        import onnx as _onnx
                        try:
                            from onnxsim import simplify as _onnx_simplify
                            _model = _onnx.load(onnx_path)
                            _sim_model, _check = _onnx_simplify(_model)
                            if _check:
                                sim_path = base + ".sim.onnx"
                                _onnx.save(_sim_model, sim_path)
                                quant_input_path = sim_path
                                logger.info(f"🧰 Simplified AOT ONNX before quantization: {sim_path}")
                        except Exception as _sim_err:
                            logger.info(f"AOT simplification skipped: {_sim_err}")
                        # No ONNX shape inference; keep original graph structure
                        # Ensure opset >= 13 for QDQ (axis attribute on DequantizeLinear)
                        try:
                            _m_tmp = _onnx.load(quant_input_path)
                            _opset = max([op.version for op in _m_tmp.opset_import]) if _m_tmp.opset_import else 11
                            if _opset < 13:
                                from onnx import version_converter as _vc
                                _m13 = _vc.convert_version(_m_tmp, 13)
                                up_path = base + ".op13.onnx"
                                _onnx.save(_m13, up_path)
                                quant_input_path = up_path
                                logger.info(f"🧰 Upgraded ONNX opset to 13 before QDQ quantization: {up_path}")
                        except Exception as _operr:
                            logger.info(f"Opset upgrade skipped: {_operr}")
                    except Exception:
                        quant_input_path = onnx_path
                else:
                    quant_path = base + ".matmul.int8.onnx"
                    ops_to_quant = ['MatMul']
                    ops_for_static = ops_to_quant
                    quant_input_path = onnx_path
                # Perform quantization if needed
                if not os.path.exists(quant_path) or FORCE_ONNX_REBUILD:
                    if is_aot:
                        # Directly perform static QDQ quantization for MatMul only (avoid Conv activations)
                        try:
                            import onnx as _onnx
                            from onnxruntime.quantization import CalibrationDataReader, quantize_static, QuantFormat, QuantType, CalibrationMethod
                            _model = _onnx.load(quant_input_path)
                            # Build input shapes from the model graph
                            input_shapes = {}
                            for inp in _model.graph.input:
                                dims = []
                                for d in inp.type.tensor_type.shape.dim:
                                    if d.dim_value > 0:
                                        dims.append(d.dim_value)
                                    else:
                                        # default fallback dimension
                                        dims.append(1)
                                input_shapes[inp.name] = dims
                            class _ZeroDataReader(CalibrationDataReader):
                                def __init__(self, input_shapes):
                                    self._shapes = input_shapes
                                    self._provided = False
                                def get_next(self):
                                    if self._provided:
                                        return None
                                    feed = {}
                                    for name, shape in self._shapes.items():
                                        # Ensure reasonable default spatial size
                                        s = list(shape)
                                        if len(s) == 4:
                                            if s[2] <= 1:
                                                s[2] = 512
                                            if s[3] <= 1:
                                                s[3] = 512
                                            # channel fallback
                                            if s[1] <= 1 and 'mask' not in name.lower():
                                                s[1] = 3
                                        feed[name] = (np.zeros(s, dtype=np.float32))
                                    self._provided = True
                                    return feed
                            dr = _ZeroDataReader(input_shapes)
                            quantize_static(
                                model_input=quant_input_path,
                                model_output=quant_path,
                                calibration_data_reader=dr,
                                quant_format=QuantFormat.QDQ,
                                activation_type=QuantType.QUInt8,
                                weight_type=QuantType.QInt8,
                                per_channel=True,
                                calibrate_method=CalibrationMethod.MinMax,
                                op_types_to_quantize=ops_for_static
                            )
                            # Validate quantized model to catch structural errors early
                            try:
                                _m_q = _onnx.load(quant_path)
                                _onnx.checker.check_model(_m_q)
                            except Exception as _qchk:
                                logger.warning(f"Quantized AOT model validation failed: {_qchk}")
                                # Remove broken quantized file to force fallback
                                try:
                                    os.remove(quant_path)
                                except Exception:
                                    pass
                            else:
                                logger.info(f"✅ Static INT8 quantization produced: {quant_path}")
                        except Exception as st_err:
                            logger.warning(f"Static ONNX quantization failed: {st_err}")
                    else:
                        # First attempt: dynamic quantization (MatMul)
                        try:
                            from onnxruntime.quantization import quantize_dynamic, QuantType
                            logger.info("🔻 Quantizing ONNX inpainting model weights to INT8 (dynamic, ops=['MatMul'])...")
                            quantize_dynamic(
                                model_input=quant_input_path,
                                model_output=quant_path,
                                weight_type=QuantType.QInt8,
                                op_types_to_quantize=['MatMul']
                            )
                        except Exception as dy_err:
                            logger.warning(f"Dynamic ONNX quantization failed: {dy_err}; attempting static quantization...")
                            # Fallback: static quantization with a zero data reader
                            try:
                                import onnx as _onnx
                                from onnxruntime.quantization import CalibrationDataReader, quantize_static, QuantFormat, QuantType, CalibrationMethod
                                _model = _onnx.load(quant_input_path)
                                # Build input shapes from the model graph
                                input_shapes = {}
                                for inp in _model.graph.input:
                                    dims = []
                                    for d in inp.type.tensor_type.shape.dim:
                                        if d.dim_value > 0:
                                            dims.append(d.dim_value)
                                        else:
                                            # default fallback dimension
                                            dims.append(1)
                                    input_shapes[inp.name] = dims
                                class _ZeroDataReader(CalibrationDataReader):
                                    def __init__(self, input_shapes):
                                        self._shapes = input_shapes
                                        self._provided = False
                                    def get_next(self):
                                        if self._provided:
                                            return None
                                        feed = {}
                                        for name, shape in self._shapes.items():
                                            # Ensure reasonable default spatial size
                                            s = list(shape)
                                            if len(s) == 4:
                                                if s[2] <= 1:
                                                    s[2] = 512
                                                if s[3] <= 1:
                                                    s[3] = 512
                                                # channel fallback
                                                if s[1] <= 1 and 'mask' not in name.lower():
                                                    s[1] = 3
                                            feed[name] = (np.zeros(s, dtype=np.float32))
                                        self._provided = True
                                        return feed
                                dr = _ZeroDataReader(input_shapes)
                                quantize_static(
                                    model_input=quant_input_path,
                                    model_output=quant_path,
                                    calibration_data_reader=dr,
                                    quant_format=QuantFormat.QDQ,
                                    activation_type=QuantType.QUInt8,
                                    weight_type=QuantType.QInt8,
                                    per_channel=True,
                                    calibrate_method=CalibrationMethod.MinMax,
                                    op_types_to_quantize=ops_for_static
                                )
                                # Validate quantized model to catch structural errors early
                                try:
                                    _m_q = _onnx.load(quant_path)
                                    _onnx.checker.check_model(_m_q)
                                except Exception as _qchk:
                                    logger.warning(f"Quantized AOT model validation failed: {_qchk}")
                                    # Remove broken quantized file to force fallback
                                    try:
                                        os.remove(quant_path)
                                    except Exception:
                                        pass
                                else:
                                    logger.info(f"✅ Static INT8 quantization produced: {quant_path}")
                            except Exception as st_err:
                                logger.warning(f"Static ONNX quantization failed: {st_err}")
                # Prefer the quantized file if it now exists
                if os.path.exists(quant_path):
                    # Validate existing quantized model before using it
                    try:
                        import onnx as _onnx
                        _m_q = _onnx.load(quant_path)
                        _onnx.checker.check_model(_m_q)
                    except Exception as _qchk:
                        logger.warning(f"Existing quantized ONNX invalid; deleting and falling back: {_qchk}")
                        try:
                            os.remove(quant_path)
                        except Exception:
                            pass
                    else:
                        session_path = quant_path
                        logger.info(f"✅ Using quantized ONNX model: {quant_path}")
                else:
                    logger.warning("ONNX quantization not applied: quantized file not created")
            # Use conservative ORT memory options to reduce RAM growth
            so = ort.SessionOptions()
            try:
                so.enable_mem_pattern = False
                so.enable_cpu_mem_arena = False
            except Exception:
                pass
            # Try to create an inference session, with graceful fallbacks
            try:
                self.onnx_session = ort.InferenceSession(session_path, sess_options=so, providers=providers)
            except Exception as e:
                err = str(e)
                logger.warning(f"ONNX session creation failed for {session_path}: {err}")
                # If quantized path failed due to unsupported ops or invalid graph, remove it and retry unquantized
                if session_path != onnx_path and ('ConvInteger' in err or 'NOT_IMPLEMENTED' in err or 'INVALID_ARGUMENT' in err):
                    try:
                        if os.path.exists(session_path):
                            os.remove(session_path)
                            logger.info(f"🧹 Deleted invalid quantized model: {session_path}")
                    except Exception:
                        pass
                    try:
                        logger.info("Retrying with unquantized ONNX model...")
                        self.onnx_session = ort.InferenceSession(onnx_path, sess_options=so, providers=providers)
                        session_path = onnx_path
                    except Exception as e2:
                        logger.warning(f"Unquantized ONNX session failed with current providers: {e2}")
                        # As a last resort, try CPU-only
                        try:
                            logger.info("Retrying ONNX on CPUExecutionProvider only...")
                            self.onnx_session = ort.InferenceSession(onnx_path, sess_options=so, providers=['CPUExecutionProvider'])
                            session_path = onnx_path
                            providers = ['CPUExecutionProvider']
                        except Exception as e3:
                            logger.error(f"Failed to create ONNX session on CPU: {e3}")
                            raise
                else:
                    # If we weren't quantized but failed on CUDA, try CPU-only once
                    if self.use_gpu and 'NOT_IMPLEMENTED' in err:
                        try:
                            logger.info("Retrying ONNX on CPUExecutionProvider only...")
                            self.onnx_session = ort.InferenceSession(session_path, sess_options=so, providers=['CPUExecutionProvider'])
                            providers = ['CPUExecutionProvider']
                        except Exception as e4:
                            logger.error(f"Failed to create ONNX session on CPU: {e4}")
                            raise
            
            # Get input/output names
            if self.onnx_session is None:
                raise RuntimeError("ONNX session was not created")
            self.onnx_input_names = [i.name for i in self.onnx_session.get_inputs()]
            self.onnx_output_names = [o.name for o in self.onnx_session.get_outputs()]
            
            # Check input shapes to detect fixed-size models
            input_shape = self.onnx_session.get_inputs()[0].shape
            if len(input_shape) == 4 and input_shape[2] == 512 and input_shape[3] == 512:
                self.onnx_fixed_size = (512, 512)
                logger.info(f"  Model expects fixed size: 512x512")
            
            logger.info(f"✅ ONNX model loaded")
            logger.info(f"  Inputs: {self.onnx_input_names}")
            logger.info(f"  Outputs: {self.onnx_output_names}")
            
            self.use_onnx = True
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load ONNX: {e}")
            self.use_onnx = False
            return False
        
    def _convert_checkpoint_key(self, key):
        """Convert checkpoint key format to model format"""
        # model.24.weight -> model_24.weight
        if re.match(r'^model\.(\d+)\.(weight|bias|running_mean|running_var)$', key):
            return re.sub(r'model\.(\d+)\.', r'model_\1.', key)
        
        # model.5.conv1.ffc.weight -> model_5_conv1_ffc.weight
        if key.startswith('model.'):
            parts = key.split('.')
            if parts[-1] in ['weight', 'bias', 'running_mean', 'running_var']:
                return '_'.join(parts[:-1]).replace('model_', 'model_') + '.' + parts[-1]
        
        return key.replace('.', '_')
    
    def _load_weights_with_mapping(self, model, state_dict):
        """Load weights with proper mapping"""
        model_dict = model.state_dict()
        
        logger.info(f"📊 Model expects {len(model_dict)} weights")
        logger.info(f"📊 Checkpoint has {len(state_dict)} weights")
        
        # Filter out num_batches_tracked
        actual_weights = {k: v for k, v in state_dict.items() if 'num_batches_tracked' not in k}
        logger.info(f"   Actual weights: {len(actual_weights)}")
        
        mapped = {}
        unmapped_ckpt = []
        unmapped_model = list(model_dict.keys())
        
        # Map checkpoint weights
        for ckpt_key, ckpt_val in actual_weights.items():
            success = False
            converted_key = self._convert_checkpoint_key(ckpt_key)
            
            if converted_key in model_dict:
                target_shape = model_dict[converted_key].shape
                
                if target_shape == ckpt_val.shape:
                    mapped[converted_key] = ckpt_val
                    success = True
                elif len(ckpt_val.shape) == 4 and len(target_shape) == 4:
                    # 4D permute for decoder convs
                    permuted = ckpt_val.permute(1, 0, 2, 3)
                    if target_shape == permuted.shape:
                        mapped[converted_key] = permuted
                        logger.info(f"   ✅ Permuted: {ckpt_key}")
                        success = True
                elif len(ckpt_val.shape) == 2 and len(target_shape) == 2:
                    # 2D transpose
                    transposed = ckpt_val.transpose(0, 1)
                    if target_shape == transposed.shape:
                        mapped[converted_key] = transposed
                        success = True
                
                if success and converted_key in unmapped_model:
                    unmapped_model.remove(converted_key)
            
            if not success:
                unmapped_ckpt.append(ckpt_key)
        
        # Try fallback mapping for unmapped
        if unmapped_ckpt:
            logger.info(f"   🔧 Fallback mapping for {len(unmapped_ckpt)} weights...")
            for ckpt_key in unmapped_ckpt[:]:
                ckpt_val = actual_weights[ckpt_key]
                for model_key in unmapped_model[:]:
                    if model_dict[model_key].shape == ckpt_val.shape:
                        if ('weight' in ckpt_key and 'weight' in model_key) or \
                           ('bias' in ckpt_key and 'bias' in model_key):
                            mapped[model_key] = ckpt_val
                            unmapped_model.remove(model_key)
                            unmapped_ckpt.remove(ckpt_key)
                            logger.info(f"   ✅ Mapped: {ckpt_key} -> {model_key}")
                            break
        
        # Initialize missing weights
        complete_dict = model_dict.copy()
        complete_dict.update(mapped)
        
        for key in unmapped_model:
            param = complete_dict[key]
            if 'weight' in key:
                if 'conv' in key.lower():
                    nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
                else:
                    nn.init.xavier_uniform_(param)
            elif 'bias' in key:
                nn.init.zeros_(param)
            elif 'running_mean' in key:
                nn.init.zeros_(param)
            elif 'running_var' in key:
                nn.init.ones_(param)
        
        # Report
        logger.info(f"✅ Mapped {len(actual_weights) - len(unmapped_ckpt)}/{len(actual_weights)} checkpoint weights")
        logger.info(f"   Filled {len(mapped)}/{len(model_dict)} model positions")
        
        if unmapped_model:
            pct = (len(unmapped_model) / len(model_dict)) * 100
            logger.info(f"   ⚠️ Initialized {len(unmapped_model)} missing weights ({pct:.1f}%)")
            if pct > 20:
                logger.warning("   ⚠️ May produce artifacts - checkpoint is incomplete")
                logger.warning("   💡 Consider downloading JIT model for better quality:")
                logger.warning(f"      inpainter.download_jit_model('{self.current_method or 'lama'}')")
        
        model.load_state_dict(complete_dict, strict=True)
        return True
    
    def download_jit_model(self, method: str) -> str:
        """Download JIT model for a method"""
        if method in LAMA_JIT_MODELS:
            model_info = LAMA_JIT_MODELS[method]
            logger.info(f"📥 Downloading {model_info['name']}...")
            
            try:
                model_path = download_model(model_info['url'], model_info['md5'])
                return model_path
            except Exception as e:
                logger.error(f"Failed to download {method}: {e}")
        else:
            logger.warning(f"No JIT model available for {method}")
        
        return None
    
    def load_model(self, method, model_path, force_reload=False):
        """Load model - supports both JIT and checkpoint files with ONNX conversion"""
        try:
            if not TORCH_AVAILABLE:
                logger.warning("PyTorch not available in this build")
                logger.info("Inpainting features will be disabled - this is normal for lightweight builds")
                logger.info("The application will continue to work without local inpainting")
                self.model_loaded = False
                return False
            
            # Additional safety check for torch being None
            if torch is None or nn is None:
                logger.warning("PyTorch modules not properly loaded")
                logger.info("Inpainting features will be disabled - this is normal for lightweight builds")
                self.model_loaded = False
                return False
            
            # Check if model path changed
            current_saved_path = self.config.get(f'{method}_model_path', '')
            if current_saved_path != model_path and current_saved_path:
                logger.info(f"📍 Model path changed for {method}")
                logger.info(f"   Old: {current_saved_path}")
                logger.info(f"   New: {model_path}")
                force_reload = True
            
            if not os.path.exists(model_path):
                # Try to auto-download JIT model if path doesn't exist
                logger.warning(f"Model not found: {model_path}")
                logger.info("Attempting to download JIT model...")
                
                try:
                    jit_path = self.download_jit_model(method)
                    if jit_path and os.path.exists(jit_path):
                        model_path = jit_path
                        logger.info(f"Using downloaded JIT model: {jit_path}")
                    else:
                        logger.error(f"Model not found and download failed: {model_path}")
                        logger.info("Inpainting will be unavailable for this session")
                        return False
                except Exception as download_error:
                    logger.error(f"Download failed: {download_error}")
                    logger.info("Inpainting will be unavailable for this session")
                    return False
            
            if self.model_loaded and self.current_method == method and not force_reload:
                logger.info(f"✅ {method.upper()} already loaded")
                return True
            
            # Clear previous model if force reload
            if force_reload:
                logger.info(f"🔄 Force reloading {method} model...")
                self.model = None
                self.onnx_session = None
                self.model_loaded = False
                self.is_jit_model = False
            
            logger.info(f"📥 Loading {method} from {model_path}")

            # Normalize path and enforce expected extension for certain methods
            try:
                _ext = os.path.splitext(model_path)[1].lower()
                _method_lower = str(method).lower()
                # For explicit ONNX methods, ensure we use a .onnx path
                if _method_lower in ("lama_onnx",) and _ext != ".onnx":
                    # If the file exists, try to detect if it's actually an ONNX model and correct the extension
                    if os.path.exists(model_path) and ONNX_AVAILABLE:
                        try:
                            import onnx as _onnx
                            _ = _onnx.load(model_path)  # will raise if not ONNX
                            # Build a corrected path under the ONNX cache dir
                            base_name = os.path.splitext(os.path.basename(model_path))[0]
                            if base_name.endswith('.pt'):
                                base_name = base_name[:-3]
                            corrected_path = os.path.join(ONNX_CACHE_DIR, base_name + ".onnx")
                            # Avoid overwriting a valid file with an invalid one
                            if model_path != corrected_path:
                                try:
                                    import shutil as _shutil
                                    _shutil.copy2(model_path, corrected_path)
                                    model_path = corrected_path
                                    logger.info(f"🔧 Corrected ONNX model extension/path: {model_path}")
                                except Exception as _cp_e:
                                    # As a fallback, try in-place rename to .onnx
                                    try:
                                        in_place = os.path.splitext(model_path)[0] + ".onnx"
                                        os.replace(model_path, in_place)
                                        model_path = in_place
                                        logger.info(f"🔧 Renamed ONNX model to: {model_path}")
                                    except Exception:
                                        logger.warning(f"Could not correct ONNX extension automatically: {_cp_e}")
                        except Exception:
                            # Not an ONNX file; leave as-is
                            pass
                    # If the path doesn't exist or still wrong, prefer the known ONNX download for this method
                    if (not os.path.exists(model_path)) or (os.path.splitext(model_path)[1].lower() != ".onnx"):
                        try:
                            _dl = self.download_jit_model("lama_onnx")
                            if _dl and os.path.exists(_dl):
                                model_path = _dl
                                logger.info(f"🔧 Using downloaded LaMa ONNX: {model_path}")
                        except Exception:
                            pass
            except Exception:
                pass

            # Check file signature to detect ONNX files with wrong extension
            with open(model_path, 'rb') as f:
                file_header = f.read(8)
            
            # Check for ONNX signature (starts with \x08)
            if file_header.startswith(b'\x08'):
                logger.info("📦 Detected ONNX file signature (regardless of extension)")
                if self.load_onnx_model(model_path):
                    self.model_loaded = True
                    self.current_method = method
                    self.use_onnx = True
                    self.is_jit_model = False
                    self.config[f'{method}_model_path'] = model_path
                    self._save_config()
                    logger.info(f"✅ {method.upper()} ONNX loaded!")
                    return True
                else:
                    logger.error("Failed to load ONNX model")
                    return False
  
            ext = model_path.lower().split('.')[-1]
            # Handle ONNX files directly
            if ext == 'onnx':
                logger.info("📥 Loading ONNX model directly...")
                if self.load_onnx_model(model_path):
                    self.model_loaded = True
                    self.current_method = method
                    self.use_onnx = True
                    self.is_jit_model = False
                    self.config[f'{method}_model_path'] = model_path
                    self._save_config()
                    logger.info(f"✅ {method.upper()} ONNX loaded!")
                    return True
                else:
                    logger.error("Failed to load ONNX model")
                    return False
            
            # Check if it's a JIT model (.pt) or checkpoint (.ckpt/.pth)
            if model_path.endswith('.pt'):
                try:
                    # Try loading as JIT/TorchScript
                    logger.info("📦 Attempting to load as JIT model...")
                    self.model = torch.jit.load(model_path, map_location=self.device or 'cpu')
                    self.model.eval()
                    
                    if self.use_gpu and self.device:
                        try:
                            self.model = self.model.to(self.device)
                        except Exception as gpu_error:
                            logger.warning(f"Could not move model to GPU: {gpu_error}")
                            logger.info("Using CPU instead")
                    
                    self.is_jit_model = True
                    self.model_loaded = True
                    self.current_method = method
                    logger.info("✅ JIT model loaded successfully!")
                    
                    # Optional FP16 precision on GPU to reduce VRAM
                    if self.quantize_enabled and self.use_gpu:
                        try:
                            if self.torch_precision in ('fp16', 'auto'):
                                self.model = self.model.half()
                                logger.info("🔻 Applied FP16 precision to inpainting model (GPU)")
                            else:
                                logger.info("Torch precision set to fp32; skipping half()")
                        except Exception as _e:
                            logger.warning(f"Could not switch inpainting model precision: {_e}")
                    
                    # Optional INT8 dynamic quantization for CPU TorchScript (best-effort)
                    if (self.int8_enabled or (self.quantize_enabled and not self.use_gpu and self.torch_precision in ('auto', 'int8'))) and not self.use_gpu:
                        try:
                            applied = False
                            # Try TorchScript dynamic quantization API (older PyTorch)
                            try:
                                from torch.quantization import quantize_dynamic_jit  # type: ignore
                                self.model = quantize_dynamic_jit(self.model, {"aten::linear"}, dtype=torch.qint8)  # type: ignore
                                applied = True
                            except Exception:
                                pass
                            # Try eager-style dynamic quantization on the scripted module (may no-op)
                            if not applied:
                                try:
                                    import torch.ao.quantization as tq  # type: ignore
                                    self.model = tq.quantize_dynamic(self.model, {nn.Linear}, dtype=torch.qint8)  # type: ignore
                                    applied = True
                                except Exception:
                                    pass
                            # Always try to optimize TorchScript for inference
                            try:
                                self.model = torch.jit.optimize_for_inference(self.model)  # type: ignore
                            except Exception:
                                pass
                            if applied:
                                logger.info("🔻 Applied INT8 dynamic quantization to JIT inpainting model (CPU)")
                            else:
                                logger.info("ℹ️ INT8 dynamic quantization not applied (unsupported for this JIT graph); using FP32 CPU")
                        except Exception as _qe:
                            logger.warning(f"INT8 quantization skipped: {_qe}")
                    
                    self.config[f'{method}_model_path'] = model_path
                    self._save_config()
                    
                    # ONNX CONVERSION (optionally in background)
                    if AUTO_CONVERT_TO_ONNX and self.model_loaded:
                        def _convert_and_switch():
                            try:
                                onnx_path = self.convert_to_onnx(model_path, method)
                                if onnx_path and self.load_onnx_model(onnx_path):
                                    logger.info("🚀 Using ONNX model for inference")
                                else:
                                    logger.info("📦 Using PyTorch JIT model for inference")
                            except Exception as onnx_error:
                                logger.warning(f"ONNX conversion failed: {onnx_error}")
                                logger.info("📦 Using PyTorch JIT model for inference")
                        
                        if os.environ.get('AUTO_CONVERT_TO_ONNX_BACKGROUND', 'true').lower() == 'true':
                            threading.Thread(target=_convert_and_switch, daemon=True).start()
                        else:
                            _convert_and_switch()
                    
                    return True
                except Exception as jit_error:
                    logger.info(f"   Not a JIT model, trying as regular checkpoint... ({jit_error})")
                    try:
                        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
                        self.is_jit_model = False
                    except Exception as load_error:
                        logger.error(f"Failed to load checkpoint: {load_error}")
                        return False
            else:
                # Load as regular checkpoint
                try:
                    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
                    self.is_jit_model = False
                except Exception as load_error:
                    logger.error(f"Failed to load checkpoint: {load_error}")
                    logger.info("This may happen if PyTorch is not fully available in the .exe build")
                    return False
            
            # If we get here, it's not JIT, so load as checkpoint
            if not self.is_jit_model:
                try:
                    # Try to create the model - this might fail if nn.Module is None
                    self.model = FFCInpaintModel()
                    
                    if isinstance(checkpoint, dict):
                        if 'gen_state_dict' in checkpoint:
                            state_dict = checkpoint['gen_state_dict']
                            logger.info("📦 Found gen_state_dict")
                        elif 'state_dict' in checkpoint:
                            state_dict = checkpoint['state_dict']
                        elif 'model' in checkpoint:
                            state_dict = checkpoint['model']
                        else:
                            state_dict = checkpoint
                    else:
                        state_dict = checkpoint
                    
                    self._load_weights_with_mapping(self.model, state_dict)
                    
                    self.model.eval()
                    if self.use_gpu and self.device:
                        try:
                            self.model = self.model.to(self.device)
                        except Exception as gpu_error:
                            logger.warning(f"Could not move model to GPU: {gpu_error}")
                            logger.info("Using CPU instead")
                    
                    # Optional INT8 dynamic quantization for CPU eager model
                    if (self.int8_enabled or (self.quantize_enabled and not self.use_gpu and self.torch_precision in ('auto', 'int8'))) and not self.use_gpu:
                        try:
                            import torch.ao.quantization as tq  # type: ignore
                            self.model = tq.quantize_dynamic(self.model, {nn.Linear}, dtype=torch.qint8)  # type: ignore
                            logger.info("🔻 Applied dynamic INT8 quantization to inpainting model (CPU)")
                        except Exception as qe:
                            logger.warning(f"INT8 dynamic quantization not applied: {qe}")
                            
                except Exception as model_error:
                    logger.error(f"Failed to create or initialize model: {model_error}")
                    logger.info("This may happen if PyTorch neural network modules are not available in the .exe build")
                    return False
            
            self.model_loaded = True
            self.current_method = method
            
            self.config[f'{method}_model_path'] = model_path
            self._save_config()
            
            logger.info(f"✅ {method.upper()} loaded!")
            
            # ONNX CONVERSION (optionally in background)
            if AUTO_CONVERT_TO_ONNX and model_path.endswith('.pt') and self.model_loaded:
                def _convert_and_switch():
                    try:
                        onnx_path = self.convert_to_onnx(model_path, method)
                        if onnx_path and self.load_onnx_model(onnx_path):
                            logger.info("🚀 Using ONNX model for inference")
                    except Exception as onnx_error:
                        logger.warning(f"ONNX conversion failed: {onnx_error}")
                        logger.info("📦 Continuing with PyTorch model")
                
                if os.environ.get('AUTO_CONVERT_TO_ONNX_BACKGROUND', 'true').lower() == 'true':
                    threading.Thread(target=_convert_and_switch, daemon=True).start()
                else:
                    _convert_and_switch()

            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            logger.error(traceback.format_exc())
            logger.info("Note: If running from .exe, some ML libraries may not be included")
            logger.info("This is normal for lightweight builds - inpainting will be disabled")
            self.model_loaded = False
            return False

    def unload(self):
        """Release all heavy resources held by this inpainter instance."""
        try:
            # Release ONNX session and metadata
            try:
                if self.onnx_session is not None:
                    self.onnx_session = None
            except Exception:
                pass
            for attr in ['onnx_input_names', 'onnx_output_names', 'current_onnx_path', 'onnx_fixed_size']:
                try:
                    if hasattr(self, attr):
                        setattr(self, attr, None)
                except Exception:
                    pass

            # Release PyTorch model
            try:
                if self.model is not None:
                    if TORCH_AVAILABLE and torch is not None:
                        try:
                            # Move to CPU then drop reference
                            self.model = self.model.to('cpu') if hasattr(self.model, 'to') else None
                        except Exception:
                            pass
                    self.model = None
            except Exception:
                pass

            # Drop bubble detector reference (not the global cache)
            try:
                self.bubble_detector = None
            except Exception:
                pass

            # Update flags
            self.model_loaded = False
            self.use_onnx = False
            self.is_jit_model = False

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
            # Never raise from unload
            pass
    
    def pad_img_to_modulo(self, img: np.ndarray, mod: int) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """Pad image to be divisible by mod"""
        if len(img.shape) == 2:
            height, width = img.shape
        else:
            height, width = img.shape[:2]
        
        pad_height = (mod - height % mod) % mod
        pad_width = (mod - width % mod) % mod
        
        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left
        
        if len(img.shape) == 2:
            padded = np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='reflect')
        else:
            padded = np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='reflect')
        
        return padded, (pad_top, pad_bottom, pad_left, pad_right)
    
    def remove_padding(self, img: np.ndarray, padding: Tuple[int, int, int, int]) -> np.ndarray:
        """Remove padding from image"""
        pad_top, pad_bottom, pad_left, pad_right = padding
        
        if len(img.shape) == 2:
            return img[pad_top:img.shape[0]-pad_bottom, pad_left:img.shape[1]-pad_right]
        else:
            return img[pad_top:img.shape[0]-pad_bottom, pad_left:img.shape[1]-pad_right, :]

    def _inpaint_tiled(self, image, mask, tile_size, overlap, refinement='normal'):
        """Process image in tiles"""
        orig_h, orig_w = image.shape[:2]
        result = image.copy()
        
        # Calculate tile positions
        for y in range(0, orig_h, tile_size - overlap):
            for x in range(0, orig_w, tile_size - overlap):
                # Calculate tile boundaries
                x_end = min(x + tile_size, orig_w)
                y_end = min(y + tile_size, orig_h)
                
                # Adjust start to ensure full tile size if possible
                if x_end - x < tile_size and x > 0:
                    x = max(0, x_end - tile_size)
                if y_end - y < tile_size and y > 0:
                    y = max(0, y_end - tile_size)
                
                # Extract tile
                tile_img = image[y:y_end, x:x_end]
                tile_mask = mask[y:y_end, x:x_end]
                
                # Skip if no inpainting needed
                if np.sum(tile_mask) == 0:
                    continue
                
                # Process this tile with the actual model
                processed_tile = self._process_single_tile(tile_img, tile_mask, tile_size, refinement)

                # Auto-retry for tile if no visible change
                try:
                    if self._is_noop(tile_img, processed_tile, tile_mask):
                        kernel = np.ones((3, 3), np.uint8)
                        expanded = cv2.dilate(tile_mask, kernel, iterations=1)
                        processed_retry = self._process_single_tile(tile_img, expanded, tile_size, 'fast')
                        if self._is_noop(tile_img, processed_retry, expanded):
                            logger.warning("Tile remained unchanged after retry; proceeding without further fallback")
                            processed_tile = processed_retry
                        else:
                            processed_tile = processed_retry
                except Exception as e:
                    logger.debug(f"Tiled no-op detection error: {e}")
                
                # Blend tile back into result
                if overlap > 0 and (x > 0 or y > 0):
                    result[y:y_end, x:x_end] = self._blend_tile(
                        result[y:y_end, x:x_end],
                        processed_tile,
                        x > 0,
                        y > 0,
                        overlap
                    )
                else:
                    result[y:y_end, x:x_end] = processed_tile
        
        logger.info(f"✅ Tiled inpainting complete ({orig_w}x{orig_h} in {tile_size}x{tile_size} tiles)")
        return result

    def _process_single_tile(self, tile_img, tile_mask, tile_size, refinement):
        """Process a single tile without tiling"""
        # Temporarily disable tiling
        old_tiling = self.tiling_enabled
        self.tiling_enabled = False
        result = self.inpaint(tile_img, tile_mask, refinement)
        self.tiling_enabled = old_tiling
        return result

    def _blend_tile(self, existing, new_tile, blend_x, blend_y, overlap):
        """Blend a tile with existing result"""
        if not blend_x and not blend_y:
            # No blending needed for first tile
            return new_tile
        
        h, w = new_tile.shape[:2]
        result = new_tile.copy()
        
        # Create blend weights
        if blend_x and overlap > 0 and w > overlap:
            # Horizontal blend on left edge
            for i in range(overlap):
                alpha = i / overlap
                result[:, i] = existing[:, i] * (1 - alpha) + new_tile[:, i] * alpha
        
        if blend_y and overlap > 0 and h > overlap:
            # Vertical blend on top edge
            for i in range(overlap):
                alpha = i / overlap
                result[i, :] = existing[i, :] * (1 - alpha) + new_tile[i, :] * alpha
        
        return result

    def _is_noop(self, original: np.ndarray, result: np.ndarray, mask: np.ndarray, threshold: float = 0.75) -> bool:
        """Return True if inpainting produced negligible change within the masked area."""
        try:
            if original is None or result is None:
                return True
            if original.shape != result.shape:
                return False
            # Normalize mask to single channel boolean
            if mask is None:
                return False
            if len(mask.shape) == 3:
                mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            else:
                mask_gray = mask
            m = mask_gray > 0
            if not np.any(m):
                return False
            # Fast path
            if np.array_equal(original, result):
                return True
            diff = cv2.absdiff(result, original)
            if len(diff.shape) == 3:
                diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            else:
                diff_gray = diff
            mean_diff = float(np.mean(diff_gray[m]))
            return mean_diff < threshold
        except Exception as e:
            logger.debug(f"No-op detection failed: {e}")
            return False

    def inpaint(self, image, mask, refinement='normal', _retry_attempt: int = 0):
        """Inpaint - compatible with JIT, checkpoint, and ONNX models"""
        # Check for stop at start
        if self._check_stop():
            self._log("⏹️ Inpainting stopped by user", "warning")
            return image
        
        if not self.model_loaded:
            self._log("No model loaded", "error")
            return image
        
        try:
            # Store original dimensions
            orig_h, orig_w = image.shape[:2]
            
            if len(mask.shape) == 3:
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            
            # Apply dilation for anime method
            if self.current_method == 'anime':
                kernel = np.ones((7, 7), np.uint8)
                mask = cv2.dilate(mask, kernel, iterations=1)

            # Use instance tiling settings for ALL models
            logger.info(f"🔍 Tiling check: enabled={self.tiling_enabled}, tile_size={self.tile_size}, image_size={orig_h}x{orig_w}")

            # If tiling is enabled and image is larger than tile size
            if self.tiling_enabled and (orig_h > self.tile_size or orig_w > self.tile_size):
                logger.info(f"🔲 Using tiled inpainting: {self.tile_size}x{self.tile_size} tiles with {self.tile_overlap}px overlap")
                return self._inpaint_tiled(image, mask, self.tile_size, self.tile_overlap, refinement)
                
            # ONNX inference path
            if self.use_onnx and self.onnx_session:
                logger.debug("Using ONNX inference")
                
                # CRITICAL: Convert BGR (OpenCV default) to RGB (ML model expected)
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Check if this is a Carve model
                is_carve_model = False
                if hasattr(self, 'current_onnx_path'):
                    is_carve_model = "lama_fp32" in self.current_onnx_path or "carve" in self.current_onnx_path.lower()
                
                # Handle fixed-size models (resize instead of padding)
                if hasattr(self, 'onnx_fixed_size') and self.onnx_fixed_size:
                    fixed_h, fixed_w = self.onnx_fixed_size
                    # Resize to fixed size
                    image_resized = cv2.resize(image_rgb, (fixed_w, fixed_h), interpolation=cv2.INTER_LANCZOS4)
                    mask_resized = cv2.resize(mask, (fixed_w, fixed_h), interpolation=cv2.INTER_NEAREST)
                    
                    # Prepare inputs based on model type
                    if is_carve_model:
                        # Carve model expects normalized input [0, 1] range
                        logger.debug("Using Carve model normalization [0, 1]")
                        img_np = image_resized.astype(np.float32) / 255.0
                        mask_np = mask_resized.astype(np.float32) / 255.0
                        mask_np = (mask_np > 0.5) * 1.0  # Binary mask
                    elif self.current_method == 'aot':
                        # AOT normalization: [-1, 1] range for image
                        logger.debug("Using AOT model normalization [-1, 1] for image, [0, 1] for mask")
                        img_np = (image_resized.astype(np.float32) / 127.5) - 1.0
                        mask_np = mask_resized.astype(np.float32) / 255.0
                        mask_np = (mask_np > 0.5) * 1.0  # Binary mask
                        img_np = img_np * (1 - mask_np[:, :, np.newaxis])  # Mask out regions
                    else:
                        # Standard LaMa normalization: [0, 1] range
                        logger.debug("Using standard LaMa normalization [0, 1]")
                        img_np = image_resized.astype(np.float32) / 255.0
                        mask_np = mask_resized.astype(np.float32) / 255.0
                        mask_np = (mask_np > 0) * 1.0
                    
                    # Convert to NCHW format
                    img_np = img_np.transpose(2, 0, 1)[np.newaxis, ...]
                    mask_np = mask_np[np.newaxis, np.newaxis, ...]
                    
                    # Run ONNX inference
                    ort_inputs = {
                        self.onnx_input_names[0]: img_np.astype(np.float32),
                        self.onnx_input_names[1]: mask_np.astype(np.float32)
                    }
                    
                    ort_outputs = self.onnx_session.run(self.onnx_output_names, ort_inputs)
                    output = ort_outputs[0]
                    
                    # Post-process output based on model type
                    if is_carve_model:
                        # CRITICAL: Carve model outputs values ALREADY in [0, 255] range!
                        # DO NOT multiply by 255 or apply any scaling
                        logger.debug("Carve model output is already in [0, 255] range")
                        raw_output = output[0].transpose(1, 2, 0)
                        logger.debug(f"Carve output stats: min={raw_output.min():.3f}, max={raw_output.max():.3f}, mean={raw_output.mean():.3f}")
                        result = raw_output  # Just transpose, no scaling
                    elif self.current_method == 'aot':
                        # AOT: [-1, 1] to [0, 255]
                        result = ((output[0].transpose(1, 2, 0) + 1.0) * 127.5)
                    else:
                        # Standard: [0, 1] to [0, 255]
                        result = output[0].transpose(1, 2, 0) * 255
                    
                    result = np.clip(np.round(result), 0, 255).astype(np.uint8)
                    # CRITICAL: Convert RGB (model output) back to BGR (OpenCV expected)
                    result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
                    
                    # Resize back to original size
                    result = cv2.resize(result, (orig_w, orig_h), interpolation=cv2.INTER_LANCZOS4)
                    
                else:
                    # Variable-size models (use padding)
                    image_padded, padding = self.pad_img_to_modulo(image_rgb, self.pad_mod)
                    mask_padded, _ = self.pad_img_to_modulo(mask, self.pad_mod)
                    
                    # Prepare inputs based on model type
                    if is_carve_model:
                        # Carve model normalization [0, 1]
                        logger.debug("Using Carve model normalization [0, 1]")
                        img_np = image_padded.astype(np.float32) / 255.0
                        mask_np = mask_padded.astype(np.float32) / 255.0
                        mask_np = (mask_np > 0.5) * 1.0
                    elif self.current_method == 'aot':
                        # AOT normalization: [-1, 1] for image
                        logger.debug("Using AOT model normalization [-1, 1] for image, [0, 1] for mask")
                        img_np = (image_padded.astype(np.float32) / 127.5) - 1.0
                        mask_np = mask_padded.astype(np.float32) / 255.0
                        mask_np = (mask_np > 0.5) * 1.0
                        img_np = img_np * (1 - mask_np[:, :, np.newaxis])  # Mask out regions
                    else:
                        # Standard LaMa normalization: [0, 1]
                        logger.debug("Using standard LaMa normalization [0, 1]")
                        img_np = image_padded.astype(np.float32) / 255.0
                        mask_np = mask_padded.astype(np.float32) / 255.0
                        mask_np = (mask_np > 0) * 1.0
                    
                    # Convert to NCHW format
                    img_np = img_np.transpose(2, 0, 1)[np.newaxis, ...]
                    mask_np = mask_np[np.newaxis, np.newaxis, ...]
                    
                    # Check for stop before inference
                    if self._check_stop():
                        self._log("⏹️ ONNX inference stopped by user", "warning")
                        return image
                    
                    # Run ONNX inference
                    ort_inputs = {
                        self.onnx_input_names[0]: img_np.astype(np.float32),
                        self.onnx_input_names[1]: mask_np.astype(np.float32)
                    }
                    
                    ort_outputs = self.onnx_session.run(self.onnx_output_names, ort_inputs)
                    output = ort_outputs[0]
                    
                    # Post-process output
                    if is_carve_model:
                        # CRITICAL: Carve model outputs values ALREADY in [0, 255] range!
                        logger.debug("Carve model output is already in [0, 255] range")
                        raw_output = output[0].transpose(1, 2, 0)
                        logger.debug(f"Carve output stats: min={raw_output.min():.3f}, max={raw_output.max():.3f}, mean={raw_output.mean():.3f}")
                        result = raw_output  # Just transpose, no scaling
                    elif self.current_method == 'aot':
                        result = ((output[0].transpose(1, 2, 0) + 1.0) * 127.5)
                    else:
                        result = output[0].transpose(1, 2, 0) * 255
                    
                    result = np.clip(np.round(result), 0, 255).astype(np.uint8)
                    # CRITICAL: Convert RGB (model output) back to BGR (OpenCV expected)
                    result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
                    
                    # Remove padding
                    result = self.remove_padding(result, padding)    
                    
            elif self.is_jit_model:
                # JIT model processing
                if self.current_method == 'aot':
                    # Special handling for AOT model
                    logger.debug("Using AOT-specific preprocessing")
                    
                    # CRITICAL: Convert BGR (OpenCV) to RGB (AOT model expected)
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    # Pad images to be divisible by mod
                    image_padded, padding = self.pad_img_to_modulo(image_rgb, self.pad_mod)
                    mask_padded, _ = self.pad_img_to_modulo(mask, self.pad_mod)
                    
                    # AOT normalization: [-1, 1] range
                    img_torch = torch.from_numpy(image_padded).permute(2, 0, 1).unsqueeze_(0).float() / 127.5 - 1.0
                    mask_torch = torch.from_numpy(mask_padded).unsqueeze_(0).unsqueeze_(0).float() / 255.0
                    
                    # Binarize mask for AOT
                    mask_torch[mask_torch < 0.5] = 0
                    mask_torch[mask_torch >= 0.5] = 1
                    
                    # Move to device
                    img_torch = img_torch.to(self.device)
                    mask_torch = mask_torch.to(self.device)
                    
                    # Optional FP16 on GPU for lower VRAM
                    if self.quantize_enabled and self.use_gpu:
                        try:
                            if self.torch_precision == 'fp16' or self.torch_precision == 'auto':
                                img_torch = img_torch.half()
                                mask_torch = mask_torch.half()
                        except Exception:
                            pass
                    
                    # CRITICAL FOR AOT: Apply mask to input image
                    img_torch = img_torch * (1 - mask_torch)
                    
                    logger.debug(f"AOT Image shape: {img_torch.shape}, Mask shape: {mask_torch.shape}")
                    
                    # Run inference
                    with torch.no_grad():
                        inpainted = self.model(img_torch, mask_torch)
                    
                    # Post-process AOT output: denormalize from [-1, 1] to [0, 255]
                    result = ((inpainted.cpu().squeeze_(0).permute(1, 2, 0).numpy() + 1.0) * 127.5)
                    result = np.clip(np.round(result), 0, 255).astype(np.uint8)
                    
                    # CRITICAL: Convert RGB (model output) back to BGR (OpenCV expected)
                    result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
                    
                    # Remove padding
                    result = self.remove_padding(result, padding)
                    
                else:
                    # LaMa/Anime model processing
                    logger.debug(f"Using standard processing for {self.current_method}")
                    
                    # CRITICAL: Convert BGR (OpenCV) to RGB (LaMa/JIT models expected)
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    # Pad images to be divisible by mod
                    image_padded, padding = self.pad_img_to_modulo(image_rgb, self.pad_mod)
                    mask_padded, _ = self.pad_img_to_modulo(mask, self.pad_mod)
                    
                    # CRITICAL: Normalize to [0, 1] range for LaMa models
                    image_norm = image_padded.astype(np.float32) / 255.0
                    mask_norm = mask_padded.astype(np.float32) / 255.0
                    
                    # Binary mask (values > 0 become 1)
                    mask_binary = (mask_norm > 0) * 1.0
                    
                    # Convert to PyTorch tensors with correct shape
                    # Image should be [B, C, H, W]  
                    image_tensor = torch.from_numpy(image_norm).permute(2, 0, 1).unsqueeze(0).float()
                    mask_tensor = torch.from_numpy(mask_binary).unsqueeze(0).unsqueeze(0).float()
                    
                    # Move to device
                    image_tensor = image_tensor.to(self.device)
                    mask_tensor = mask_tensor.to(self.device)
                    
                    # Optional FP16 on GPU for lower VRAM
                    if self.quantize_enabled and self.use_gpu:
                        try:
                            if self.torch_precision == 'fp16' or self.torch_precision == 'auto':
                                image_tensor = image_tensor.half()
                                mask_tensor = mask_tensor.half()
                        except Exception:
                            pass
                    
                    # Debug shapes
                    logger.debug(f"Image tensor shape: {image_tensor.shape}")  # Should be [1, 3, H, W]
                    logger.debug(f"Mask tensor shape: {mask_tensor.shape}")    # Should be [1, 1, H, W]
                    
                    # Ensure spatial dimensions match
                    if image_tensor.shape[2:] != mask_tensor.shape[2:]:
                        logger.warning(f"Spatial dimension mismatch: image {image_tensor.shape[2:]}, mask {mask_tensor.shape[2:]}")
                        # Resize mask to match image
                        mask_tensor = F.interpolate(mask_tensor, size=image_tensor.shape[2:], mode='nearest')
                    
                    # Run inference with proper error handling
                    with torch.no_grad():
                        try:
                            # Standard LaMa JIT models expect (image, mask)
                            inpainted = self.model(image_tensor, mask_tensor)
                        except RuntimeError as e:
                            error_str = str(e)
                            logger.error(f"Model inference failed: {error_str}")
                            
                            # If tensor size mismatch, log detailed info
                            if "size of tensor" in error_str.lower():
                                logger.error(f"Image shape: {image_tensor.shape}")
                                logger.error(f"Mask shape: {mask_tensor.shape}")
                                
                                # Try transposing if needed
                                if "dimension 3" in error_str and "880" in error_str:
                                    # This suggests the tensors might be in wrong format
                                    # Try different permutation
                                    logger.info("Attempting to fix tensor format...")
                                    
                                    # Ensure image is [B, C, H, W] not [B, H, W, C]
                                    if image_tensor.shape[1] > 3:
                                        image_tensor = image_tensor.permute(0, 3, 1, 2)
                                        logger.info(f"Permuted image to: {image_tensor.shape}")
                                    
                                    # Try again
                                    inpainted = self.model(image_tensor, mask_tensor)
                                else:
                                    # As last resort, try swapped arguments
                                    logger.info("Trying swapped arguments (mask, image)...")
                                    inpainted = self.model(mask_tensor, image_tensor)
                            else:
                                raise e
                    
                    # Process output
                    # Output should be [B, C, H, W]
                    if len(inpainted.shape) == 4:
                        # Remove batch dimension and permute to [H, W, C]
                        result = inpainted[0].permute(1, 2, 0).detach().cpu().numpy()
                    else:
                        # Handle unexpected output shape
                        result = inpainted.detach().cpu().numpy()
                        if len(result.shape) == 3 and result.shape[0] == 3:
                            result = result.transpose(1, 2, 0)
                    
                    # Denormalize to 0-255 range
                    result = np.clip(result * 255, 0, 255).astype(np.uint8)
                    
                    # CRITICAL: Convert RGB (model output) back to BGR (OpenCV expected)
                    result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
                    
                    # Remove padding
                    result = self.remove_padding(result, padding)
            
            else:
                # Original checkpoint model processing (keep as is)
                h, w = image.shape[:2]
                size = 768 if self.current_method == 'anime' else 512
                
                img_resized = cv2.resize(image, (size, size), interpolation=cv2.INTER_LANCZOS4)
                mask_resized = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
                
                img_norm = img_resized.astype(np.float32) / 127.5 - 1
                mask_norm = mask_resized.astype(np.float32) / 255.0
                
                img_tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).float()
                mask_tensor = torch.from_numpy(mask_norm).unsqueeze(0).unsqueeze(0).float()
                
                if self.use_gpu and self.device:
                    img_tensor = img_tensor.to(self.device)
                    mask_tensor = mask_tensor.to(self.device)
                
                with torch.no_grad():
                    output = self.model(img_tensor, mask_tensor)
                
                result = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
                result = ((result + 1) * 127.5).clip(0, 255).astype(np.uint8)
                result = cv2.resize(result, (w, h), interpolation=cv2.INTER_LANCZOS4)
            
            # Ensure result matches original size exactly
            if result.shape[:2] != (orig_h, orig_w):
                result = cv2.resize(result, (orig_w, orig_h), interpolation=cv2.INTER_LANCZOS4)
            
            # Apply refinement blending if requested
            if refinement != 'fast':
                # Ensure mask is same size as result
                if mask.shape[:2] != (orig_h, orig_w):
                    mask = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
                
                mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0
                kernel = cv2.getGaussianKernel(21, 5)
                kernel = kernel @ kernel.T
                mask_blur = cv2.filter2D(mask_3ch, -1, kernel)
                result = (result * mask_blur + image * (1 - mask_blur)).astype(np.uint8)
            
            # No-op detection and auto-retry
            try:
                if self._is_noop(image, result, mask):
                    if _retry_attempt == 0:
                        logger.warning("⚠️ Inpainting produced no visible change; retrying with slight mask dilation and fast refinement")
                        kernel = np.ones((3, 3), np.uint8)
                        expanded_mask = cv2.dilate(mask, kernel, iterations=1)
                        return self.inpaint(image, expanded_mask, refinement='fast', _retry_attempt=1)
                    elif _retry_attempt == 1:
                        logger.warning("⚠️ Still no visible change after retry; attempting a second dilation and fast refinement")
                        kernel = np.ones((5, 5), np.uint8)
                        expanded_mask2 = cv2.dilate(mask, kernel, iterations=1)
                        return self.inpaint(image, expanded_mask2, refinement='fast', _retry_attempt=2)
                    else:
                        logger.warning("⚠️ No further retries; returning last result without fallback")
            except Exception as e:
                logger.debug(f"No-op detection step failed: {e}")
            
            logger.info("✅ Inpainted successfully!")
            return result
            
        except Exception as e:
            logger.error(f"❌ Inpainting failed: {e}")
            logger.error(traceback.format_exc())
            
            # Return original image on failure
            logger.warning("Returning original image due to error")
            return image
    
    def inpaint_with_prompt(self, image, mask, prompt=None):
        """Compatibility method"""
        return self.inpaint(image, mask)
    
    def batch_inpaint(self, images, masks):
        """Batch inpainting"""
        return [self.inpaint(img, mask) for img, mask in zip(images, masks)]
    
    def load_bubble_model(self, model_path: str) -> bool:
        """Load bubble detection model"""
        if not BUBBLE_DETECTOR_AVAILABLE:
            logger.warning("Bubble detector not available")
            return False
        
        if self.bubble_detector is None:
            self.bubble_detector = BubbleDetector()
        
        if self.bubble_detector.load_model(model_path):
            self.bubble_model_loaded = True
            self.config['bubble_model_path'] = model_path
            self._save_config()
            logger.info("✅ Bubble detection model loaded")
            return True
        
        return False
    
    def detect_bubbles(self, image_path: str, confidence: float = 0.5) -> List[Tuple[int, int, int, int]]:
        """Detect speech bubbles in image"""
        if not self.bubble_model_loaded or self.bubble_detector is None:
            logger.warning("No bubble model loaded")
            return []
        
        return self.bubble_detector.detect_bubbles(image_path, confidence=confidence)
    
    def create_bubble_mask(self, image: np.ndarray, bubbles: List[Tuple[int, int, int, int]], 
                          expand_pixels: int = 5) -> np.ndarray:
        """Create mask from detected bubbles"""
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        for x, y, bw, bh in bubbles:
            x1 = max(0, x - expand_pixels)
            y1 = max(0, y - expand_pixels)
            x2 = min(w, x + bw + expand_pixels)
            y2 = min(h, y + bh + expand_pixels)
            
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
        
        return mask
    
    def inpaint_with_bubble_detection(self, image_path: str, confidence: float = 0.5,
                                     expand_pixels: int = 5, refinement: str = 'normal') -> np.ndarray:
        """Inpaint using automatic bubble detection"""
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Failed to load image: {image_path}")
            return None
        
        bubbles = self.detect_bubbles(image_path, confidence)
        if not bubbles:
            logger.warning("No bubbles detected")
            return image
        
        logger.info(f"Detected {len(bubbles)} bubbles")
        
        mask = self.create_bubble_mask(image, bubbles, expand_pixels)
        result = self.inpaint(image, mask, refinement)
        
        return result
    
    def batch_inpaint_with_bubbles(self, image_paths: List[str], **kwargs) -> List[np.ndarray]:
        """Batch inpaint multiple images with bubble detection"""
        results = []
        
        for i, image_path in enumerate(image_paths):
            logger.info(f"Processing image {i+1}/{len(image_paths)}")
            result = self.inpaint_with_bubble_detection(image_path, **kwargs)
            results.append(result)
        
        return results


# Compatibility classes - MAINTAIN ALL ORIGINAL CLASSES
class LaMaModel(FFCInpaintModel):
    pass

class MATModel(FFCInpaintModel):
    pass

class AOTModel(FFCInpaintModel):
    pass

class SDInpaintModel(FFCInpaintModel):
    pass

class AnimeMangaInpaintModel(FFCInpaintModel):
    pass

class LaMaOfficialModel(FFCInpaintModel):
    pass


class HybridInpainter:
    """Hybrid inpainter for compatibility"""
    
    def __init__(self):
        self.inpainters = {}
    
    def add_method(self, name, method, model_path):
        """Add a method - maintains compatibility"""
        try:
            inpainter = LocalInpainter()
            if inpainter.load_model(method, model_path):
                self.inpainters[name] = inpainter
                return True
        except:
            pass
        return False
    
    def inpaint_ensemble(self, image: np.ndarray, mask: np.ndarray, 
                        weights: Dict[str, float] = None) -> np.ndarray:
        """Ensemble inpainting"""
        if not self.inpainters:
            logger.error("No inpainters loaded")
            return image
        
        if weights is None:
            weights = {name: 1.0 / len(self.inpainters) for name in self.inpainters}
        
        results = []
        for name, inpainter in self.inpainters.items():
            result = inpainter.inpaint(image, mask)
            weight = weights.get(name, 1.0 / len(self.inpainters))
            results.append(result * weight)
        
        ensemble = np.sum(results, axis=0).astype(np.uint8)
        return ensemble


# Helper function for quick setup
def setup_inpainter_for_manga(auto_download=True):
    """Quick setup for manga inpainting"""
    inpainter = LocalInpainter()
    
    if auto_download:
        # Try to download anime JIT model
        jit_path = inpainter.download_jit_model('anime')
        if jit_path:
            inpainter.load_model('anime', jit_path)
            logger.info("✅ Manga inpainter ready with JIT model")
    
    return inpainter


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "download_jit":
            # Download JIT models
            inpainter = LocalInpainter()
            for method in ['lama', 'anime', 'lama_official']:
                print(f"\nDownloading {method}...")
                path = inpainter.download_jit_model(method)
                if path:
                    print(f"  ✅ Downloaded to: {path}")
        
        elif len(sys.argv) > 2:
            # Test with model
            inpainter = LocalInpainter()
            inpainter.load_model('lama', sys.argv[1])
            print("Model loaded - check logs for details")
    
    else:
        print("\nLocal Inpainter - Compatible Version")
        print("=====================================")
        print("\nSupports both:")
        print("  - JIT models (.pt) - RECOMMENDED")
        print("  - Checkpoint files (.ckpt) - With warnings")
        print("\nTo download JIT models:")
        print("  python local_inpainter.py download_jit")
        print("\nTo test:")
        print("  python local_inpainter.py <model_path>")