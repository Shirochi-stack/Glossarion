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
ONNX_CACHE_DIR = os.environ.get('ONNX_CACHE_DIR', 'onnx_cache')
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
        logger.warning(f"PyTorch not available in frozen environment: {e}")
        logger.info("💡 Inpainting will use OpenCV fallback methods")
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
        logger.warning("PyTorch not available - will use OpenCV fallback")

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
            return image  # Return input image as fallback
            
        if not TORCH_AVAILABLE or torch is None:
            logger.error("PyTorch not available for forward pass")
            return image  # Return input image as fallback
            
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
            return image  # Return input image as fallback
    
    def _ffc_block(self, x_l, x_g, idx, conv_type):
        if not self._pytorch_available:
            return x_l, x_g  # Return unchanged inputs as fallback
            
        if not TORCH_AVAILABLE:
            return x_l, x_g  # Return unchanged inputs as fallback
            
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
            return x_l, x_g  # Return unchanged inputs as fallback


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
        self.use_opencv_fallback = False
        self.onnx_session = None
        self.use_onnx = False
        self.is_jit_model = False
        self.pad_mod = 8
        
        # Bubble detection
        self.bubble_detector = None
        self.bubble_model_loaded = False
        
        # Create directories
        os.makedirs(ONNX_CACHE_DIR, exist_ok=True)
        os.makedirs(CACHE_DIR, exist_ok=True)
        logger.info(f"📁 ONNX cache directory: {ONNX_CACHE_DIR}")
        
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
        
        # Initialize bubble detector if available
        if BUBBLE_DETECTOR_AVAILABLE:
            try:
                self.bubble_detector = BubbleDetector()
                logger.info("🗨️ Bubble detection available")
            except:
                self.bubble_detector = None
                logger.info("🗨️ Bubble detection not available")
    
    def _load_config(self):
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def _save_config(self):
        # Load existing config
        full_config = {}
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r', encoding='utf-8') as f:
                full_config = json.load(f)
        
        # Update with inpainter settings directly
        full_config.update(self.config)
        
        # Write back
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(full_config, f, indent=2, ensure_ascii=False)

    def convert_to_onnx(self, model_path: str, method: str) -> Optional[str]:
        """Convert a PyTorch model to ONNX format"""
        if not ONNX_AVAILABLE:
            logger.warning("ONNX not available, skipping conversion")
            return None
        
        # Skip ONNX conversion for models with FFT operations
        fft_models = ['lama', 'anime', 'lama_official']
        if method in fft_models:
            logger.info(f"ℹ️ {method.upper()} uses FFT operations that ONNX doesn't support")
            logger.info("  Using optimized PyTorch JIT model instead")
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
            
            # Create dummy inputs based on model type
            if method == 'aot':
                # AOT expects normalized inputs in [-1, 1]
                dummy_image = torch.randn(1, 3, 512, 512).to(self.device)
                dummy_mask = torch.randn(1, 1, 512, 512).to(self.device)
            else:
                # Other models
                dummy_image = torch.randn(1, 3, 512, 512).to(self.device)
                dummy_mask = torch.randn(1, 1, 512, 512).to(self.device)
            
            # Export to ONNX
            torch.onnx.export(
                self.model,
                (dummy_image, dummy_mask),
                onnx_path,
                export_params=True,
                opset_version=11,  # AOT should work with opset 11
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
            
            # Verify the ONNX model
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            logger.info("✅ ONNX model verified successfully")
            
            return onnx_path
            
        except Exception as e:
            logger.error(f"❌ ONNX conversion failed: {e}")
            logger.error(traceback.format_exc())
            return None
        
    def load_onnx_model(self, onnx_path: str) -> bool:
        """Load an ONNX model for inference"""
        if not ONNX_AVAILABLE:
            logger.error("ONNX Runtime not available")
            return False
        
        try:
            logger.info(f"📥 Loading ONNX model: {onnx_path}")
            
            # Create ONNX Runtime session
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.use_gpu else ['CPUExecutionProvider']
            self.onnx_session = ort.InferenceSession(onnx_path, providers=providers)
            
            # Get input/output names
            self.onnx_input_names = [i.name for i in self.onnx_session.get_inputs()]
            self.onnx_output_names = [o.name for o in self.onnx_session.get_outputs()]
            
            self.use_onnx = True
            logger.info(f"✅ ONNX model loaded with providers: {providers}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load ONNX model: {e}")
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
                    
                    self.config[f'{method}_model_path'] = model_path
                    self._save_config()
                    
                    # ONNX CONVERSION (but handle failures gracefully)
                    if AUTO_CONVERT_TO_ONNX and self.model_loaded:
                        try:
                            onnx_path = self.convert_to_onnx(model_path, method)
                            if onnx_path and self.load_onnx_model(onnx_path):
                                logger.info("🚀 Using ONNX model for inference")
                            else:
                                logger.info("📦 Using PyTorch JIT model for inference")
                        except Exception as onnx_error:
                            logger.warning(f"ONNX conversion failed: {onnx_error}")
                            logger.info("📦 Using PyTorch JIT model for inference")
                    
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
                            
                except Exception as model_error:
                    logger.error(f"Failed to create or initialize model: {model_error}")
                    logger.info("This may happen if PyTorch neural network modules are not available in the .exe build")
                    return False
            
            self.model_loaded = True
            self.current_method = method
            
            self.config[f'{method}_model_path'] = model_path
            self._save_config()
            
            logger.info(f"✅ {method.upper()} loaded!")
            
            # ONNX CONVERSION
            if AUTO_CONVERT_TO_ONNX and model_path.endswith('.pt') and self.model_loaded:
                try:
                    onnx_path = self.convert_to_onnx(model_path, method)
                    if onnx_path and self.load_onnx_model(onnx_path):
                        logger.info("🚀 Using ONNX model for inference")
                except Exception as onnx_error:
                    logger.warning(f"ONNX conversion failed: {onnx_error}")
                    logger.info("📦 Continuing with PyTorch model")

            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            logger.error(traceback.format_exc())
            logger.info("Note: If running from .exe, some ML libraries may not be included")
            logger.info("This is normal for lightweight builds - inpainting will be disabled")
            self.model_loaded = False
            return False
    
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

    def inpaint(self, image, mask, refinement='normal'):
        """Inpaint - compatible with JIT, checkpoint, and ONNX models"""
        if not self.model_loaded:
            logger.error("No model loaded")
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
            
            # ONNX inference path
            if self.use_onnx and self.onnx_session:
                logger.debug("Using ONNX inference")
                
                # Convert BGR to RGB
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Pad images
                image_padded, padding = self.pad_img_to_modulo(image_rgb, self.pad_mod)
                mask_padded, _ = self.pad_img_to_modulo(mask, self.pad_mod)
                
                # Prepare inputs based on model type
                if self.current_method == 'aot':
                    # AOT normalization
                    img_np = image_padded.astype(np.float32) / 127.5 - 1.0
                    mask_np = mask_padded.astype(np.float32) / 255.0
                    mask_np[mask_np < 0.5] = 0
                    mask_np[mask_np >= 0.5] = 1
                    
                    # Apply mask to input for AOT
                    img_np = img_np * (1 - mask_np[:, :, np.newaxis])
                    
                    # Convert to NCHW format
                    img_np = img_np.transpose(2, 0, 1)[np.newaxis, ...]
                    mask_np = mask_np[np.newaxis, np.newaxis, ...]
                else:
                    # LaMa normalization
                    img_np = image_padded.astype(np.float32) / 255.0
                    mask_np = mask_padded.astype(np.float32) / 255.0
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
                
                # Post-process output
                if self.current_method == 'aot':
                    # Denormalize from [-1, 1] to [0, 255]
                    result = ((output[0].transpose(1, 2, 0) + 1.0) * 127.5)
                else:
                    # Denormalize from [0, 1] to [0, 255]
                    result = output[0].transpose(1, 2, 0) * 255
                
                result = np.clip(np.round(result), 0, 255).astype(np.uint8)
                
                # Convert RGB to BGR
                result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
                
                # Remove padding
                result = self.remove_padding(result, padding)
            
            elif self.is_jit_model:
                # JIT model processing
                if self.current_method == 'aot':
                    # Special handling for AOT model
                    logger.debug("Using AOT-specific preprocessing")
                    
                    # Convert BGR to RGB
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
                    
                    # CRITICAL FOR AOT: Apply mask to input image
                    img_torch = img_torch * (1 - mask_torch)
                    
                    logger.debug(f"AOT Image shape: {img_torch.shape}, Mask shape: {mask_torch.shape}")
                    
                    # Run inference
                    with torch.no_grad():
                        inpainted = self.model(img_torch, mask_torch)
                    
                    # Post-process AOT output: denormalize from [-1, 1] to [0, 255]
                    result = ((inpainted.cpu().squeeze_(0).permute(1, 2, 0).numpy() + 1.0) * 127.5)
                    result = np.clip(np.round(result), 0, 255).astype(np.uint8)
                    
                    # Convert RGB back to BGR
                    result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
                    
                    # Remove padding
                    result = self.remove_padding(result, padding)
                    
                else:
                    # LaMa/Anime model processing
                    logger.debug(f"Using standard processing for {self.current_method}")
                    
                    # Convert BGR to RGB (CRITICAL - JIT models expect RGB)
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
                    image_tensor = torch.from_numpy(image_norm).float()
                    
                    # Ensure image is [H, W, C] -> [C, H, W]
                    if len(image_tensor.shape) == 3:
                        if image_tensor.shape[2] == 3:  # [H, W, C] format
                            image_tensor = image_tensor.permute(2, 0, 1)
                    
                    # Add batch dimension [C, H, W] -> [B, C, H, W]
                    image_tensor = image_tensor.unsqueeze(0)
                    
                    # Mask should be [B, 1, H, W]
                    mask_tensor = torch.from_numpy(mask_binary).float()
                    
                    # Add dimensions as needed
                    while len(mask_tensor.shape) < 4:
                        mask_tensor = mask_tensor.unsqueeze(0)
                    
                    # Ensure mask has single channel
                    if mask_tensor.shape[1] != 1:
                        mask_tensor = mask_tensor[:, :1, :, :]
                    
                    # Move to device
                    image_tensor = image_tensor.to(self.device)
                    mask_tensor = mask_tensor.to(self.device)
                    
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
                    
                    # Convert RGB back to BGR for OpenCV
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
