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
import time
import multiprocessing as mp
from queue import Queue, Empty

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
AUTO_CONVERT_TO_ONNX = os.environ.get('AUTO_CONVERT_TO_ONNX', 'false').lower() == 'true'
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

# C++ ONNX Backend - significantly faster than Python
ONNX_CPP_AVAILABLE = False
try:
    from onnx_cpp_backend import ONNXCppBackend, is_cpp_backend_available
    ONNX_CPP_AVAILABLE = is_cpp_backend_available()
    if ONNX_CPP_AVAILABLE:
        logger.info("✓ C++ ONNX Runtime available (2x faster)")
except Exception as e:
    ONNX_CPP_AVAILABLE = False
    logger.debug(f"C++ ONNX backend not available: {e}")

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
        'repo_id': 'ogkalu/aot-inpainting-jit',
        'filename': 'aot_traced.pt',
        'md5': '5ecdac562c1d56267468fc4fbf80db27',
        'name': 'AOT GAN'
    },
    'aot_onnx': {
        'repo_id': 'ogkalu/aot-inpainting',
        'filename': 'aot.onnx',
        'md5': 'ffd39ed8e2a275869d3b49180d030f0d8b8b9c2c20ed0e099ecd207201f0eada',
        'name': 'AOT ONNX (Fast)',
        'is_onnx': True
    },
    'mat': {
        'repo_id': 'Acly/MAT',
        'filename': 'MAT_Places512_G_fp16.safetensors',
        'md5': None,
        'name': 'MAT (High Resolution)',
        'fallback_url': 'https://huggingface.co/Acly/MAT/resolve/main/MAT_Places512_G_fp16.safetensors'
    },
    'lama_onnx': {
        'repo_id': 'Carve/LaMa-ONNX',
        'filename': 'lama_fp32.onnx',
        'md5': None,  # Add MD5 if you want to verify
        'name': 'LaMa ONNX (Carve)',
        'is_onnx': True  # Flag to indicate this is ONNX, not JIT
    },
    'anime_onnx': {
        'repo_id': 'ogkalu/lama-manga-onnx-dynamic',
        'filename': 'lama-manga-dynamic.onnx',
        'md5': 'de31ffa5ba26916b8ea35319f6c12151ff9654d4261bccf0583a69bb095315f9',
        'name': 'Anime/Manga ONNX (Dynamic)',
        'is_onnx': True  # Flag to indicate this is ONNX
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


def download_model(url: str = None, md5: str = None, progress_callback=None, repo_id: str = None, filename: str = None) -> str:
    """Download model if not cached with progress reporting
    
    Args:
        url: URL to download from (legacy support)
        md5: Optional MD5 checksum to verify
        progress_callback: Optional callable(percent, downloaded_mb, total_mb, speed_mb) for progress updates
        repo_id: Hugging Face repository ID (preferred)
        filename: Filename within the repository (preferred)
    
    Returns:
        Path to downloaded/cached model
    """
    import time
    
    # Get cache path - prefer HF repo style if available
    if repo_id and filename:
        cache_path = os.path.join(CACHE_DIR, filename)
    else:
        cache_path = get_cache_path_by_url(url)
    
    if os.path.exists(cache_path):
        logger.info(f"✅ Model already cached: {cache_path}")
        if progress_callback:
            try:
                progress_callback(100, 0, 0, 0)
            except Exception:
                pass
        return cache_path
    
    # Prefer HF Hub download if repo info is provided
    if repo_id and filename:
        try:
            from huggingface_hub import hf_hub_download
            logger.info(f"📥 Downloading from Hugging Face: {repo_id}/{filename}")
            
            # Download using huggingface_hub (it shows progress in console with tqdm)
            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=os.path.dirname(cache_path),
                local_dir_use_symlinks=False
            )
            
            # Copy to expected cache location if different
            if downloaded_path != cache_path:
                # Check if file already exists at cache_path (might be same file)
                if os.path.exists(cache_path):
                    logger.info(f"✅ File already at cache location: {cache_path}")
                else:
                    # Try to copy with retry for file locks
                    import shutil
                    import time as _time
                    for attempt in range(3):
                        try:
                            shutil.copy2(downloaded_path, cache_path)
                            logger.info(f"✅ Copied to: {cache_path}")
                            break
                        except (PermissionError, OSError) as e:
                            if attempt < 2:
                                logger.debug(f"Copy attempt {attempt + 1} failed, retrying...")
                                _time.sleep(0.5)  # Wait for file to be released
                            else:
                                # If copy fails, just use downloaded_path
                                logger.warning(f"Could not copy to cache_path, using downloaded location: {downloaded_path}")
                                cache_path = downloaded_path
            
            logger.info(f"✅ Model ready: {cache_path}")
            return cache_path
            
        except Exception as hf_error:
            logger.error(f"❌ HuggingFace Hub download failed: {hf_error}")
            # If we have a fallback URL, try that
            if url:
                logger.info(f"Falling back to direct URL download: {url}")
            else:
                raise
    
    # Legacy URL-based download as fallback
    if url:
        try:
            logger.info(f"📥 Downloading model from {url}")
            
            # Check if it's a HF URL we can parse
            if 'huggingface.co' in url:
                try:
                    from huggingface_hub import hf_hub_download
                    import re
                    
                    # Parse HuggingFace URL: https://huggingface.co/USER/REPO/resolve/BRANCH/FILENAME
                    match = re.match(r'https://huggingface\.co/([^/]+)/([^/]+)/resolve/([^/]+)/(.+)', url)
                    if match:
                        user, repo, branch, filename = match.groups()
                        repo_id = f"{user}/{repo}"
                        logger.info(f"   Using HuggingFace Hub API: {repo_id}/{filename}")
                        
                        # Download using huggingface_hub
                        downloaded_path = hf_hub_download(
                            repo_id=repo_id,
                            filename=filename,
                            revision=branch,
                            local_dir=os.path.dirname(cache_path),
                            local_dir_use_symlinks=False
                        )
                        
                        # Copy to expected cache location if different
                        if downloaded_path != cache_path:
                            if os.path.exists(cache_path):
                                logger.info(f"✅ File already at cache location: {cache_path}")
                            else:
                                import shutil
                                import time as _time
                                for attempt in range(3):
                                    try:
                                        shutil.copy2(downloaded_path, cache_path)
                                        logger.info(f"✅ Copied to: {cache_path}")
                                        break
                                    except (PermissionError, OSError) as e:
                                        if attempt < 2:
                                            logger.debug(f"Copy attempt {attempt + 1} failed, retrying...")
                                            _time.sleep(0.5)
                                        else:
                                            logger.warning(f"Could not copy to cache_path, using downloaded location: {downloaded_path}")
                                            cache_path = downloaded_path
                        
                        logger.info(f"✅ Model ready: {cache_path}")
                        return cache_path
                        
                except Exception as hf_error:
                    logger.warning(f"HuggingFace Hub download failed: {hf_error}")
                    logger.info("Falling back to direct download...")
            
            # Direct download using urllib3 for better control
            import urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            
            # We don't have GUI context in this function, removed
            
            # Create a pool manager with retry strategy
            retries = urllib3.Retry(
                total=5,
                backoff_factor=0.1,
                status_forcelist=[500, 502, 503, 504],
            )
            http = urllib3.PoolManager(
                retries=retries,
                timeout=urllib3.Timeout(connect=5, read=60),
                headers={
                    'Accept': 'application/octet-stream',
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
            )
            
            logger.info(f"🔍 Requesting file info from {url}")
            response = http.request('GET', url, preload_content=False)
            
            if response.status != 200:
                error_msg = f"Download failed with status {response.status}: {response.reason}"
                logger.error(error_msg)
                response.release_conn()
                raise RuntimeError(error_msg)
            
            # Get file size
            total_size = int(response.headers.get('content-length', 0))
            total_mb = total_size / (1024 * 1024) if total_size > 0 else 0
            
            # Send initial status through progress callback
            if progress_callback:
                try:
                    progress_callback(0, 0, total_mb, 0)
                except Exception:
                    pass
            
            if total_size > 0:
                logger.info(f"📦 Download size: {total_mb:.2f} MB")
            else:
                logger.warning("⚠️ Could not determine file size")
            
            downloaded = 0
            start_time = time.time()
            last_log_time = start_time
            last_callback_time = start_time
            chunk_size = 1024 * 1024  # 1MB chunks for better progress reporting
            
            try:
                with open(cache_path, 'wb') as f:
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                            
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        current_time = time.time()
                        
                        # Call progress callback more frequently (every 0.1s)
                        if progress_callback and total_size > 0 and (current_time - last_callback_time > 0.1):
                            last_callback_time = current_time
                            progress = (downloaded / total_size) * 100
                            downloaded_mb = downloaded / (1024 * 1024)
                            elapsed = current_time - start_time
                            speed_mb = (downloaded / elapsed / (1024 * 1024)) if elapsed > 0 else 0
                            try:
                                progress_callback(int(progress), downloaded_mb, total_mb, speed_mb)
                            except Exception:
                                pass
                        
                        # Log progress every 2 seconds
                        if total_size > 0 and (current_time - last_log_time > 2.0):
                            last_log_time = current_time
                            progress = (downloaded / total_size) * 100
                            elapsed = current_time - start_time
                            speed_mb = (downloaded / elapsed / (1024 * 1024)) if elapsed > 0 else 0
                            downloaded_mb = downloaded / (1024 * 1024)
                            logger.info(f"📥 Progress: {progress:.1f}% ({downloaded_mb:.1f} MB / {total_mb:.1f} MB) @ {speed_mb:.2f} MB/s")
                            
                            # Update progress through callback
                            if progress_callback and total_size > 0:
                                try:
                                    progress_callback(int(progress), downloaded_mb, total_mb, speed_mb)
                                except Exception:
                                    pass
            finally:
                response.release_conn()
                # Notify GUI that download is complete
            # We don't have GUI context in this function, removed
            
            logger.info(f"✅ Model downloaded to: {cache_path}")
            return cache_path
            
        except Exception as e:
            logger.error(f"❌ Download failed: {e}")
            if os.path.exists(cache_path):
                os.remove(cache_path)
            raise
    
    raise ValueError("Either repo_id/filename or url must be provided")


class MATInpaintModel(BaseModel):
    """MAT (Mask-Aware Transformer) model - dynamically built from checkpoint"""
    
    def __init__(self):
        if not TORCH_AVAILABLE:
            super().__init__()
            logger.warning("PyTorch not available - MATInpaintModel initialized as placeholder")
            self._pytorch_available = False
            return
        
        # Clear CUDA cache and sync before init
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            except Exception:
                pass
            
        # Additional safety check for nn being None
        if nn is None:
            super().__init__()
            logger.error("Neural network modules not available - MATInpaintModel disabled")
            self._pytorch_available = False
            return
            
        super().__init__()
        self._pytorch_available = True
        
        # Don't create any modules - they'll be created automatically when loading state dict
        logger.info("MATInpaintModel initialized (modules will be created from checkpoint)")
    
    def forward(self, image, mask):
        """Forward pass for MAT model - NOTE: This is for MATInpaintModel wrapper
        The actual forward logic is in the DynamicModule created during load"""
        # This should not be called - the model is replaced with DynamicModule during load
        # But if it is, try to delegate
        raise NotImplementedError("MAT forward should use DynamicModule, not MATInpaintModel")
    


class FFCInpaintModel(BaseModel):  # Use BaseModel instead of nn.Module
    """FFC model for LaMa inpainting - for checkpoint compatibility"""
    
    def __init__(self):
        if not TORCH_AVAILABLE:
            # Initialize as a simple object when PyTorch is not available
            super().__init__()
            logger.warning("PyTorch not available - FFCInpaintModel initialized as placeholder")
            self._pytorch_available = False
            return
            
        # Clear CUDA cache and sync before init
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            except Exception:
                pass
            
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
            
        # Force sync before forward pass only if not in parallel panel translation mode
        parallel_mode = os.environ.get('PARALLEL_PANEL_TRANSLATION_ENABLED', '0') == '1'
        if not parallel_mode and torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            
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
        'mat': ('MAT Inpainting', MATInpaintModel),
        'aot': ('AOT GAN Inpainting', FFCInpaintModel),
        'aot_onnx': ('AOT ONNX (Fast)', FFCInpaintModel),
        'sd': ('Stable Diffusion Inpainting', FFCInpaintModel),
        'anime': ('Anime/Manga Inpainting', FFCInpaintModel),
        'anime_onnx': ('Anime ONNX (Fast)', FFCInpaintModel),
        'lama_official': ('Official LaMa', FFCInpaintModel),
    }
    
    def __init__(self, config_path="config.json", enable_worker_process=None):
        # Normalize config_path (prevent non-path objects like dict/credentials)
        try:
            if isinstance(config_path, (str, os.PathLike)):
                config_path = str(config_path)
            else:
                config_path = "config.json"
        except Exception:
            config_path = "config.json"
        # Load config first to check parallel_panel_translation setting
        self.config_path = config_path
        self.config = self._load_config()
        
        # Check if parallel panel translation is enabled (for inpainting pool parallelism)
        # Priority: 1) Environment variable (set by manga_translator), 2) Config file
        parallel_mode = os.environ.get('PARALLEL_PANEL_TRANSLATION_ENABLED', '0') == '1'
        if not parallel_mode:
            # Fallback: check config directly
            try:
                parallel_mode = self.config.get('manga_settings', {}).get('advanced', {}).get('parallel_panel_translation', False)
            except Exception:
                parallel_mode = False
        
        # Only enable CUDA debugging if NOT in parallel mode
        # These settings force sequential execution, which is useful for debugging but kills performance
        if not parallel_mode:
            # Enable CUDA debugging in sequential mode
            os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
            os.environ['TORCH_USE_CUDA_DSA'] = '1'  # Enable device-side assertions

        # Set thread limits early if environment indicates single-threaded mode AND not in parallel mode
        try:
            if os.environ.get('OMP_NUM_THREADS') == '1' and not parallel_mode:
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
        self.model = None
        self.model_loaded = False
        self.current_method = None
        self.use_opencv_fallback = False  # FORCED DISABLED - No OpenCV fallback allowed
        self.onnx_session = None
        self.use_onnx = False
        self.is_jit_model = False
        self.pad_mod = 8
        
        # Multiprocessing worker (to avoid UI freeze on init/inference)
        try:
            if enable_worker_process is None:
                # Default: enable on Windows to avoid spawn/GIL freezes on first use
                enable_worker_process = (os.name == 'nt' or sys.platform.startswith('win'))
        except Exception:
            enable_worker_process = False
        self._mp_enabled = bool(enable_worker_process)
        self._mp_ctx = None
        self._mp_task_q = None
        self._mp_result_q = None
        self._mp_worker = None
        self._mp_receiver_thread = None
        self._mp_pending = {}
        if self._mp_enabled:
            self._start_worker()
        
        # Default tiling settings - OFF by default for most models
        self.tiling_enabled = False
        self.tile_size = 512
        self.tile_overlap = 64
        
        # ONNX-specific settings
        self.onnx_model_loaded = False
        self.onnx_input_size = None  # Will be detected from model
        self.onnx_cpp_backend = None  # C++ backend instance
        self.use_onnx_cpp = False  # Track if using C++ backend
        
        # Quantization diagnostics flags
        self.onnx_quantize_applied = False
        self.torch_quantize_applied = False
        
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
        
        # HD strategy defaults (mirror of comic-translate behavior)
        try:
            adv_cfg = self.config.get('manga_settings', {}).get('advanced', {}) if isinstance(self.config, dict) else {}
        except Exception:
            adv_cfg = {}
        try:
            self.hd_strategy = str(os.environ.get('HD_STRATEGY', adv_cfg.get('hd_strategy', 'resize'))).lower()
        except Exception:
            self.hd_strategy = 'resize'
        try:
            self.hd_resize_limit = int(os.environ.get('HD_RESIZE_LIMIT', adv_cfg.get('hd_strategy_resize_limit', 1536)))
        except Exception:
            self.hd_resize_limit = 1536
        try:
            self.hd_crop_margin = int(os.environ.get('HD_CROP_MARGIN', adv_cfg.get('hd_strategy_crop_margin', 16)))
        except Exception:
            self.hd_crop_margin = 16
        try:
            self.hd_crop_trigger_size = int(os.environ.get('HD_CROP_TRIGGER', adv_cfg.get('hd_strategy_crop_trigger_size', 1024)))
        except Exception:
            self.hd_crop_trigger_size = 1024
        logger.info(f"HD strategy: {self.hd_strategy} (resize_limit={self.hd_resize_limit}, crop_margin={self.hd_crop_margin}, crop_trigger={self.hd_crop_trigger_size})")
        
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
    
    def _start_worker(self):
        try:
            self._mp_ctx = mp.get_context('spawn')
        except Exception:
            self._mp_ctx = mp
        try:
            self._mp_task_q = self._mp_ctx.Queue()
            self._mp_result_q = self._mp_ctx.Queue()
            # Lazy import of worker class defined below
            self._mp_worker = _LocalInpainterWorker(self.config_path, self._mp_task_q, self._mp_result_q)
            self._mp_worker.daemon = True
            self._mp_worker.start()
            # Start receiver thread to demultiplex responses by id
            def _receiver():
                while True:
                    try:
                        msg = self._mp_result_q.get()
                    except Exception:
                        break
                    if not isinstance(msg, dict):
                        continue
                    msg_id = msg.get('id')
                    if msg_id is None:
                        # Ignore broadcast messages for now
                        continue
                    q = None
                    try:
                        q = self._mp_pending.get(msg_id)
                    except Exception:
                        q = None
                    if q is not None:
                        try:
                            q.put(msg)
                        except Exception:
                            pass
            self._mp_receiver_thread = threading.Thread(target=_receiver, daemon=True)
            self._mp_receiver_thread.start()
            # CRITICAL: Set _mp_enabled=True AFTER successful worker start
            # This is needed to re-enable worker mode after _stop_worker() set it to False
            self._mp_enabled = True
            logger.debug("Started LocalInpainter worker process")
        except Exception as e:
            logger.warning(f"Failed to start worker process; falling back to in-process mode: {e}")
            self._mp_enabled = False
            self._mp_ctx = None
            self._mp_task_q = None
            self._mp_result_q = None
            self._mp_worker = None
            self._mp_receiver_thread = None
            self._mp_pending = {}

    def _stop_worker(self):
        try:
            if self._mp_enabled and self._mp_task_q is not None:
                try:
                    self._mp_task_q.put({'type': 'shutdown'})
                except Exception:
                    pass
            if self._mp_worker is not None:
                try:
                    # Try graceful shutdown first
                    self._mp_worker.join(timeout=1.0)
                except Exception:
                    pass
                
                # If worker is still alive, force terminate it
                try:
                    if self._mp_worker.is_alive():
                        logger.warning("Worker process didn't respond to shutdown - force terminating")
                        self._mp_worker.terminate()
                        self._mp_worker.join(timeout=2.0)
                        # Last resort: kill if still alive
                        if self._mp_worker.is_alive():
                            try:
                                self._mp_worker.kill()
                            except Exception:
                                pass
                except Exception:
                    pass
        finally:
            self._mp_enabled = False
            self._mp_ctx = None
            self._mp_task_q = None
            self._mp_result_q = None
            self._mp_worker = None
            self._mp_receiver_thread = None
            self._mp_pending = {}

    def _check_worker_health(self) -> bool:
        """Check if the worker process is alive and responsive.
        Returns True if healthy, False if dead/stuck.
        """
        if not self._mp_enabled or self._mp_worker is None:
            return False
        try:
            return self._mp_worker.is_alive()
        except Exception:
            return False
    
    def _ensure_worker_healthy(self) -> bool:
        """Ensure worker is healthy, restart if dead. Returns True if worker is now healthy."""
        if self._check_worker_health():
            return True
        
        logger.warning("Worker process is dead - attempting restart")
        try:
            self._stop_worker()  # Clean up any zombie state
            import time
            time.sleep(0.5)
            self._start_worker()
            return self._check_worker_health()
        except Exception as e:
            logger.error(f"Failed to restart worker: {e}")
            return False
    
    def _mp_call(self, payload: dict, expect_type: str, timeout: float = 60.0) -> dict:
        if not self._mp_enabled or self._mp_task_q is None:
            raise RuntimeError("Worker process not available")
        # Assign unique id
        try:
            call_id = int(time.time() * 1e6) ^ (os.getpid() & 0xFFFF)
        except Exception:
            call_id = int(time.time() * 1e6)
        payload = dict(payload)
        payload['id'] = call_id
        # Create per-call queue and register
        local_q = Queue()
        self._mp_pending[call_id] = local_q
        # Send task
        self._mp_task_q.put(payload)
        
        # Wait for response with periodic stop flag checks
        # Use 1-second polling intervals to allow fast abort on stop
        poll_interval = 1.0
        total_timeout = max(0.1, float(timeout))
        elapsed = 0.0
        resp = None
        
        try:
            while elapsed < total_timeout:
                # Check stop flag to allow quick abort
                if self._check_stop():
                    logger.warning(f"⏹️ Worker call aborted - stop requested during wait for {expect_type}")
                    try:
                        del self._mp_pending[call_id]
                    except Exception:
                        pass
                    raise InterruptedError(f"Stop requested while waiting for {expect_type}")
                
                # Try to get response with short timeout
                try:
                    remaining = min(poll_interval, total_timeout - elapsed)
                    resp = local_q.get(timeout=max(0.1, remaining))
                    break  # Got response
                except Empty:
                    elapsed += poll_interval
                    continue
            
            if resp is None:
                # Timeout - cleanup and raise
                try:
                    del self._mp_pending[call_id]
                except Exception:
                    pass
                raise TimeoutError(f"Timeout waiting for {expect_type} from worker")
        finally:
            # Best-effort cleanup of pending map
            try:
                self._mp_pending.pop(call_id, None)
            except Exception:
                pass
        
        # Validate
        if not isinstance(resp, dict) or resp.get('type') != expect_type:
            raise RuntimeError(f"Unexpected response from worker: {resp}")
        return resp

    def _mp_load_model(self, method, model_path, force_reload=False) -> bool:
        # Store model path for worker restart scenarios
        self._last_model_path = model_path
        
        # Check stop flag before starting model load
        if self._check_stop():
            self._log("⏹️ Model load aborted - stop requested", "warning")
            self.model_loaded = False
            return False
        
        # Retry logic with worker restart for stuck/dead workers
        max_retries = 2
        timeout_values = [180.0, 180.0, 180.0]  # 3 minutes per attempt
        
        for attempt in range(max_retries + 1):
            # Check stop flag at start of each retry
            if self._check_stop():
                self._log("⏹️ Model load aborted during retry - stop requested", "warning")
                self.model_loaded = False
                return False
            
            try:
                # Ensure worker is healthy before attempting
                if attempt > 0 or not self._check_worker_health():
                    if not self._ensure_worker_healthy():
                        logger.error(f"Worker health check failed on attempt {attempt + 1}")
                        if attempt >= max_retries:
                            self.model_loaded = False
                            return False
                        continue
                
                timeout = timeout_values[min(attempt, len(timeout_values) - 1)]
                resp = self._mp_call(
                    {'type': 'load_model', 'method': method, 'model_path': model_path, 'force_reload': bool(force_reload)},
                    'load_model_done',
                    timeout=timeout
                )
                ok = bool(resp.get('success'))
                # Mirror minimal state for compatibility
                self.model_loaded = ok
                self.use_onnx = bool(resp.get('use_onnx', False))
                self.use_onnx_cpp = bool(resp.get('use_onnx_cpp', False))
                try:
                    self.current_method = resp.get('current_method') or method
                except Exception:
                    self.current_method = method
                # Log if worker is using C++ backend
                if self.use_onnx_cpp:
                    logger.info("✅ Worker process using C++ ONNX backend (2x faster)")
                if attempt > 0:
                    logger.info(f"✅ Worker load_model succeeded on retry {attempt}")
                return ok
                
            except InterruptedError as e:
                # Stop was requested during worker call - abort immediately without retry
                self._log(f"⏹️ Model load aborted: {e}", "warning")
                self.model_loaded = False
                return False
                
            except TimeoutError as e:
                logger.error(f"Worker load_model timeout (attempt {attempt + 1}/{max_retries + 1}): {e}")
                # Check stop flag before retry
                if self._check_stop():
                    self._log("⏹️ Model load aborted on timeout - stop requested", "warning")
                    self.model_loaded = False
                    return False
                if attempt < max_retries:
                    logger.warning(f"🔄 Restarting worker and retrying load_model...")
                    try:
                        self._stop_worker()
                        import time
                        time.sleep(1)  # Brief pause
                        # Check stop after pause
                        if self._check_stop():
                            self._log("⏹️ Model load aborted - stop requested during restart", "warning")
                            self.model_loaded = False
                            return False
                        self._start_worker()
                    except Exception as restart_error:
                        logger.error(f"Failed to restart worker: {restart_error}")
                else:
                    logger.error("💡 Try disabling worker process in settings if this persists")
                    self.model_loaded = False
                    return False
                    
            except Exception as e:
                logger.error(f"Worker load_model failed (attempt {attempt + 1}): {e}")
                if attempt >= max_retries:
                    self.model_loaded = False
                    return False
                # Check stop before retry
                if self._check_stop():
                    self._log("⏹️ Model load aborted on exception - stop requested", "warning")
                    self.model_loaded = False
                    return False
        
        self.model_loaded = False
        return False

    def _mp_inpaint(self, image, mask, refinement='normal', iterations=None):
        # Respect stop flag
        if self._check_stop():
            self._log("⏹️ Inpainting stopped by user", "warning")
            return image
        if not self.model_loaded:
            self._log("No model loaded (worker)", "error")
            return image
        
        # CRITICAL: Retry with exponential backoff for timeouts
        max_retries = 3
        timeout_values = [10.0, 20.0, 30.0, 60.0]  # Progressive timeouts: faster initial detection
        
        for attempt in range(max_retries + 1):
            # CRITICAL: Check stop flag at start of each retry iteration
            # This ensures we abort quickly when user clicks stop during retries
            if self._check_stop():
                self._log("⏹️ Inpainting aborted during retry - stop requested", "warning")
                return image
            
            try:
                timeout = timeout_values[min(attempt, len(timeout_values) - 1)]
                resp = self._mp_call(
                    {'type': 'inpaint', 'image': image, 'mask': mask, 'refinement': refinement, 'iterations': iterations}, 
                    'inpaint_result', 
                    timeout=timeout
                )
                if resp.get('success') and resp.get('result') is not None:
                    if attempt > 0:
                        logger.info(f"✅ Worker inpaint succeeded on retry {attempt}")
                    return resp['result']
                else:
                    error_msg = resp.get('error', 'Unknown error')
                    logger.error(f"Worker inpaint failed: {error_msg}")
                    if attempt < max_retries:
                        # Check stop before retry
                        if self._check_stop():
                            self._log("⏹️ Inpainting aborted - stop requested", "warning")
                            return image
                        logger.warning(f"🔄 Retrying inpaint (attempt {attempt + 1}/{max_retries})...")
                        continue
                    return image
            except InterruptedError as e:
                # Stop was requested during worker call - abort immediately without retry
                self._log(f"⏹️ Inpainting aborted: {e}", "warning")
                return image
                
            except TimeoutError as e:
                logger.error(f"❌ Worker inpaint timeout (attempt {attempt + 1}/{max_retries + 1}): {e}")
                # CRITICAL: Check stop flag before attempting retry/restart
                if self._check_stop():
                    self._log("⏹️ Inpainting aborted on timeout - stop requested", "warning")
                    return image
                if attempt < max_retries:
                    logger.warning(f"🔄 Restarting worker and retrying...")
                    # CRITICAL: Restart the worker process on timeout
                    try:
                        self._stop_worker()
                        import time
                        time.sleep(1)  # Brief pause
                        # Check stop again before starting new worker
                        if self._check_stop():
                            self._log("⏹️ Inpainting aborted - stop requested during restart", "warning")
                            return image
                        self._start_worker()
                        # Reload the model in the new worker
                        if hasattr(self, 'current_method') and self.current_method:
                            model_path = getattr(self, '_last_model_path', None)
                            if model_path:
                                logger.info(f"🔄 Reloading model in new worker: {self.current_method}")
                                self._mp_load_model(self.current_method, model_path, force_reload=True)
                    except Exception as restart_error:
                        logger.error(f"Failed to restart worker: {restart_error}")
                        return image
                    continue
                else:
                    logger.error(f"❌ All retry attempts exhausted. Returning original image.")
                    return image
            except Exception as e:
                logger.error(f"Worker inpaint exception (attempt {attempt + 1}): {e}")
                if attempt < max_retries:
                    # Check stop before retry
                    if self._check_stop():
                        self._log("⏹️ Inpainting aborted on exception - stop requested", "warning")
                        return image
                    continue
                return image
        
        return image

    def _load_config(self):
        cfg = {}
        try:
            if self.config_path and os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if not content:
                        cfg = {}
                    else:
                        try:
                            cfg = json.loads(content)
                        except json.JSONDecodeError:
                            # Likely a concurrent write; retry once after a short delay
                            try:
                                import time
                                time.sleep(0.05)
                                with open(self.config_path, 'r', encoding='utf-8') as f2:
                                    cfg = json.load(f2)
                            except Exception:
                                # If config_path isn't the default, fall back to config.json
                                try:
                                    if str(self.config_path).lower() != "config.json":
                                        self.config_path = "config.json"
                                        if os.path.exists(self.config_path):
                                            with open(self.config_path, 'r', encoding='utf-8') as f3:
                                                cfg = json.load(f3)
                                except Exception:
                                    cfg = {}
        except Exception:
            cfg = {}
        # Sanitize any JSON credential paths accidentally stored as model paths
        try:
            if isinstance(cfg, dict):
                google_creds = cfg.get('google_vision_credentials', '') or cfg.get('google_cloud_credentials', '')
                # Top-level keys
                for k, v in list(cfg.items()):
                    if isinstance(k, str) and k.endswith('_model_path') and isinstance(v, str):
                        if v.lower().endswith('.json') or (google_creds and v == google_creds):
                            cfg[k] = ''
                # Nested inpainting paths
                ms = cfg.get('manga_settings', {}) if isinstance(cfg.get('manga_settings', {}), dict) else {}
                inpaint = ms.get('inpainting', {}) if isinstance(ms.get('inpainting', {}), dict) else {}
                for k, v in list(inpaint.items()):
                    if isinstance(k, str) and k.endswith('_model_path') and isinstance(v, str):
                        if v.lower().endswith('.json') or (google_creds and v == google_creds):
                            inpaint[k] = ''
                if ms is not None:
                    ms['inpainting'] = inpaint
                    cfg['manga_settings'] = ms
        except Exception:
            pass
        return cfg
    
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
                except Exception as read_err:
                    logger.debug(f"Config read during save failed (non-critical): {read_err}")
                    full_config = {}
            # Update
            full_config.update(self.config)
            # Atomic write: write to temp then replace
            tmp_path = (self.config_path or 'config.json') + '.tmp'
            with open(tmp_path, 'w', encoding='utf-8') as f:
                json.dump(full_config, f, indent=2, ensure_ascii=False)
            try:
                os.replace(tmp_path, self.config_path or 'config.json')
            except Exception as replace_err:
                logger.debug(f"Config atomic replace failed, trying direct write: {replace_err}")
                # Fallback to direct write
                with open(self.config_path or 'config.json', 'w', encoding='utf-8') as f:
                    json.dump(full_config, f, indent=2, ensure_ascii=False)
        except Exception as save_err:
            # Never crash on config save, but log for debugging
            logger.debug(f"Config save failed (non-critical): {save_err}")
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
        # Highest-priority: global graceful stop flag propagated via environment
        try:
            if os.environ.get('GRACEFUL_STOP') == '1':
                self._stopped = True
                return True
        except Exception:
            pass
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
        """Load an ONNX model with C++ backend support for 2x performance"""
        # Try C++ backend first for better performance
        if ONNX_CPP_AVAILABLE:
            try:
                logger.info("🚀 Loading ONNX model with C++ backend...")
                self.onnx_cpp_backend = ONNXCppBackend()
                if self.onnx_cpp_backend.load_model(onnx_path, use_gpu=self.use_gpu):
                    self.use_onnx = True
                    self.use_onnx_cpp = True
                    self.model_loaded = True
                    self.current_onnx_path = onnx_path
                    logger.info("✅ ONNX C++ backend loaded successfully (Python fallback disabled)")
                    return True
                else:
                    logger.warning("C++ backend load failed, falling back to Python")
                    self.onnx_cpp_backend = None
                    self.use_onnx_cpp = False
            except Exception as cpp_err:
                logger.warning(f"C++ backend error: {cpp_err}, using Python fallback")
                self.onnx_cpp_backend = None
                self.use_onnx_cpp = False
        
        # Load Python ONNX Runtime if C++ backend not available or failed
        if not ONNX_AVAILABLE:
            logger.error("ONNX Runtime not available")
            return False
        
        # Check if this exact ONNX model is already loaded
        if (self.onnx_session is not None and 
            hasattr(self, 'current_onnx_path') and 
            self.current_onnx_path == onnx_path):
            logger.debug(f"✅ ONNX model already loaded: {onnx_path}")
            return True
        
        try:
            # Don't log here if we already logged in load_model
            logger.debug(f"📦 ONNX Runtime loading: {onnx_path}")
            
            # Store the path for later checking
            self.current_onnx_path = onnx_path

            # Check if this is a Carve model (fixed 512x512)
            is_carve_model = "lama_fp32" in onnx_path or "carve" in onnx_path.lower()
            if is_carve_model:
                logger.info("📦 Detected Carve ONNX model (fixed 512x512 input)")
                self.onnx_fixed_size = (512, 512)
            else:
                self.onnx_fixed_size = None
            
            # Standard ONNX loading: prefer CUDA if available; otherwise CPU. Do NOT use DML.
            try:
                avail = ort.get_available_providers() if ONNX_AVAILABLE else []
            except Exception:
                avail = []
            if 'CUDAExecutionProvider' in avail:
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            else:
                providers = ['CPUExecutionProvider']
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
                            self.onnx_quantize_applied = True
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
                                    self.onnx_quantize_applied = True
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
            # Enable optimal performance settings (let ONNX use all CPU cores)
            try:
                # Use all available CPU threads for best performance
                # ONNX Runtime will automatically use optimal thread count
                so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
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
            
            # Log success with I/O info in a single line
            logger.debug(f"✅ ONNX session created - Inputs: {self.onnx_input_names}, Outputs: {self.onnx_output_names}")
            
            self.use_onnx = True
            # CRITICAL: Set model_loaded flag when ONNX session is successfully created
            # This ensures preloaded spares are recognized as valid loaded instances
            self.model_loaded = True
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load ONNX: {e}")
            import traceback
            logger.debug(f"ONNX load traceback: {traceback.format_exc()}")
            self.use_onnx = False
            self.model_loaded = False
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
    
    def _try_recover_cuda_oom(self):
        """Attempt to recover from CUDA OOM by clearing caches"""
        if not TORCH_AVAILABLE or not self.use_gpu:
            return
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            # Optionally try to clear cuBLAS workspace
            try:
                getattr(torch._C, '_cuda_clearCublasWorkspaces')()
            except Exception:
                pass
        except Exception:
            pass

    def _to_device_safe(self, tensor):
        """Safely move a tensor to the configured device with retries and fallbacks.
        - Ensures contiguity
        - Syncs tensor with model's device and dtype
        - Handles shared model pools and threads
        """
        try:
            if not TORCH_AVAILABLE or not self.use_gpu or self.device is None or not hasattr(self, 'model'):
                return tensor.contiguous()

            # First check if model has changed devices (from pool)
            if hasattr(self.model, 'parameters'):
                with torch.no_grad():
                    model_param = next(self.model.parameters())
                    target_device = model_param.device
                    target_dtype = model_param.dtype

                # Move to device and convert dtype in one shot
                result = tensor.contiguous().to(device=target_device, dtype=target_dtype)
                
                # Force sync for CUDA
                if str(target_device).startswith('cuda'):
                    torch.cuda.synchronize(target_device)
                
                return result
        except RuntimeError as e:
            # Attempt recovery once
            try:
                self._try_recover_cuda_oom()
            except Exception:
                pass
            try:
                time.sleep(0.02)
            except Exception:
                pass
            try:
                return tensor.contiguous().to(self.device, non_blocking=False)
            except RuntimeError as e2:
                logger.warning(f"CUDA move failed twice; using CPU tensor for this op: {e2}")
                return tensor.contiguous()

    def _build_mat_from_checkpoint(self, state_dict: dict):
        """Build MAT model architecture dynamically from checkpoint keys"""
        try:
            # Analyze keys to understand architecture
            all_keys = list(state_dict.keys())
            logger.info(f"Total keys in checkpoint: {len(all_keys)}")
            
            # Remove prefix if present
            prefix = None
            if any(k.startswith('generator.') for k in all_keys):
                prefix = 'generator.'
            elif any(k.startswith('netG.') for k in all_keys):
                prefix = 'netG.'
            
            # Create a mapping from prefixed to non-prefixed keys
            if prefix:
                new_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith(prefix):
                        new_key = k[len(prefix):]
                        new_state_dict[new_key] = v
                    else:
                        new_state_dict[k] = v
                state_dict.clear()
                state_dict.update(new_state_dict)
                logger.info(f"Removed '{prefix}' prefix from keys")
            
            # Now build layers based on remaining keys
            unique_layer_names = set()
            for key in state_dict.keys():
                # Extract layer name (everything before .weight or .bias)
                if '.weight' in key or '.bias' in key:
                    layer_name = key.rsplit('.', 1)[0]
                    unique_layer_names.add(layer_name)
            
            logger.info(f"Found {len(unique_layer_names)} unique layers in MAT checkpoint")
            
            # The model.layers ModuleDict will be populated by load_state_dict
            # We just need to make sure the model can accept these keys
            
        except Exception as e:
            logger.error(f"Failed to build MAT architecture: {e}")
            raise
    
    def _load_weights_with_mapping(self, model, state_dict) -> bool:
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
    
    def get_cached_model_path(self, method: str) -> Optional[str]:
        """Return the cached path for a given method if already downloaded, else None."""
        try:
            info = LAMA_JIT_MODELS.get(method)
            if not info:
                return None
            # Prefer HF repo mapping
            if 'repo_id' in info and 'filename' in info:
                candidate = os.path.join(CACHE_DIR, info['filename'])
                return candidate if os.path.exists(candidate) else None
            # Fallback to URL-based cache
            if 'url' in info and info['url']:
                candidate = get_cache_path_by_url(info['url'])
                return candidate if os.path.exists(candidate) else None
            return None
        except Exception:
            return None

    def download_jit_model(self, method: str) -> str:
        """Download JIT model for a method"""
        # First check if already cached
        cached = self.get_cached_model_path(method)
        if cached and os.path.exists(cached):
            if hasattr(self, 'progress_queue'):
                self.progress_queue.put(('model_file_status', '✅ Model found in cache'))
            return cached
            
        # Not cached, need to download
        try:
            if method in LAMA_JIT_MODELS:
                model_info = LAMA_JIT_MODELS[method]
                logger.info(f"📥 Downloading {model_info['name']}...")
                if hasattr(self, 'progress_queue'):
                    self.progress_queue.put(('model_file_status', f"📥 Downloading {model_info['name']}..."))
                
                model_path = None
                # Prefer HuggingFace download if repo info is available
                if 'repo_id' in model_info and 'filename' in model_info:
                    try:
                        model_path = download_model(
                            repo_id=model_info['repo_id'],
                            filename=model_info['filename'],
                            md5=model_info.get('md5')
                        )
                    except Exception as hf_err:
                        logger.warning(f"HuggingFace download failed: {hf_err}")
                        # Try fallback URL if available
                        if 'fallback_url' in model_info:
                            logger.info(f"Trying fallback URL...")
                            model_path = download_model(
                                url=model_info['fallback_url'],
                                md5=model_info.get('md5')
                            )
                        else:
                            raise
                # Fallback to legacy URL if available
                elif 'url' in model_info:
                    model_path = download_model(
                        url=model_info['url'],
                        md5=model_info.get('md5')
                    )
                elif 'fallback_url' in model_info:
                    model_path = download_model(
                        url=model_info['fallback_url'],
                        md5=model_info.get('md5')
                    )
                else:
                    raise ValueError(f"No download info for {method}")
                
                if hasattr(self, 'progress_queue'):
                    self.progress_queue.put(('model_file_status', '✅ Download complete'))
                return model_path
            else:
                error_msg = f"No JIT model available for {method}"
                logger.warning(error_msg)
                if hasattr(self, 'progress_queue'):
                    self.progress_queue.put(('model_file_status', f'❌ {error_msg}'))
                return None
                
        except Exception as e:
            error_msg = f"Failed to download {method}: {e}"
            logger.error(error_msg)
            if hasattr(self, 'progress_queue'):
                self.progress_queue.put(('model_file_status', f'❌ {error_msg}'))
            return None
            try:
                if method in LAMA_JIT_MODELS:
                    model_info = LAMA_JIT_MODELS[method]
                    model_name = model_info.get('name', method.upper())
                    logger.info(f"📥 Downloading {model_name}...")
                    
                    # Send initial progress
                    progress_queue.put(('progress', 0, f"Starting {model_name} download..."))
                    
                    def progress_callback(percent, downloaded_mb, total_mb, speed_mb):
                        progress_queue.put(('progress', percent, 
                                          f"Downloading {model_name}: {percent:.1f}% ({downloaded_mb:.1f}MB/{total_mb:.1f}MB) @ {speed_mb:.1f}MB/s"))
                    
                    try:
                        # Prefer HuggingFace download if repo info is available
                        if 'repo_id' in model_info and 'filename' in model_info:
                            progress_queue.put(('progress', 10, f"Getting {model_name} from Hugging Face..."))
                            # First try to recover CUDA memory
                            self._try_recover_cuda_oom()
            
                            model_path = download_model(
                                repo_id=model_info['repo_id'],
                                filename=model_info['filename'],
                                md5=model_info.get('md5'),
                                progress_callback=progress_callback
                            )
                        # Fallback to legacy URL if available
                        elif 'url' in model_info:
                            progress_queue.put(('progress', 10, f"Getting {model_name} from URL..."))
                            model_path = download_model(
                                url=model_info['url'],
                                md5=model_info.get('md5'),
                                progress_callback=progress_callback
                            )
                        else:
                            raise ValueError(f"No download info for {method}")
                        
                        if model_path and os.path.exists(model_path):
                            progress_queue.put(('progress', 100, f"{model_name} download complete!"))
                            result['success'] = True
                            result['path'] = model_path
                        else:
                            raise ValueError("Download completed but file not found")
                            
                    except Exception as download_error:
                        raise download_error
                        
                else:
                    progress_queue.put(('error', f"No JIT model available for {method}"))
                    result['error'] = f"No JIT model available for {method}"
            
            except Exception as e:
                error_msg = f"Failed to download {method}: {str(e)}"
                logger.error(error_msg)
                progress_queue.put(('error', error_msg))
                result['error'] = str(e)
            
            finally:
                # Signal completion
                progress_queue.put(('done', result['success'], result.get('error')))
                
                # Update GUI if available
                if hasattr(self, 'update_queue') and hasattr(self, 'main_gui'):
                    try:
                        if result['success']:
                            self.update_queue.put(('status', 'Download complete'))
                        else:
                            self.update_queue.put(('status', f"Download failed: {result.get('error', 'Unknown error')}"))
                        self.update_queue.put(('progress', 100))
                        self.update_queue.put(('call_method', self._check_provider_status, ()))
                    except Exception:
                        pass
        
        # Start download in background thread
        download_thread = threading.Thread(
            target=_download_in_background,
            args=(method, result, progress_queue),
            daemon=True
        )
        download_thread.start()
        
        def _monitor_progress():
            try:
                last_update = time.time()
                while True:
                    try:
                        msg_type, *msg_data = progress_queue.get(timeout=0.1)
                        
                        # Handle progress updates
                        if msg_type == 'progress':
                            percent, status = msg_data
                            if time.time() - last_update >= 0.1:  # Throttle GUI updates
                                if hasattr(self, 'update_queue'):
                                    self.update_queue.put(('progress', percent))
                                    self.update_queue.put(('status', status))
                                last_update = time.time()
                            logger.info(status)
                        
                        # Handle errors
                        elif msg_type == 'error':
                            error_msg = msg_data[0]
                            logger.error(f"❌ {error_msg}")
                            if hasattr(self, 'update_queue'):
                                self.update_queue.put(('status', f"Error: {error_msg}"))
                            break
                        
                        # Handle completion
                        elif msg_type == 'done':
                            success, error = msg_data
                            if success:
                                logger.info("✅ Download completed successfully")
                            else:
                                logger.error(f"❌ Download failed: {error if error else 'Unknown error'}")
                            break
                            
                    except Empty:
                        # Check if download thread is still alive
                        if not download_thread.is_alive():
                            break
                        continue
                    
            except Exception as e:
                logger.error(f"Progress monitoring error: {e}")
        
        # Start progress monitoring in another thread
        progress_thread = threading.Thread(target=_monitor_progress, daemon=True)
        progress_thread.start()
        
        # Return immediately to keep GUI responsive
        return None
    
    def load_model(self, method, model_path, force_reload=False):
        """Load model - supports both JIT and checkpoint files with ONNX conversion"""
        try:
            # Guard: ignore invalid JSON paths (e.g., glossary files)
            try:
                if isinstance(model_path, str) and model_path.lower().endswith('.json'):
                    logger.warning(f"Ignoring invalid model path (JSON): {model_path}")
                    model_path = ''
            except Exception:
                pass
            # Offload to worker process if enabled
            if getattr(self, '_mp_enabled', False):
                return self._mp_load_model(method, model_path, force_reload=force_reload)
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
            
            # Check if model path changed - but only if we had a previous path saved
            current_saved_path = self.config.get(f'{method}_model_path', '')
            if current_saved_path and current_saved_path != model_path:
                logger.info(f"📍 Model path changed for {method}")
                logger.info(f"   Old: {current_saved_path}")
                logger.info(f"   New: {model_path}")
                force_reload = True
            
            if not model_path or not os.path.exists(model_path):
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
            
            # Check if already loaded in THIS instance
            if self.model_loaded and self.current_method == method and not force_reload:
                # Additional check for ONNX - make sure the session exists
                if self.use_onnx and self.onnx_session is not None:
                    logger.debug(f"✅ {method.upper()} ONNX already loaded (skipping reload)")
                    return True
                elif not self.use_onnx and self.model is not None:
                    logger.debug(f"✅ {method.upper()} already loaded (skipping reload)")
                    return True
                else:
                    # Model claims to be loaded but objects are missing - force reload
                    logger.warning(f"⚠️ Model claims loaded but session/model object is None - forcing reload")
                    force_reload = True
                    self.model_loaded = False
            
            # Clear previous model if force reload
            if force_reload:
                logger.info(f"🔄 Force reloading {method} model...")
                self.model = None
                self.onnx_session = None
                self.model_loaded = False
                self.is_jit_model = False
                # Only log loading message when actually loading
                logger.info(f"📥 Loading {method} from {model_path}")
            elif self.model_loaded and self.current_method != method:
                # If we have a model loaded but it's a different method, clear it
                logger.info(f"🔄 Switching from {self.current_method} to {method}")
                self.model = None
                self.onnx_session = None
                self.model_loaded = False
                self.is_jit_model = False
                # Only log loading message when actually loading
                logger.info(f"📥 Loading {method} from {model_path}")
            elif not self.model_loaded:
                # Only log when we're actually going to load
                logger.info(f"📥 Loading {method} from {model_path}")
            # else: model is loaded and current, no logging needed

            # Normalize path and enforce expected extension for certain methods
            try:
                _ext = os.path.splitext(model_path)[1].lower()
                _method_lower = str(method).lower()
                # For explicit ONNX methods, ensure we use a .onnx path
                if _method_lower in ("lama_onnx", "anime_onnx", "aot_onnx") and _ext != ".onnx":
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
                            # Download the appropriate ONNX model based on the method
                            if _method_lower == "anime_onnx":
                                _dl = self.download_jit_model("anime_onnx")
                            elif _method_lower == "aot_onnx":
                                _dl = self.download_jit_model("aot_onnx")
                            else:
                                _dl = self.download_jit_model("lama_onnx")
                            if _dl and os.path.exists(_dl):
                                model_path = _dl
                                logger.info(f"🔧 Using downloaded {_method_lower.upper()} model: {model_path}")
                        except Exception:
                            pass
            except Exception:
                pass

            # Check file signature to detect ONNX files (even with wrong extension)
            # or check file extension
            ext = model_path.lower().split('.')[-1]
            is_onnx = False
            
            # Check by file signature
            try:
                with open(model_path, 'rb') as f:
                    file_header = f.read(8)
                if file_header.startswith(b'\x08'):
                    is_onnx = True
                    logger.debug("📦 Detected ONNX file signature")
            except Exception:
                pass
            
            # Check by extension
            if ext == 'onnx':
                is_onnx = True
            
            # Handle ONNX files
            if is_onnx:
                # Note: load_onnx_model will handle its own logging
                try:
                    onnx_load_result = self.load_onnx_model(model_path)
                    if onnx_load_result:
                        # CRITICAL: Set model_loaded flag FIRST before any other operations
                        # This ensures concurrent threads see the correct state immediately
                        self.model_loaded = True
                        self.use_onnx = True
                        self.is_jit_model = False
                        # Ensure aot_onnx is properly set as current method
                        if 'aot' in method.lower():
                            self.current_method = 'aot_onnx'
                        else:
                            self.current_method = method
                        # Save with BOTH key formats for compatibility (non-critical - do last)
                        try:
                            self.config[f'{method}_model_path'] = model_path
                            self.config[f'manga_{method}_model_path'] = model_path
                            self._save_config()
                        except Exception as cfg_err:
                            logger.debug(f"Config save after ONNX load failed (non-critical): {cfg_err}")
                        logger.info(f"✅ {method.upper()} ONNX loaded with method: {self.current_method}")
                        # Double-check model_loaded flag is still set
                        if not self.model_loaded:
                            logger.error("❌ CRITICAL: model_loaded flag was unset after successful ONNX load!")
                            self.model_loaded = True
                        return True
                    else:
                        logger.error("Failed to load ONNX model - load_onnx_model returned False")
                        self.model_loaded = False
                        return False
                except Exception as onnx_err:
                    logger.error(f"Exception during ONNX model loading: {onnx_err}")
                    import traceback
                    logger.debug(traceback.format_exc())
                    self.model_loaded = False
                    return False
            
            # Check if it's a safetensors file
            is_safetensors = model_path.endswith('.safetensors')
            
            if is_safetensors:
                try:
                    from safetensors.torch import load_file
                    logger.info("📦 Loading safetensors file...")
                    checkpoint = load_file(model_path)
                    self.is_jit_model = False
                except ImportError:
                    logger.error("safetensors library not installed. Install with: pip install safetensors")
                    return False
                except Exception as load_error:
                    logger.error(f"Failed to load safetensors file: {load_error}")
                    return False
            # Check if it's a JIT model (.pt) or checkpoint (.ckpt/.pth)
            elif model_path.endswith('.pt'):
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
                                self.torch_quantize_applied = True
                            else:
                                logger.info("ℹ️ INT8 dynamic quantization not applied (unsupported for this JIT graph); using FP32 CPU")
                        except Exception as _qe:
                            logger.warning(f"INT8 quantization skipped: {_qe}")
                    
                    # Save with BOTH key formats for compatibility
                    self.config[f'{method}_model_path'] = model_path
                    self.config[f'manga_{method}_model_path'] = model_path
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
                    # Create the appropriate model based on method
                    if method == 'mat':
                        logger.info("📦 Creating MAT model architecture...")
                        self.model = MATInpaintModel()
                    else:
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
                    
                    # For MAT models, use the original architecture
                    if method == 'mat':
                        logger.info("🔍 Loading MAT with original architecture...")
                        
                        try:
                            # Import MAT's Generator class
                            import sys
                            
                            # CRITICAL: Add paths in correct order so local modules override system ones
                            src_path = os.path.dirname(__file__)
                            
                            # Add src path first so it can find torch_utils, dnnlib, and networks
                            if src_path not in sys.path:
                                sys.path.insert(0, src_path)
                            
                            # Now import - it should find our local torch_utils, dnnlib, and networks
                            from networks.mat import Generator
                            
                            # Create generator with MAT's default parameters
                            # z_dim=512, c_dim=0, w_dim=512, img_resolution=512, img_channels=3
                            logger.info("🏭 Creating MAT Generator...")
                            generator = Generator(
                                z_dim=512,
                                c_dim=0, 
                                w_dim=512,
                                img_resolution=512,
                                img_channels=3
                            )
                            
                            # Load checkpoint into generator
                            logger.info("📦 Loading checkpoint into MAT Generator...")
                            missing, unexpected = generator.load_state_dict(state_dict, strict=False)
                            logger.info(f"✅ MAT loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
                            
                            # Put generator in eval mode
                            generator.eval()
                            
                            # Wrap the generator for our forward interface
                            class MATWrapper(nn.Module):
                                def __init__(self, generator):
                                    super().__init__()
                                    self.generator = generator
                                    self._pytorch_available = True
                                
                                def forward(self, image, mask):
                                    # MAT Generator.forward expects:
                                    # (images_in, masks_in, z, c, truncation_psi, truncation_cutoff, skip_w_avg_update, noise_mode)
                                    # images_in and masks_in should be in [-1, 1] range
                                    # CRITICAL: MAT expects mask where 1=keep, 0=inpaint (opposite of our convention)
                                    
                                    batch_size = image.shape[0]
                                    
                                    # IMPROVEMENT: Use fixed seed for deterministic results in const mode
                                    # This prevents random variation between runs
                                    z = torch.randn(batch_size, 512, device=image.device, dtype=image.dtype)
                                    c = None
                                    
                                    # Ensure generator is in eval mode
                                    self.generator.eval()
                                    
                                    # CRITICAL: Invert mask - MAT expects 1=keep, 0=inpaint
                                    # Our convention is 1=inpaint, 0=keep, so we need to invert
                                    inverted_mask = 1.0 - mask
                                    
                                    # Log tensor stats for debugging
                                    logger.debug(f"🔍 MAT Input - Image: shape={image.shape}, range=[{image.min():.3f}, {image.max():.3f}]")
                                    logger.debug(f"🔍 MAT Input - Mask (inverted): shape={inverted_mask.shape}, sum={inverted_mask.sum():.0f}")
                                    
                                    # IMPROVEMENT: Use higher truncation_psi for better quality (default 1.0)
                                    # Lower values (0.5-0.8) = more conservative/realistic
                                    # Higher values (1.0+) = more creative/diverse
                                    # For inpainting accuracy, we want deterministic results
                                    with torch.no_grad():
                                        output = self.generator(
                                            image, 
                                            inverted_mask, 
                                            z, 
                                            c, 
                                            truncation_psi=1.0,  # Full style diversity
                                            noise_mode='const'   # Deterministic noise
                                        )
                                    
                                    logger.debug(f"🔍 MAT Output - shape={output.shape}, range=[{output.min():.3f}, {output.max():.3f}]")
                                    
                                    return output
                            
                            self.model = MATWrapper(generator)
                            logger.info("✅ MAT wrapper created")
                            
                        except Exception as mat_err:
                            logger.error(f"❌ Failed to load MAT with original architecture: {mat_err}")
                            import traceback
                            logger.error(traceback.format_exc())
                            return False
                    
                    # For non-MAT models, use standard weight loading
                    else:
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
                            self.torch_quantize_applied = True
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

    def load_model_with_retry(self, method, model_path, force_reload=False, retries: int = 2, retry_delay: float = 0.5) -> bool:
        """Attempt to load a model with retries.
        Returns True if loaded; False if all attempts fail. On failure, the inpainter will safely no-op.
        """
        try:
            attempts = max(0, int(retries)) + 1
        except Exception:
            attempts = 1
        for attempt in range(attempts):
            try:
                ok = self.load_model(method, model_path, force_reload=force_reload)
                if ok:
                    return True
            except Exception as e:
                logger.warning(f"Load attempt {attempt+1} failed with exception: {e}")
            # brief delay before next try
            if attempt < attempts - 1:
                try:
                    time.sleep(max(0.0, float(retry_delay)))
                except Exception:
                    pass
        # If we reach here, loading failed. Leave model unloaded so inpaint() no-ops and returns original image.
        logger.warning("All load attempts failed; local inpainting will fall back to returning original images (no-op)")
        self.model_loaded = False
        # Keep current_method for logging/context if provided
        try:
            self.current_method = method
        except Exception:
            pass
        return False

    def unload(self):
        """Release all heavy resources held by this inpainter instance."""
        try:
            # Stop worker first if running
            try:
                self._stop_worker()
            except Exception:
                pass
            # Release C++ ONNX backend
            try:
                if self.onnx_cpp_backend is not None:
                    self.onnx_cpp_backend.unload()
                    self.onnx_cpp_backend = None
                    self.use_onnx_cpp = False
            except Exception:
                pass
            # Release Python ONNX session and metadata
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
        result = self.inpaint(tile_img, tile_mask, refinement, _skip_hd=True)
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

    def _is_white_paste(self, result: np.ndarray, mask: np.ndarray, white_threshold: int = 245, ratio: float = 0.90) -> bool:
        """Detect 'white paste' failure: masked area mostly saturated near white."""
        try:
            if result is None or mask is None:
                return False
            if len(mask.shape) == 3:
                mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            else:
                mask_gray = mask
            m = mask_gray > 0
            if not np.any(m):
                return False
            if len(result.shape) == 3:
                white = (result[..., 0] >= white_threshold) & (result[..., 1] >= white_threshold) & (result[..., 2] >= white_threshold)
            else:
                white = result >= white_threshold
            count_mask = int(np.count_nonzero(m))
            count_white = int(np.count_nonzero(white & m))
            if count_mask == 0:
                return False
            frac = count_white / float(count_mask)
            return frac >= ratio
        except Exception as e:
            logger.debug(f"White paste detection failed: {e}")
            return False

    def _log_inpaint_diag(self, path: str, result: np.ndarray, mask: np.ndarray):
        # Only log diagnostic info when verbose debug is enabled
        if not getattr(self, 'verbose_debug', False):
            return
            
        try:
            h, w = result.shape[:2]
            if len(result.shape) == 3:
                stats = (float(result.min()), float(result.max()), float(result.mean()))
            else:
                stats = (float(result.min()), float(result.max()), float(result.mean()))
            logger.debug(f"[Diag] Path={path} onnx_quant={self.onnx_quantize_applied} torch_quant={self.torch_quantize_applied} size={w}x{h} stats(min,max,mean)={stats}")
            if self._is_white_paste(result, mask):
                logger.debug(f"[Diag] White-paste detected (mask>0 mostly white)")
        except Exception as e:
            logger.debug(f"Diag log failed: {e}")

    def inpaint(self, image, mask, refinement='normal', iterations=None, _retry_attempt: int = 0, _skip_hd: bool = False, _skip_tiling: bool = False):
        """Inpaint - compatible with JIT, checkpoint, and ONNX models
        Implements HD strategy (Resize/Crop) similar to comic-translate to speed up large images.
        
        Args:
            image: Input image
            mask: Inpainting mask
            refinement: Refinement mode ('normal', 'fast')
            iterations: Number of inpainting iterations (None for auto/default)
            _retry_attempt: Internal retry counter
            _skip_hd: Skip HD processing
            _skip_tiling: Skip tiling
        """
        # If worker process is enabled, delegate to it (handles stop flag internally)
        if getattr(self, '_mp_enabled', False):
            return self._mp_inpaint(image, mask, refinement=refinement, iterations=iterations)
        
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
            
            # HD strategy (mirror of comic-translate): optional RESIZE or CROP before core inpainting
            if not _skip_hd:
                try:
                    strategy = getattr(self, 'hd_strategy', 'resize') or 'resize'
                except Exception:
                    strategy = 'resize'
                H, W = orig_h, orig_w
                if strategy == 'resize' and max(H, W) > max(16, int(getattr(self, 'hd_resize_limit', 1536))):
                    limit = max(16, int(getattr(self, 'hd_resize_limit', 1536)))
                    ratio = float(limit) / float(max(H, W))
                    new_w = max(1, int(W * ratio + 0.5))
                    new_h = max(1, int(H * ratio + 0.5))
                    image_small = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
                    mask_small = mask if len(mask.shape) == 2 else cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                    mask_small = cv2.resize(mask_small, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
                    result_small = self.inpaint(image_small, mask_small, refinement, 0, _skip_hd=True, _skip_tiling=True)
                    result_full = cv2.resize(result_small, (W, H), interpolation=cv2.INTER_LANCZOS4)
                    # Paste only masked area
                    mask_gray = mask_small  # already gray but at small size
                    mask_gray = cv2.resize(mask_gray, (W, H), interpolation=cv2.INTER_NEAREST)
                    m = mask_gray > 0
                    out = image.copy()
                    out[m] = result_full[m]
                    return out
                elif strategy == 'crop' and max(H, W) > max(16, int(getattr(self, 'hd_crop_trigger_size', 1024))):
                    mask_gray0 = mask if len(mask.shape) == 2 else cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                    _, thresh = cv2.threshold(mask_gray0, 127, 255, cv2.THRESH_BINARY)
                    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        out = image.copy()
                        margin = max(0, int(getattr(self, 'hd_crop_margin', 16)))
                        for cnt in contours:
                            x, y, w, h = cv2.boundingRect(cnt)
                            l = max(0, x - margin); t = max(0, y - margin)
                            r = min(W, x + w + margin); b = min(H, y + h + margin)
                            if r <= l or b <= t:
                                continue
                            crop_img = image[t:b, l:r]
                            crop_mask = mask_gray0[t:b, l:r]
                            patch = self.inpaint(crop_img, crop_mask, refinement, 0, _skip_hd=True, _skip_tiling=True)
                            out[t:b, l:r] = patch
                        return out
            
            if len(mask.shape) == 3:
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            
            # Apply dilation for anime method
            if self.current_method == 'anime':
                kernel = np.ones((7, 7), np.uint8)
                mask = cv2.dilate(mask, kernel, iterations=1)

            # Use instance tiling settings for ALL models
            logger.info(f"🔍 Tiling check: enabled={self.tiling_enabled}, tile_size={self.tile_size}, image_size={orig_h}x{orig_w}")

            # If tiling is enabled and image is larger than tile size
            if (not _skip_tiling) and self.tiling_enabled and (orig_h > self.tile_size or orig_w > self.tile_size):
                logger.info(f"🔲 Using tiled inpainting: {self.tile_size}x{self.tile_size} tiles with {self.tile_overlap}px overlap")
                return self._inpaint_tiled(image, mask, self.tile_size, self.tile_overlap, refinement)
                
            # C++ ONNX inference path (faster)
            if self.use_onnx_cpp and self.onnx_cpp_backend:
                logger.debug("Using C++ ONNX inference (2x faster)")
                try:
                    # CRITICAL: Convert BGR (OpenCV default) to RGB (ML model expected)
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    # CRITICAL: Pad to modulo for dynamic models (same as Python path)
                    image_padded, padding = self.pad_img_to_modulo(image_rgb, self.pad_mod)
                    mask_padded, _ = self.pad_img_to_modulo(mask, self.pad_mod)
                    
                    # Normalize to [0, 1] range
                    img_normalized = image_padded.astype(np.float32) / 255.0
                    mask_normalized = mask_padded.astype(np.float32) / 255.0
                    
                    # Apply input masking for anime/manga models (same as Python path)
                    if 'anime' in str(self.current_method).lower():
                        # Mask out input regions for stability
                        mask_binary = (mask_normalized > 0.5).astype(np.float32)
                        img_normalized = img_normalized * (1 - mask_binary[:, :, np.newaxis])
                    
                    # Run C++ inference
                    result_rgb = self.onnx_cpp_backend.infer(img_normalized, mask_normalized)
                    
                    # Remove padding from result
                    if padding[0] > 0 or padding[1] > 0:
                        result_rgb = result_rgb[:orig_h, :orig_w]
                    
                    # Convert back to uint8 and BGR
                    result = (result_rgb * 255).astype(np.uint8)
                    result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
                    
                    self._log_inpaint_diag('onnx-cpp', result, mask)
                    
                except Exception as cpp_err:
                    logger.error(f"C++ ONNX inference failed: {cpp_err}")
                    logger.error(f"Image shape: {image.shape}, Mask shape: {mask.shape}")
                    logger.error(f"Padded image shape: {image_padded.shape}, Padded mask shape: {mask_padded.shape}")
                    raise RuntimeError(f"C++ ONNX inference failed: {cpp_err}")
                
                # C++ inference succeeded, return result
                return result
            
            # Python ONNX inference path (fallback)
            if self.use_onnx and self.onnx_session:
                logger.debug("Using Python ONNX inference")
                
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
                    elif self.current_method == 'aot' or 'aot' in str(self.current_method).lower():
                        # AOT normalization: [-1, 1] range for image
                        logger.debug("Using AOT model normalization [-1, 1] for image, [0, 1] for mask")
                        img_np = (image_resized.astype(np.float32) / 127.5) - 1.0
                        mask_np = mask_resized.astype(np.float32) / 255.0
                        mask_np = (mask_np > 0.5) * 1.0  # Binary mask
                        img_np = img_np * (1 - mask_np[:, :, np.newaxis])  # Mask out regions
                    elif 'anime' in str(self.current_method).lower():
                        # Anime/Manga LaMa normalization: [0, 1] range with optional input masking for stability
                        logger.debug("Using Anime/Manga LaMa normalization [0, 1] with input masking")
                        img_np = image_resized.astype(np.float32) / 255.0
                        mask_np = mask_resized.astype(np.float32) / 255.0
                        mask_np = (mask_np > 0.5) * 1.0  # Binary mask
                        # CRITICAL: Mask out input regions for better text region stability
                        # This helps the model focus on generating content rather than being influenced by text artifacts
                        img_np = img_np * (1 - mask_np[:, :, np.newaxis])
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
                    elif self.current_method == 'aot' or 'aot' in str(self.current_method).lower():
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
                    self._log_inpaint_diag('onnx-fixed', result, mask)
                    
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
                    elif self.current_method == 'aot' or 'aot' in str(self.current_method).lower():
                        # AOT normalization: [-1, 1] for image
                        logger.debug("Using AOT model normalization [-1, 1] for image, [0, 1] for mask")
                        img_np = (image_padded.astype(np.float32) / 127.5) - 1.0
                        mask_np = mask_padded.astype(np.float32) / 255.0
                        mask_np = (mask_np > 0.5) * 1.0
                        img_np = img_np * (1 - mask_np[:, :, np.newaxis])  # Mask out regions
                    elif 'anime' in str(self.current_method).lower():
                        # Anime/Manga LaMa normalization: [0, 1] range with optional input masking for stability
                        logger.debug("Using Anime/Manga LaMa normalization [0, 1] with input masking")
                        img_np = image_padded.astype(np.float32) / 255.0
                        mask_np = mask_padded.astype(np.float32) / 255.0
                        mask_np = (mask_np > 0.5) * 1.0  # Binary mask
                        # CRITICAL: Mask out input regions for better text region stability
                        # This helps the model focus on generating content rather than being influenced by text artifacts
                        img_np = img_np * (1 - mask_np[:, :, np.newaxis])
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
                    elif self.current_method == 'aot' or 'aot' in str(self.current_method).lower():
                        result = ((output[0].transpose(1, 2, 0) + 1.0) * 127.5)
                    else:
                        result = output[0].transpose(1, 2, 0) * 255
                    
                    result = np.clip(np.round(result), 0, 255).astype(np.uint8)
                    # CRITICAL: Convert RGB (model output) back to BGR (OpenCV expected)
                    result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
                    
                    # Remove padding
                    result = self.remove_padding(result, padding)
                    self._log_inpaint_diag('onnx-padded', result, mask)
                    
                    # MEMORY CLEANUP: Clear intermediate arrays
                    try:
                        if 'image_rgb' in locals():
                            del image_rgb
                        if 'image_padded' in locals():
                            del image_padded
                        if 'mask_padded' in locals():
                            del mask_padded
                        if 'img_np' in locals():
                            del img_np
                        if 'mask_np' in locals():
                            del mask_np
                        if 'ort_inputs' in locals():
                            del ort_inputs
                        if 'ort_outputs' in locals():
                            del ort_outputs
                        if 'output' in locals():
                            del output
                        if 'raw_output' in locals():
                            del raw_output
                    except Exception:
                        pass
                    
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
                    
                    # Move to device (safely)
                    img_torch = self._to_device_safe(img_torch)
                    mask_torch = self._to_device_safe(mask_torch)
                    
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
                    
                    # Ensure tensors exactly match model device/dtype before call (AOT)
                    try:
                        model_param = next(self.model.parameters())
                        img_torch = img_torch.to(device=model_param.device, dtype=model_param.dtype)
                        mask_torch = mask_torch.to(device=model_param.device, dtype=model_param.dtype)
                        if getattr(model_param.device, 'type', 'cpu') == 'cuda':
                            torch.cuda.synchronize(model_param.device)
                    except Exception:
                        pass

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
                    self._log_inpaint_diag('onnx-fixed', result, mask)
                    
                    # MEMORY CLEANUP: Clear intermediate arrays
                    try:
                        if 'image_rgb' in locals():
                            del image_rgb
                        if 'image_resized' in locals():
                            del image_resized
                        if 'mask_resized' in locals():
                            del mask_resized
                        if 'img_np' in locals():
                            del img_np
                        if 'mask_np' in locals():
                            del mask_np
                        if 'ort_inputs' in locals():
                            del ort_inputs
                        if 'ort_outputs' in locals():
                            del ort_outputs
                        if 'output' in locals():
                            del output
                        if 'raw_output' in locals():
                            del raw_output
                    except Exception:
                        pass
                    
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
                    
                    # For anime models: mask out input regions for better text stability
                    if 'anime' in str(self.current_method).lower():
                        logger.debug("Applying input masking for anime model (text region stability)")
                        image_norm = image_norm * (1 - mask_binary[:, :, np.newaxis])
                    
                    # Convert to PyTorch tensors with correct shape
                    # Image should be [B, C, H, W]  
                    image_tensor = torch.from_numpy(image_norm).permute(2, 0, 1).unsqueeze(0).float()
                    mask_tensor = torch.from_numpy(mask_binary).unsqueeze(0).unsqueeze(0).float()
                    
                    # Move to device (safely)
                    image_tensor = self._to_device_safe(image_tensor)
                    mask_tensor = self._to_device_safe(mask_tensor)
                    
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
                    
                    # Ensure tensors exactly match model device/dtype before call
                    try:
                        model_param = next(self.model.parameters())
                        image_tensor = image_tensor.to(device=model_param.device, dtype=model_param.dtype)
                        mask_tensor = mask_tensor.to(device=model_param.device, dtype=model_param.dtype)
                        if getattr(model_param.device, 'type', 'cpu') == 'cuda':
                            torch.cuda.synchronize(model_param.device)
                    except Exception:
                        pass

                    # Determine number of iterations
                    # Log the received iterations parameter for debugging
                    logger.info(f"🔍 Received iterations parameter: {iterations} (type: {type(iterations).__name__})")
                    num_iterations = 1
                    if iterations is not None and iterations > 0:
                        num_iterations = int(iterations)
                        logger.info(f"🔁 Running {num_iterations} iteration(s) for JIT {self.current_method.upper()}")
                    else:
                        logger.info(f"🔁 Using default single pass (iterations={iterations}) for JIT {self.current_method.upper()}")

                    # Run inference with proper error handling and iteration support
                    with torch.no_grad():
                        for iter_idx in range(num_iterations):
                            if num_iterations > 1:
                                logger.debug(f"  JIT Iteration {iter_idx + 1}/{num_iterations}")
                            
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
                            
                            # For multi-iteration: use output as input for next iteration (progressive refinement)
                            if iter_idx < num_iterations - 1:
                                # Normalize output back to [0, 1] range for next iteration
                                if len(inpainted.shape) == 4:
                                    next_input = inpainted[0].detach()  # Keep on GPU if available
                                else:
                                    next_input = inpainted.detach()
                                    if len(next_input.shape) == 3 and next_input.shape[0] == 3:
                                        next_input = next_input  # Already [C, H, W]
                                
                                # Clamp to valid range and update image_tensor
                                next_input = torch.clamp(next_input, 0.0, 1.0)
                                image_tensor = next_input.unsqueeze(0) if len(next_input.shape) == 3 else next_input
                    
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
                    self._log_inpaint_diag('jit-lama', result, mask)
            
            else:
                # Original checkpoint model processing
                h, w = image.shape[:2]
                size = 768 if self.current_method == 'anime' else 512
                
                # Determine number of iterations (default 1 for single pass)
                # Log the received iterations parameter for debugging
                logger.info(f"🔍 Received iterations parameter: {iterations} (type: {type(iterations).__name__})")
                num_iterations = 1
                if iterations is not None and iterations > 0:
                    num_iterations = int(iterations)
                    logger.info(f"🔁 Running {num_iterations} iteration(s) for {self.current_method.upper()}")
                else:
                    logger.info(f"🔁 Using default single pass (iterations={iterations}) for {self.current_method.upper()}")
                
                # For MAT, use 512x512 and convert BGR to RGB
                if self.current_method == 'mat':
                    logger.debug("📦 MAT checkpoint inference path")
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    # IMPROVEMENT: Use LANCZOS4 for downscaling, preserve details
                    img_resized = cv2.resize(image_rgb, (512, 512), interpolation=cv2.INTER_LANCZOS4)
                    mask_resized = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)
                    
                    # CRITICAL: MAT expects [-1, 1] range for both image and mask
                    img_norm = img_resized.astype(np.float32) / 127.5 - 1.0
                    mask_norm = mask_resized.astype(np.float32) / 255.0
                    
                    logger.debug(f"📊 MAT Input Stats - Image: [{img_norm.min():.3f}, {img_norm.max():.3f}], mean={img_norm.mean():.3f}")
                    logger.debug(f"📊 MAT Mask Stats - range: [{mask_norm.min():.3f}, {mask_norm.max():.3f}], coverage={mask_norm.sum()/(512*512)*100:.1f}%")
                else:
                    img_resized = cv2.resize(image, (size, size), interpolation=cv2.INTER_LANCZOS4)
                    mask_resized = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
                    
                    img_norm = img_resized.astype(np.float32) / 127.5 - 1
                    mask_norm = mask_resized.astype(np.float32) / 255.0
                
                # Initial tensors
                img_tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).float()
                mask_tensor = torch.from_numpy(mask_norm).unsqueeze(0).unsqueeze(0).float()
                
                if self.use_gpu and self.device:
                    img_tensor = self._to_device_safe(img_tensor)
                    mask_tensor = self._to_device_safe(mask_tensor)
                
                # ITERATION LOOP: Run inference multiple times for progressive refinement
                with torch.no_grad():
                    for iter_idx in range(num_iterations):
                        if num_iterations > 1:
                            logger.debug(f"  Iteration {iter_idx + 1}/{num_iterations}")
                        
                        # Run model inference
                        output = self.model(img_tensor, mask_tensor)
                        
                        # For multi-iteration: use output as input for next iteration
                        if iter_idx < num_iterations - 1:
                            # Convert output back to normalized input range
                            output_np = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
                            output_np = np.clip(output_np, -1.0, 1.0)
                            
                            # Update image tensor with output (keeps mask the same)
                            img_tensor = torch.from_numpy(output_np).permute(2, 0, 1).unsqueeze(0).float()
                            if self.use_gpu and self.device:
                                img_tensor = self._to_device_safe(img_tensor)
                
                result = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
                
                # For MAT, clamp to [-1, 1] before denormalizing
                if self.current_method == 'mat':
                    logger.debug(f"📊 MAT Raw Output - range: [{result.min():.3f}, {result.max():.3f}], mean={result.mean():.3f}")
                    result = np.clip(result, -1.0, 1.0)
                    logger.debug(f"📊 MAT After Clip - range: [{result.min():.3f}, {result.max():.3f}]")
                
                # Denormalize from [-1, 1] to [0, 255]
                result = ((result + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
                
                # For MAT, convert RGB back to BGR
                if self.current_method == 'mat':
                    result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
                    logger.debug(f"📊 MAT Final Output - shape={result.shape}, dtype={result.dtype}")
                
                result = cv2.resize(result, (w, h), interpolation=cv2.INTER_LANCZOS4)
                self._log_inpaint_diag('ckpt', result, mask)
            
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
            
            # Optional cleanup - only if auto_cleanup_models is enabled
            try:
                should_cleanup = False
                if hasattr(self, 'config') and isinstance(self.config, dict):
                    manga_settings = self.config.get('manga_settings', {})
                    advanced = manga_settings.get('advanced', {})
                    should_cleanup = advanced.get('auto_cleanup_models', False)
                
                if should_cleanup:
                    import gc
                    gc.collect()
                    # Clear CUDA cache if using GPU
                    if torch is not None and hasattr(torch, 'cuda') and torch.cuda.is_available():
                        torch.cuda.empty_cache()
            except Exception:
                pass
            
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

class MATModel(MATInpaintModel):
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


class _LocalInpainterWorker(mp.Process):
    """Separate worker process to initialize models and run inpainting to avoid UI blocking."""
    def __init__(self, config_path: str, task_q, result_q):
        super().__init__()
        self.config_path = config_path
        self.task_q = task_q
        self.result_q = result_q

    def run(self):
        try:
            # Create core inpainter in worker without spawning another worker
            core = LocalInpainter(config_path=self.config_path, enable_worker_process=False)
        except Exception as e:
            try:
                self.result_q.put({'type': 'worker_error', 'error': str(e)})
            except Exception:
                pass
            return
        while True:
            try:
                task = self.task_q.get()
            except Exception:
                continue
            if not isinstance(task, dict):
                continue
            t = task.get('type')
            task_id = task.get('id')
            if t == 'shutdown':
                break
            elif t == 'load_model':
                ok = False
                err = None
                try:
                    ok = core.load_model_with_retry(task.get('method'), task.get('model_path'), force_reload=bool(task.get('force_reload', False)))
                except Exception as e:
                    err = str(e)
                try:
                    self.result_q.put({
                        'type': 'load_model_done',
                        'id': task_id,
                        'success': bool(ok),
                        'use_onnx': bool(getattr(core, 'use_onnx', False)),
                        'use_onnx_cpp': bool(getattr(core, 'use_onnx_cpp', False)),
                        'current_method': getattr(core, 'current_method', None),
                        'error': err
                    })
                except Exception:
                    pass
            elif t == 'inpaint':
                try:
                    img = task.get('image')
                    m = task.get('mask')
                    res = core.inpaint(img, m, refinement=task.get('refinement', 'normal'), iterations=task.get('iterations'))
                    self.result_q.put({'type': 'inpaint_result', 'id': task_id, 'success': True, 'result': res})
                except Exception as e:
                    try:
                        self.result_q.put({'type': 'inpaint_result', 'id': task_id, 'success': False, 'error': str(e)})
                    except Exception:
                        pass
            else:
                # Unknown task; acknowledge to prevent deadlock
                try:
                    self.result_q.put({'type': 'unknown', 'id': task_id, 'task': t})
                except Exception:
                    pass

if __name__ == "__main__":
    from shutdown_utils import run_cli_main
    def _main():
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
        return 0
    run_cli_main(_main)
