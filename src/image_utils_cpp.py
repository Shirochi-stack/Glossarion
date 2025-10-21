"""
Python wrapper for C++ image utilities
Fast image operations for improved performance
"""

import ctypes
import numpy as np
import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Global library instance
_lib = None
_lib_available = False


def _load_library():
    """Load the C++ shared library"""
    global _lib, _lib_available
    
    if _lib is not None:
        return _lib_available
    
    # Try to find the library
    possible_names = [
        'image_utils.dll',  # Windows (MSVC)
        'libimage_utils.dll',  # Windows (MinGW)
        'libimage_utils.so',  # Linux
        'libimage_utils.dylib',  # Mac
    ]
    
    # Search in current directory and parent directory
    search_dirs = [
        Path(__file__).parent,
        Path(__file__).parent.parent,
        Path.cwd(),
    ]
    
    lib_path = None
    for search_dir in search_dirs:
        for lib_name in possible_names:
            candidate = search_dir / lib_name
            if candidate.exists():
                lib_path = str(candidate)
                break
        if lib_path:
            break
    
    if not lib_path:
        logger.warning("C++ image utils not available - using Python fallback")
        _lib_available = False
        return False
    
    try:
        _lib = ctypes.CDLL(lib_path)
        _setup_function_signatures()
        logger.info(f"✓ Loaded C++ image utils from: {lib_path}")
        _lib_available = True
        return True
    except Exception as e:
        logger.warning(f"Failed to load C++ image utils: {e} - using Python fallback")
        _lib_available = False
        return False


def _setup_function_signatures():
    """Define C function signatures"""
    global _lib
    
    # void rgb_to_bgr_inplace(uint8_t* data, int width, int height, int channels)
    _lib.rgb_to_bgr_inplace.argtypes = [
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int
    ]
    _lib.rgb_to_bgr_inplace.restype = None
    
    # void rgb_to_bgr_copy(const uint8_t* src, uint8_t* dst, int w, int h, int c)
    _lib.rgb_to_bgr_copy.argtypes = [
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int
    ]
    _lib.rgb_to_bgr_copy.restype = None
    
    # void resize_bilinear(const uint8_t* src, uint8_t* dst, ...)
    _lib.resize_bilinear.argtypes = [
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_int, ctypes.c_int,  # src_w, src_h
        ctypes.c_int, ctypes.c_int,  # dst_w, dst_h
        ctypes.c_int  # channels
    ]
    _lib.resize_bilinear.restype = None
    
    # void resize_nearest(const uint8_t* src, uint8_t* dst, ...)
    _lib.resize_nearest.argtypes = [
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_int, ctypes.c_int,  # src_w, src_h
        ctypes.c_int, ctypes.c_int,  # dst_w, dst_h
        ctypes.c_int  # channels
    ]
    _lib.resize_nearest.restype = None
    
    # void alpha_blend(uint8_t* base, const uint8_t* overlay, ...)
    _lib.alpha_blend.argtypes = [
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_int, ctypes.c_int,  # width, height
        ctypes.c_int, ctypes.c_int   # base_channels, overlay_channels
    ]
    _lib.alpha_blend.restype = None
    
    # const char* image_utils_version()
    _lib.image_utils_version.argtypes = []
    _lib.image_utils_version.restype = ctypes.c_char_p


# Initialize library on import
_load_library()


def is_available() -> bool:
    """Check if C++ image utils are available"""
    return _lib_available


def convert_rgb_bgr(image: np.ndarray, inplace: bool = False) -> np.ndarray:
    """
    Fast RGB <-> BGR conversion using C++
    
    Args:
        image: Input image (H, W, C), uint8
        inplace: If True, modify in-place. Otherwise create copy.
    
    Returns:
        Converted image
    """
    if not _lib_available:
        # Python fallback
        if inplace:
            image[:, :, [0, 2]] = image[:, :, [2, 0]]
            return image
        else:
            return image[:, :, ::-1].copy()
    
    if image.dtype != np.uint8:
        raise ValueError("Image must be uint8")
    
    h, w, c = image.shape
    
    if inplace:
        # In-place conversion
        data_ptr = image.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        _lib.rgb_to_bgr_inplace(data_ptr, w, h, c)
        return image
    else:
        # Copy conversion
        output = np.empty_like(image)
        src_ptr = image.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        dst_ptr = output.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        _lib.rgb_to_bgr_copy(src_ptr, dst_ptr, w, h, c)
        return output


def resize_image(image: np.ndarray, size: tuple, method: str = 'bilinear') -> np.ndarray:
    """
    Fast image resize using C++
    
    Args:
        image: Input image (H, W, C), uint8
        size: Target size as (width, height)
        method: 'bilinear' or 'nearest'
    
    Returns:
        Resized image
    """
    if not _lib_available:
        # Python fallback using PIL
        from PIL import Image
        pil_img = Image.fromarray(image)
        resample = Image.BILINEAR if method == 'bilinear' else Image.NEAREST
        resized = pil_img.resize(size, resample)
        return np.array(resized)
    
    if image.dtype != np.uint8:
        raise ValueError("Image must be uint8")
    
    h, w, c = image.shape
    dst_w, dst_h = size
    
    output = np.empty((dst_h, dst_w, c), dtype=np.uint8)
    
    src_ptr = image.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
    dst_ptr = output.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
    
    if method == 'bilinear':
        _lib.resize_bilinear(src_ptr, dst_ptr, w, h, dst_w, dst_h, c)
    elif method == 'nearest':
        _lib.resize_nearest(src_ptr, dst_ptr, w, h, dst_w, dst_h, c)
    else:
        raise ValueError(f"Unknown resize method: {method}")
    
    return output


def blend_images(base: np.ndarray, overlay: np.ndarray) -> np.ndarray:
    """
    Fast alpha blending using C++
    
    Args:
        base: Base image (H, W, 3 or 4), uint8
        overlay: Overlay image with alpha (H, W, 4), uint8
    
    Returns:
        Blended image (modifies base in-place)
    """
    if not _lib_available:
        # Python fallback
        if overlay.shape[2] == 4:
            alpha = overlay[:, :, 3:4].astype(np.float32) / 255.0
            base[:, :, :3] = (
                base[:, :, :3] * (1 - alpha) + 
                overlay[:, :, :3] * alpha
            ).astype(np.uint8)
        else:
            base[:, :, :3] = overlay[:, :, :3]
        return base
    
    if base.dtype != np.uint8 or overlay.dtype != np.uint8:
        raise ValueError("Images must be uint8")
    
    h, w = base.shape[:2]
    base_c = base.shape[2]
    overlay_c = overlay.shape[2]
    
    base_ptr = base.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
    overlay_ptr = overlay.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
    
    _lib.alpha_blend(base_ptr, overlay_ptr, w, h, base_c, overlay_c)
    
    return base


def get_version() -> str:
    """Get C++ library version"""
    if not _lib_available:
        return "unavailable"
    
    try:
        version = _lib.image_utils_version()
        return version.decode('utf-8') if version else "unknown"
    except Exception:
        return "unknown"
