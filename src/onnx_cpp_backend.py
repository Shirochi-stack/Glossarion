"""
Python wrapper for C++ ONNX Runtime backend
Lightweight ctypes interface for high-performance inference
"""

import ctypes
import numpy as np
import os
import sys
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ONNXCppBackend:
    """Python wrapper for C++ ONNX Runtime inference"""
    
    def __init__(self):
        self._lib = None
        self._session = None
        self._load_library()
    
    def _load_library(self):
        """Load the C++ shared library"""
        # Try to find the library
        possible_names = [
            'onnx_inpainter.dll',  # Windows (MSVC)
            'libonnx_inpainter.dll',  # Windows (MinGW)
            'libonnx_inpainter.so',  # Linux
            'libonnx_inpainter.dylib',  # Mac
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
            raise RuntimeError(
                f"Could not find ONNX C++ library. Tried: {possible_names}\n"
                f"Please build the library first using CMake:\n"
                f"  cd src && cmake -B build && cmake --build build"
            )
        
        try:
            self._lib = ctypes.CDLL(lib_path)
            logger.info(f"✓ Loaded ONNX C++ backend from: {lib_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load {lib_path}: {e}")
        
        # Define function signatures
        self._setup_function_signatures()
    
    def _setup_function_signatures(self):
        """Define C function signatures"""
        # int onnx_init()
        self._lib.onnx_init.argtypes = []
        self._lib.onnx_init.restype = ctypes.c_int
        
        # ONNXInpainter* onnx_create_session(const char* model_path, int use_gpu)
        self._lib.onnx_create_session.argtypes = [ctypes.c_char_p, ctypes.c_int]
        self._lib.onnx_create_session.restype = ctypes.c_void_p
        
        # int onnx_infer(ONNXInpainter*, float* image, float* mask, 
        #                int batch, int channels, int height, int width, float* output)
        self._lib.onnx_infer.argtypes = [
            ctypes.c_void_p,  # session
            ctypes.POINTER(ctypes.c_float),  # image_data
            ctypes.POINTER(ctypes.c_float),  # mask_data
            ctypes.c_int,  # batch
            ctypes.c_int,  # channels
            ctypes.c_int,  # height
            ctypes.c_int,  # width
            ctypes.POINTER(ctypes.c_float),  # output_data
        ]
        self._lib.onnx_infer.restype = ctypes.c_int
        
        # int onnx_get_input_shape(ONNXInpainter*, int* shape_out)
        self._lib.onnx_get_input_shape.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int)
        ]
        self._lib.onnx_get_input_shape.restype = ctypes.c_int
        
        # void onnx_destroy_session(ONNXInpainter*)
        self._lib.onnx_destroy_session.argtypes = [ctypes.c_void_p]
        self._lib.onnx_destroy_session.restype = None
        
        # const char* onnx_version()
        self._lib.onnx_version.argtypes = []
        self._lib.onnx_version.restype = ctypes.c_char_p
        
        # int onnx_detect_bubbles(ONNXInpainter*, float* image, int h, int w,
        #                         int orig_w, int orig_h, Detection* det, int max, int* num)
        self._lib.onnx_detect_bubbles.argtypes = [
            ctypes.c_void_p,  # session
            ctypes.POINTER(ctypes.c_float),  # image_data
            ctypes.c_int,  # height
            ctypes.c_int,  # width
            ctypes.c_int,  # orig_width
            ctypes.c_int,  # orig_height
            ctypes.c_void_p,  # detections array
            ctypes.c_int,  # max_detections
            ctypes.POINTER(ctypes.c_int),  # num_detections
        ]
        self._lib.onnx_detect_bubbles.restype = ctypes.c_int
    
    def get_version(self) -> str:
        """Get ONNX Runtime version"""
        try:
            version = self._lib.onnx_version()
            return version.decode('utf-8') if version else "unknown"
        except Exception:
            return "unknown"
    
    def load_model(self, model_path: str, use_gpu: bool = False) -> bool:
        """Load ONNX model from file"""
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            return False
        
        try:
            # Initialize ONNX Runtime
            if self._lib.onnx_init() != 0:
                logger.error("Failed to initialize ONNX Runtime")
                return False
            
            # Create session
            model_path_bytes = model_path.encode('utf-8')
            self._session = self._lib.onnx_create_session(
                model_path_bytes,
                1 if use_gpu else 0
            )
            
            if not self._session:
                logger.error("Failed to create ONNX session")
                return False
            
            # Get input shape info
            shape = (ctypes.c_int * 4)()
            is_dynamic = self._lib.onnx_get_input_shape(self._session, shape)
            
            logger.info(f"Model input shape: [{shape[0]}, {shape[1]}, {shape[2]}, {shape[3]}]")
            if is_dynamic:
                logger.info("Model supports dynamic input sizes")
            else:
                logger.info(f"Model expects fixed size: {shape[2]}x{shape[3]}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def infer(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Run inference on image and mask
        
        Args:
            image: RGB image [H, W, 3] or [C, H, W], float32, normalized [0, 1]
            mask: Binary mask [H, W] or [H, W, 1] or [1, H, W], float32
        
        Returns:
            Inpainted image [H, W, 3], float32, normalized [0, 1]
        """
        if not self._session:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Normalize inputs
        orig_shape = image.shape
        
        # Convert to float32
        image = image.astype(np.float32)
        mask = mask.astype(np.float32)
        
        # Handle different input formats
        if len(image.shape) == 3 and image.shape[2] == 3:
            # HWC -> CHW
            image = np.transpose(image, (2, 0, 1))
        
        if len(mask.shape) == 2:
            # HW -> 1HW
            mask = mask[np.newaxis, ...]
        elif len(mask.shape) == 3 and mask.shape[2] == 1:
            # HW1 -> 1HW
            mask = np.transpose(mask, (2, 0, 1))
        
        # Add batch dimension if needed
        if len(image.shape) == 3:
            image = image[np.newaxis, ...]  # 1CHW
        if len(mask.shape) == 3:
            mask = mask[np.newaxis, ...]  # 11HW
        
        batch, channels, height, width = image.shape
        
        # Ensure contiguous memory
        image = np.ascontiguousarray(image, dtype=np.float32)
        mask = np.ascontiguousarray(mask, dtype=np.float32)
        
        # Allocate output buffer
        output = np.zeros_like(image, dtype=np.float32)
        
        # Get pointers
        image_ptr = image.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        mask_ptr = mask.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        output_ptr = output.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        
        # Run inference
        result = self._lib.onnx_infer(
            self._session,
            image_ptr,
            mask_ptr,
            batch,
            channels,
            height,
            width,
            output_ptr
        )
        
        if result != 0:
            raise RuntimeError("Inference failed")
        
        # Convert back to original format
        output = output[0]  # Remove batch dimension
        
        if len(orig_shape) == 3 and orig_shape[2] == 3:
            # CHW -> HWC
            output = np.transpose(output, (1, 2, 0))
        
        return output
    
    def unload(self):
        """Release model resources"""
        # WORKAROUND: Skip cleanup to avoid access violation bug in C++ allocator->Free()
        # The session cleanup has a bug where it calls allocator->Free() incorrectly
        # Until the DLL is rebuilt with the fix, we skip cleanup (OS will clean up on exit)
        if self._session:
            # self._lib.onnx_destroy_session(self._session)  # Commented out - causes access violation
            self._session = None
            logger.info("✓ ONNX C++ session marked for cleanup (skipped buggy destroy to avoid crash)")
    
    def detect_bubbles(self, image: np.ndarray, confidence: float = 0.3) -> dict:
        """
        Run RT-DETR bubble detection
        
        Args:
            image: RGB image [H, W, 3], float32, normalized [0, 1]
            confidence: Minimum confidence threshold
        
        Returns:
            Dictionary with 'bubbles', 'text_bubbles', 'text_free' lists
            Each item is (x, y, width, height) tuple
        """
        if not self._session:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Store original size
        orig_h, orig_w = image.shape[:2]
        
        # Resize to 640x640
        from PIL import Image
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        
        # Convert to PIL for resizing
        image_pil = Image.fromarray((image * 255).astype(np.uint8))
        image_resized = image_pil.resize((640, 640))
        
        # Convert to float32 CHW format
        image_arr = np.array(image_resized).astype(np.float32) / 255.0
        image_chw = np.transpose(image_arr, (2, 0, 1))  # HWC -> CHW
        image_chw = np.ascontiguousarray(image_chw, dtype=np.float32)
        
        # Prepare output buffer (max 300 detections)
        max_det = 300
        
        # Define Detection structure (must match C structure)
        class Detection(ctypes.Structure):
            _fields_ = [
                ('x1', ctypes.c_float),
                ('y1', ctypes.c_float),
                ('x2', ctypes.c_float),
                ('y2', ctypes.c_float),
                ('score', ctypes.c_float),
                ('label', ctypes.c_int)
            ]
        
        detections = (Detection * max_det)()
        num_det = ctypes.c_int(0)
        
        # Get pointer to image data
        image_ptr = image_chw.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        
        # Call C function
        result = self._lib.onnx_detect_bubbles(
            self._session,
            image_ptr,
            640,  # height
            640,  # width
            orig_w,  # original width
            orig_h,  # original height
            detections,
            max_det,
            ctypes.byref(num_det)
        )
        
        if result != 0:
            raise RuntimeError("Bubble detection failed")
        
        # Parse results and group by class
        results = {
            'bubbles': [],        # Class 0: empty bubbles
            'text_bubbles': [],   # Class 1: text in bubbles
            'text_free': []       # Class 2: free text
        }
        
        class_names = ['bubbles', 'text_bubbles', 'text_free']
        
        for i in range(num_det.value):
            det = detections[i]
            
            if det.score < confidence:
                continue
            
            # Convert to (x, y, width, height) format
            x = int(det.x1)
            y = int(det.y1)
            w = int(det.x2 - det.x1)
            h = int(det.y2 - det.y1)
            
            # Group by class
            if det.label >= 0 and det.label < len(class_names):
                results[class_names[det.label]].append((x, y, w, h))
        
        return results
    
    def __del__(self):
        """Cleanup on destruction"""
        # WORKAROUND: Skip cleanup to avoid access violation bug in C++ allocator->Free()
        # The OS will clean up resources when process exits anyway
        pass
        # self.unload()  # Commented out - causes access violation


# Convenience function to check if C++ backend is available
def is_cpp_backend_available() -> bool:
    """Check if C++ ONNX backend is available"""
    try:
        backend = ONNXCppBackend()
        backend.unload()
        return True
    except:
        return False


if __name__ == "__main__":
    # Test the backend
    print("Testing ONNX C++ Backend")
    print("-" * 50)
    
    try:
        backend = ONNXCppBackend()
        version = backend.get_version()
        print(f"✓ ONNX Runtime version: {version}")
        print("✓ C++ backend loaded successfully")
        
        # Test with dummy data
        print("\nTesting inference with dummy data...")
        if backend.load_model("test.onnx", use_gpu=False):
            image = np.random.rand(512, 512, 3).astype(np.float32)
            mask = np.random.rand(512, 512).astype(np.float32)
            result = backend.infer(image, mask)
            print(f"✓ Inference successful, output shape: {result.shape}")
        
    except Exception as e:
        print(f"✗ Error: {e}")
