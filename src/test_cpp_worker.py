"""
Test script to verify C++ ONNX backend works in worker process
"""
import logging
import numpy as np
from local_inpainter import LocalInpainter

# Enable detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_worker_backend():
    print("=" * 70)
    print("Testing C++ ONNX Backend in Worker Process")
    print("=" * 70)
    
    # Test 1: Create inpainter with worker enabled (default on Windows)
    print("\n[Test 1] Creating inpainter with worker process...")
    inpainter = LocalInpainter(enable_worker_process=True)
    
    # Check if worker was started
    if inpainter._mp_enabled:
        print("✅ Worker process started successfully")
    else:
        print("⚠️ Worker process not enabled (falling back to in-process)")
    
    # Test 2: Load ONNX model through worker
    print("\n[Test 2] Loading ONNX model through worker...")
    print("Note: You need an ONNX model file for this test")
    print("Example: models/aot.onnx or models/lama.onnx")
    
    # You can uncomment and modify this to test with your actual model:
    # success = inpainter.load_model('aot_onnx', 'path/to/your/model.onnx')
    # if success:
    #     print(f"✅ Model loaded in worker")
    #     print(f"   use_onnx: {inpainter.use_onnx}")
    #     print(f"   use_onnx_cpp: {inpainter.use_onnx_cpp}")
    #     if inpainter.use_onnx_cpp:
    #         print("🚀 C++ ONNX backend is active in worker process!")
    #     else:
    #         print("📦 Using Python ONNX (C++ not available or DLL not built)")
    
    # Test 3: Test in-process mode (no worker)
    print("\n[Test 3] Creating inpainter WITHOUT worker (direct mode)...")
    inpainter_direct = LocalInpainter(enable_worker_process=False)
    
    # Uncomment to test direct loading:
    # success = inpainter_direct.load_model('aot_onnx', 'path/to/your/model.onnx')
    # if success:
    #     print(f"✅ Model loaded directly (no worker)")
    #     print(f"   use_onnx: {inpainter_direct.use_onnx}")
    #     print(f"   use_onnx_cpp: {inpainter_direct.use_onnx_cpp}")
    #     if inpainter_direct.use_onnx_cpp:
    #         print("🚀 C++ ONNX backend is active in direct mode!")
    
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    print("Worker process:", "✅ Enabled" if inpainter._mp_enabled else "⚠️ Disabled")
    print("\nTo fully test C++ backend:")
    print("1. Build the C++ DLL: .\\build_onnx_cpp.bat")
    print("2. Uncomment the test code above")
    print("3. Provide a valid ONNX model path")
    print("4. Look for: '✅ Worker process using C++ ONNX backend (2x faster)'")
    
    # Cleanup
    inpainter.unload()
    inpainter_direct.unload()

def test_backend_detection():
    """Test if C++ backend is available at module level"""
    print("\n" + "=" * 70)
    print("C++ Backend Availability Check")
    print("=" * 70)
    
    try:
        from onnx_cpp_backend import is_cpp_backend_available
        available = is_cpp_backend_available()
        print(f"C++ ONNX backend: {'✅ Available' if available else '❌ Not available'}")
        
        if not available:
            print("\nTo enable C++ backend:")
            print("1. Download ONNX Runtime from:")
            print("   https://github.com/microsoft/onnxruntime/releases")
            print("2. Extract to: C:\\Users\\{you}\\onnxruntime")
            print("3. Set environment variable:")
            print("   $env:ONNXRUNTIME_DIR = \"$env:USERPROFILE\\onnxruntime\"")
            print("4. Build: .\\build_onnx_cpp.bat")
        else:
            print("✅ C++ backend ready to use!")
            print("   Worker processes will automatically use it")
            
    except Exception as e:
        print(f"❌ C++ backend not available: {e}")
        print("   Run build script: .\\build_onnx_cpp.bat")

if __name__ == "__main__":
    test_backend_detection()
    print()
    test_worker_backend()
