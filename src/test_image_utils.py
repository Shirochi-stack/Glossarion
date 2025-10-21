"""Test C++ image utilities performance"""

import numpy as np
import time
import cv2
from image_utils_cpp import is_available, convert_rgb_bgr, resize_image, get_version

print("=" * 60)
print("C++ Image Utils Test")
print("=" * 60)

# Check if library is available
if not is_available():
    print("✗ C++ library not available - will use Python fallback")
else:
    print(f"✓ C++ library loaded - version {get_version()}")

print()

# Test image
print("Creating test image (1920x1080)...")
test_img = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)

# Test 1: RGB/BGR conversion
print("\n" + "=" * 60)
print("Test 1: RGB <-> BGR Conversion")
print("=" * 60)

# Python version
start = time.perf_counter()
for _ in range(100):
    _ = test_img[:, :, ::-1].copy()
python_time = time.perf_counter() - start

# C++ version
start = time.perf_counter()
for _ in range(100):
    _ = convert_rgb_bgr(test_img, inplace=False)
cpp_time = time.perf_counter() - start

print(f"Python:  {python_time*1000:.2f}ms (100 iterations)")
print(f"C++:     {cpp_time*1000:.2f}ms (100 iterations)")
print(f"Speedup: {python_time/cpp_time:.2f}x faster")

# Test 2: Image resizing
print("\n" + "=" * 60)
print("Test 2: Image Resize (1920x1080 -> 640x360)")
print("=" * 60)

# Python version (cv2)
start = time.perf_counter()
for _ in range(100):
    _ = cv2.resize(test_img, (640, 360))
python_time = time.perf_counter() - start

# C++ version
start = time.perf_counter()
for _ in range(100):
    _ = resize_image(test_img, (640, 360), method='bilinear')
cpp_time = time.perf_counter() - start

print(f"Python:  {python_time*1000:.2f}ms (100 iterations)")
print(f"C++:     {cpp_time*1000:.2f}ms (100 iterations)")
print(f"Speedup: {python_time/cpp_time:.2f}x faster")

# Test 3: Nearest neighbor (thumbnails)
print("\n" + "=" * 60)
print("Test 3: Nearest Resize (1920x1080 -> 100x100)")
print("=" * 60)

# Python version (cv2)
start = time.perf_counter()
for _ in range(100):
    _ = cv2.resize(test_img, (100, 100), interpolation=cv2.INTER_NEAREST)
python_time = time.perf_counter() - start

# C++ version
start = time.perf_counter()
for _ in range(100):
    _ = resize_image(test_img, (100, 100), method='nearest')
cpp_time = time.perf_counter() - start

print(f"Python:  {python_time*1000:.2f}ms (100 iterations)")
print(f"C++:     {cpp_time*1000:.2f}ms (100 iterations)")
print(f"Speedup: {python_time/cpp_time:.2f}x faster")

print("\n" + "=" * 60)
print("✓ All tests completed!")
print("=" * 60)
