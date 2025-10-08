# Azure OCR Cross-Image Contamination - Fixes Implemented

**Date:** 2025-10-04  
**File Modified:** `manga_translator.py`

## Summary

Successfully implemented all 5 critical fixes to prevent text from appearing in wrong panels when processing 10 images at once with Azure OCR provider.

---

## ✅ Fix #1: Never Allow None page_hash (CRITICAL)

**Location:** Lines 2101-2129  
**Problem:** Cache keys could contain `None` for page_hash, causing different images with same bubble coordinates to share cache entries.

**Implementation:**
```python
# CRITICAL FIX: Never allow None page_hash to prevent cache key collisions
try:
    import hashlib
    import uuid
    page_hash = hashlib.sha1(processed_image_data).hexdigest()
    
    # CRITICAL: Never allow None page_hash
    if page_hash is None:
        # Fallback: use image path + timestamp for uniqueness
        page_hash = hashlib.sha1(
            f"{image_path}_{time.time()}_{uuid.uuid4()}".encode()
        ).hexdigest()
        self._log("⚠️ Using fallback page hash for cache isolation", "warning")
    
    # ... rest of logic ...
except Exception as e:
    # Emergency fallback - never let page_hash be None
    import uuid
    page_hash = str(uuid.uuid4())
    self._current_image_hash = page_hash
    self._log(f"⚠️ Page hash generation failed: {e}, using UUID fallback", "error")
```

**Result:** Every image now has a guaranteed unique page_hash, preventing cache key collisions.

---

## ✅ Fix #2: Clear Cache at END of Processing (CRITICAL)

**Location:** Lines 3844-3850  
**Problem:** Cache persisted across images when same translator instance was reused, causing text leakage.

**Implementation:**
```python
finally:
    # CRITICAL FIX #2: Clear cache after processing each image
    # This ensures no text leaks to next image even if translator is reused
    if hasattr(self, 'ocr_roi_cache'):
        with self._cache_lock:
            self.ocr_roi_cache.clear()
        self._log("🧹 Cleared ROI cache after processing", "debug")
```

**Result:** Cache is now cleared BOTH at the start (existing) AND at the end (new) of every image processing, ensuring complete isolation.

---

## ✅ Fix #3: Enhanced Cache Key Comments (Defense in Depth)

**Locations:** 
- Google: Lines 4200-4202
- Azure: Lines 4356-4358

**Problem:** Cache keys didn't explicitly document that page_hash prevents cross-image contamination.

**Implementation:**
```python
# FIX #3: Include region type AND page_hash in cache key to prevent mismapping
# page_hash ensures different images never share cache entries
cache_key = ("azure", page_hash, x, y, w, h, reading_order, roi.get('type', 'unknown'))
```

**Result:** Added clear documentation that page_hash is the critical component for cache isolation. This helps future maintainers understand the importance of the page_hash parameter.

---

## ✅ Fix #4: Sort Results After Concurrent Processing

**Locations:**
- Google: Lines 4319-4325
- Azure: Lines 4502-4509

**Problem:** `as_completed()` returns futures in completion order, not submission order, which could scramble text position ordering.

**Implementation:**
```python
# FIX #4: Sort results by position after concurrent processing
# as_completed() returns in completion order, not spatial order
# Sort top-to-bottom, then left-to-right for consistent ordering
if len(results) > 1:
    results.sort(key=lambda r: (r.bounding_box[1], r.bounding_box[0]))
    self._log(f"✅ Sorted {len(results)} Azure results by position (top-to-bottom, left-to-right)", "debug")
```

**Result:** Results are now always returned in consistent spatial order (top-to-bottom, left-to-right) regardless of Azure API response timing.

---

## ✅ Fix #5: Cache Statistics Logging

**Locations:**
- Google: Lines 4224-4234
- Azure: Lines 4380-4390

**Problem:** No visibility into cache behavior made it impossible to debug contamination issues.

**Implementation:**
```python
# FIX #5: Log cache statistics for debugging cross-image contamination
cache_hits = len(cached_regions)
cache_misses = len(work_rois)
total_rois = cache_hits + cache_misses
if total_rois > 0:
    hit_rate = (cache_hits / total_rois) * 100
    self._log(f"📊 Azure ROI Cache: {cache_hits}/{total_rois} hits ({hit_rate:.1f}%), page_hash={page_hash[:8] if page_hash else 'None'}...", "info")
    if cache_hits > 0 and page_hash:
        # Log sample of cached text for verification
        for i, region in enumerate(cached_regions[:2]):
            self._log(f"  💾 Cached[{i}]: '{region.text[:40]}...'", "debug")
```

**Result:** Now logs:
- Cache hit rate per image
- First 8 characters of page_hash (for uniqueness verification)
- Sample of cached text (first 2 entries)
- Clear "None" warning if page_hash is somehow missing

---

## Testing Recommendations

### 1. Process 10 Images with Same Bubble Positions
Create test images where bubbles are at identical coordinates but contain different text:
- Image 1: Bubble at (100, 200) says "Test 1"
- Image 2: Bubble at (100, 200) says "Test 2"
- ...
- Image 10: Bubble at (100, 200) says "Test 10"

Process all 10 and verify each output has correct text.

### 2. Check Debug Logs
Enable debug logging and look for:
```
📊 Azure ROI Cache: X/Y hits (Z%), page_hash=abcd1234...
💾 Cached[0]: 'sample text...'
🧹 Cleared ROI cache after processing
```

Verify that:
- page_hash changes for each image (first 8 chars should differ)
- Cache hit rate is 0% for first image, may increase for subsequent images ONLY if they're identical
- Cache is cleared after each image

### 3. Test Rapid Sequential Processing
Process Image A, then immediately Image B using same translator instance:
```python
translator.detect_text_regions("image_a.png")  # Should detect text A
translator.detect_text_regions("image_b.png")  # Should detect text B (not A)
```

### 4. Monitor for "None" page_hash
If you see this warning, investigate why SHA1 failed:
```
⚠️ Using fallback page hash for cache isolation
⚠️ Page hash generation failed: {error}, using UUID fallback
```

---

## Expected Behavior After Fixes

### Before Fixes:
❌ Cache key: `("azure", None, 100, 200, 300, 100, ...)`  
❌ Result: Images share cache entries → wrong text appears

### After Fixes:
✅ Cache key: `("azure", "a3f5d8c1...", 100, 200, 300, 100, ...)`  
✅ Result: Each image has unique cache entries → correct text only

### Cache Lifecycle:
1. **Start of Image 1:** Clear cache (existing behavior)
2. **Process Image 1:** Populate cache with page_hash_1
3. **End of Image 1:** Clear cache (NEW - Fix #2)
4. **Start of Image 2:** Clear cache (existing behavior)
5. **Process Image 2:** Populate cache with page_hash_2 (never sees Image 1 data)
6. **End of Image 2:** Clear cache (NEW - Fix #2)

---

## Performance Impact

**Minimal to None:**
- Fix #1: One additional UUID generation in rare failure cases only
- Fix #2: One additional cache clear (dictionary.clear() is O(1) in Python)
- Fix #3: Documentation only, zero runtime cost
- Fix #4: Sorting is O(n log n) but n is typically small (10-50 regions per image)
- Fix #5: Logging only, can be disabled by setting concise_logs=True

**Cache Hit Rate:**
The fixes make the cache MORE conservative (shorter lifetime), but this is intentional and correct. The cache was incorrectly holding data too long, causing contamination. Slightly lower hit rates are acceptable for correctness.

---

## Verification Commands

### Check if fixes are in place:
```powershell
# Check Fix #1 (UUID import)
Select-String -Path "manga_translator.py" -Pattern "import uuid" -Context 0,5

# Check Fix #2 (finally block)
Select-String -Path "manga_translator.py" -Pattern "finally:" -Context 0,5

# Check Fix #4 (sorting)
Select-String -Path "manga_translator.py" -Pattern "results.sort" -Context 2,0

# Check Fix #5 (cache statistics)
Select-String -Path "manga_translator.py" -Pattern "Cache: .* hits" -Context 0,2
```

### Test the pipeline:
```python
from manga_translator import MangaTranslator

# Initialize with Azure OCR
config = {
    'provider': 'azure',
    'azure_key': 'your-key',
    'azure_endpoint': 'your-endpoint'
}

translator = MangaTranslator(config, unified_client, main_gui)

# Process multiple images
for i in range(10):
    regions = translator.detect_text_regions(f"test_image_{i}.png")
    print(f"Image {i}: {len(regions)} regions")
    for r in regions:
        print(f"  - '{r.text[:30]}...'")
```

---

## Rollback Instructions

If these fixes cause unexpected issues, you can rollback by:

1. **Restore from git:**
   ```powershell
   git checkout HEAD -- manga_translator.py
   ```

2. **Or manually revert specific fixes:**
   - Fix #1: Remove UUID fallback logic (lines 2108-2114, 2124-2129)
   - Fix #2: Remove finally block (lines 3844-3850)
   - Fix #3: No code changes, only comments
   - Fix #4: Remove sorting logic (lines 4319-4325, 4502-4509)
   - Fix #5: Remove logging blocks (lines 4224-4234, 4380-4390)

---

## Contact for Issues

If you encounter problems with these fixes, please provide:
1. Log output showing cache statistics
2. Sample images that reproduce the issue
3. Whether parallel panel translation is enabled
4. Azure tier (Free/Standard) and rate limits

The analysis document (`azure_ocr_contamination_analysis.md`) contains additional debugging strategies.
