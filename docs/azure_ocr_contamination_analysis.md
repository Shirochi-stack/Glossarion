# Azure OCR Cross-Image Contamination Analysis

**Date:** 2025-10-04  
**Issue:** Text appearing in wrong panels when processing 10 images at once with Azure OCR provider

## Executive Summary

After analyzing the manga translation pipeline code, I've identified **3 HIGH-RISK areas** and **2 MEDIUM-RISK areas** where cross-image contamination could occur when sending 10 images at once with Azure OCR in default settings.

## Critical Findings

### 🔴 HIGH RISK #1: ROI Cache Key Missing Page Hash in Some Paths

**Location:** `manga_translator.py:4267-4269` (Google) and `manga_translator.py:4414-4417` (Azure)

**Problem:** 
The cache fallback logic can create cache keys without the `page_hash`:
```python
ck = roi.get('cache_key') or ("google", page_hash, x, y, w, h, tuple(lang_hints), detection_mode)
```

If `roi.get('cache_key')` is None AND the fallback is used, but `page_hash` is also None (failure in line 2104), the cache key becomes:
```python
("google", None, x, y, w, h, ...)
```

**Impact:** 
- Two different images with the same ROI coordinates would share the same cache entry
- Text from Image #1 could appear in Image #2 if they have overlapping bubble positions
- This is **especially likely in manga** where bubble positions are often consistent across panels

**Evidence in Code:**
- Line 2102-2116: `page_hash` can be None if SHA1 computation fails
- Line 4267: Cache key uses `page_hash` which could be None
- Line 4181: Cache key includes `roi.get('type', 'unknown')` - good, but not enough

**Reproduction Scenario:**
1. Process Image A with bubble at (100, 200, 300, 100) containing "こんにちは"
2. Cache stores: `("azure", None, 100, 200, 300, 100, ...) -> "こんにちは"`
3. Process Image B with bubble at same coordinates
4. Cache returns: "こんにちは" for Image B's bubble (WRONG!)

---

### 🔴 HIGH RISK #2: Parallel Panel Translation with Shared Translator Instance

**Location:** `manga_translator.py:236-241` (Cache initialization)

**Problem:**
While the cache has a lock (`_cache_lock`), if multiple panels/images are processed in parallel using the **same translator instance**, they share:
- `self.ocr_roi_cache` 
- `self._current_image_hash`

The cache clearing logic at line 2044-2046 only clears when processing NEW images through `detect_text_regions()`, but:

**Critical Issue:**
If processing 10 images concurrently and all use the same `MangaTranslator` instance:
1. Thread 1 processes Image A, populates cache
2. Thread 2 starts processing Image B before Thread 1 finishes
3. Thread 2 reads from cache **before** the cache is cleared for Image B
4. Thread 2 gets text from Image A

**Evidence:**
- Line 2108-2112: Cache only cleared when `_current_image_hash` changes
- Line 236: `self.ocr_roi_cache = {}` is instance-level, not thread-local
- Line 241: `self._cache_lock` prevents race conditions but NOT cross-image pollution

**Key Question:** Are you using parallel panel translation feature?
- Check: `manga_settings['advanced']['parallel_panel_translation']`
- If enabled with shared translator → HIGH contamination risk

---

### 🔴 HIGH RISK #3: Azure ROI Concurrent Processing Order Issues

**Location:** `manga_translator.py:4441-4446` (Azure concurrent ROI processing)

**Problem:**
```python
with ThreadPoolExecutor(max_workers=max_workers) as ex:
    fut_map = {ex.submit(ocr_one, r): r for r in work_rois}
    for fut in as_completed(fut_map):
        reg = fut.result()
        if reg is not None:
            results.append(reg)
```

The `as_completed()` iterator returns futures **in completion order, NOT submission order**.

**Impact:**
If you have 10 ROIs from the same image:
1. ROI #1-10 submitted to Azure
2. ROI #7 completes first → appended to `results[0]`
3. ROI #2 completes second → appended to `results[1]`
4. **Final results list is in completion order, not spatial order**

Combined with caching issues, this could cause:
- Text from ROI position (100, 200) to be assigned to different visual position
- Especially problematic if cache keys are colliding across images

**Verification Needed:**
- Check if results are re-sorted by position after this function returns
- If not, text order gets scrambled

---

## 🟡 MEDIUM RISK #4: Cache Type Field Mismatch

**Location:** `manga_translator.py:4181, 4316`

**Problem:**
Cache keys include `roi.get('type', 'unknown')` which can be:
- `'text_bubble'`
- `'free_text'`
- `'unknown'`

If ROI preparation sometimes fails to set type:
```python
# Line 4144-4148
out.append({
    'id': rec['id'],
    'bbox': (x, y, w, h),
    'bytes': img_bytes,
    'type': rec['type'],  # What if rec doesn't have 'type'?
    'page_hash': page_hash
})
```

A cache hit with wrong type could return:
- Text from a speech bubble cached as 'text_bubble'
- Applied to a free text region expecting different text

**Lower severity because:**
- Types are explicitly set in `_prepare_ocr_rois_from_bubbles` (line 4054)
- But edge cases could still occur

---

## 🟡 MEDIUM RISK #5: Page Hash Collision (Low Probability)

**Location:** `manga_translator.py:2104`

**Problem:**
```python
page_hash = hashlib.sha1(processed_image_data).hexdigest()
```

SHA1 collisions are theoretically possible (though astronomically unlikely for 10 images).

More realistic issue: If `processed_image_data` is identical for two different source images (e.g., duplicate detection panels), they'd share cache entries.

---

## Root Cause Analysis

The fundamental issue is **cache lifetime management**:

1. ✅ **Good:** Cache is cleared at start of `detect_text_regions()` (line 2044-2046)
2. ✅ **Good:** Cache uses locks for thread safety (line 2044, 4183, 4268, 4318, 4416)
3. ❌ **BAD:** Cache survives across image processing if same translator instance reused
4. ❌ **BAD:** Cache keys can have `None` page_hash causing false sharing
5. ❌ **BAD:** No per-thread or per-image isolation in parallel processing

---

## Recommended Fixes

### Fix #1: Always Generate Valid Page Hash (Critical)
```python
# Line 2102-2116
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
        
    # CRITICAL: If image hash changed, force clear ROI cache
    if hasattr(self, '_current_image_hash') and self._current_image_hash != page_hash:
        if hasattr(self, 'ocr_roi_cache'):
            with self._cache_lock:
                self.ocr_roi_cache.clear()
            self._log("🧹 Image changed - cleared ROI cache", "debug")
    self._current_image_hash = page_hash
except Exception as e:
    # Emergency fallback - never let page_hash be None
    import uuid
    page_hash = str(uuid.uuid4())
    self._current_image_hash = page_hash
    self._log(f"⚠️ Page hash generation failed: {e}, using UUID fallback", "error")
```

### Fix #2: Clear Cache at END of Processing (Critical)
```python
# In detect_text_regions(), after line 3524 (at end of function, before return)
finally:
    # CRITICAL: Clear cache after processing each image
    # This ensures no text leaks to next image even if translator is reused
    if hasattr(self, 'ocr_roi_cache'):
        with self._cache_lock:
            self.ocr_roi_cache.clear()
        self._log("🧹 Cleared ROI cache after processing", "debug")
```

### Fix #3: Add Image Path to Cache Key (Defense in Depth)
```python
# Line 4316 (Azure) and 4181 (Google)
# Change cache key to include image path as well
cache_key = ("azure", page_hash, image_path, x, y, w, h, reading_order, roi.get('type', 'unknown'))
```

### Fix #4: Sort Results by Position After Concurrent Processing
```python
# After line 4446, before return
# Sort results by bounding box position (top-to-bottom, left-to-right)
results.sort(key=lambda r: (r.bounding_box[1], r.bounding_box[0]))
self._log(f"✅ Sorted {len(results)} results by position", "debug")
return results
```

### Fix #5: Add Cache Statistics Logging
```python
# Add to _azure_ocr_rois_concurrent after line 4333
cache_hits = len(cached_regions)
cache_misses = len(work_rois)
total = cache_hits + cache_misses
if total > 0:
    hit_rate = (cache_hits / total) * 100
    self._log(f"📊 Cache: {cache_hits}/{total} hits ({hit_rate:.1f}%)", "info")
```

---

## Testing Strategy

### Test #1: Identical ROI Positions
1. Create 10 test images with bubbles at EXACT same coordinates
2. Each bubble has different text: "Image 1", "Image 2", etc.
3. Process all 10 simultaneously with Azure OCR
4. Verify each output has correct text (not from other images)

### Test #2: Rapid Sequential Processing
1. Process Image A → verify text
2. **Immediately** process Image B (same translator instance)
3. Check if Image B has any text from Image A

### Test #3: Parallel Panel Translation
1. Enable `parallel_panel_translation` in advanced settings
2. Process multiple pages with overlapping bubble positions
3. Check for text mismapping

### Test #4: Cache Key Uniqueness
Add this debug logging temporarily:
```python
# After line 4316 or 4181
self._log(f"🔑 Cache key: {cache_key}", "debug")
```
Then verify no duplicate keys appear across different images.

---

## Immediate Action Items

1. **Add emergency logging** to track cache behavior:
   ```python
   # In _azure_ocr_rois_concurrent, after line 4319
   if text_cached:
       self._log(f"💾 Cache HIT: {x},{y},{w},{h} type={roi.get('type')} hash={page_hash[:8]}... text={text_cached[:30]}...", "debug")
   ```

2. **Enable debug logging** and process the problematic image set
   - Look for cache hits with mismatched page_hash
   - Look for None page_hash values

3. **Check translator instance reuse**:
   - Search for where `MangaTranslator` is instantiated
   - Verify if same instance is reused across multiple images
   - If yes → HIGH contamination risk confirmed

4. **Review parallel processing settings**:
   - Check `manga_settings['advanced']['parallel_panel_translation']`
   - Check `manga_settings['advanced']['panel_max_workers']`

---

## Questions to Investigate

1. **When you process "10 images at once":**
   - Are you using batch translation feature?
   - Are they processed sequentially or in parallel threads?
   - Is the same `MangaTranslator` instance reused?

2. **The wrong text appearing:**
   - Is it from the previous image in sequence?
   - Or random from any of the 10 images?
   - Does it happen to all 10 images or just some?

3. **Reproducibility:**
   - Does it happen every time with same 10 images?
   - Does changing image order change which text appears where?
   - Does processing images one-by-one avoid the issue?

---

## Conclusion

The most likely culprit is **HIGH RISK #1** (cache key with None page_hash) combined with **HIGH RISK #2** (shared translator instance across images).

The cache is designed to be per-image but fails to maintain isolation due to:
1. Potential None values in cache keys
2. Insufficient clearing between images in batch mode
3. Shared instance state in parallel processing

**Priority Fixes:**
1. ✅ Implement Fix #1 (Never allow None page_hash)
2. ✅ Implement Fix #2 (Clear cache at end of processing)
3. ✅ Implement Fix #5 (Add cache statistics logging)
4. Then test and monitor before implementing other fixes

Would you like me to implement these fixes immediately?
