# Region Mapping Cache Issue - Free Text/Bubble Mismapping

## Problem
Free text regions are being mismapped or overlapping with speech bubbles, causing incorrect rendering and inpainting behavior.

## Root Cause Analysis

### The Issue
The OCR ROI cache stores text results but **also implicitly relies on the ROI's `type` metadata** to assign `bubble_type` to cached regions. This creates potential for mismapping:

1. **ROI Type Assignment** (line 3752):
   ```python
   regions.append({'type': 'text_bubble' if key == 'text_bubbles' else 'free_text',
                   'bbox': (int(bx), int(by), int(bw), int(bh)),
                   'id': f"{key}_{i}"})
   ```

2. **ROI Passed to Cache** (line 3846):
   ```python
   out.append({
       'id': rec['id'],
       'bbox': (x, y, w, h),
       'bytes': img_bytes,
       'type': rec['type'],  # ⚠️ Type stored with ROI
       'page_hash': page_hash
   })
   ```

3. **Cached Region Retrieval** (lines 3891, 3977, 4025):
   ```python
   # When reading from cache
   region.bubble_type = 'free_text' if roi.get('type') == 'free_text' else 'text_bubble'
   ```

### The Problem
The cache key is built from:
```python
cache_key = ("google", page_hash, x, y, w, h, tuple(lang_hints), detection_mode)
```

**Notice**: The `type` field is NOT part of the cache key!

This means:
- If two different images have overlapping bounding boxes at the same coordinates
- Or if the page hash doesn't change properly between images
- Or if RT-DETR detects the same coordinates differently (bubble vs free text)
- **The wrong `type` metadata can be associated with cached OCR text**

### Example Scenario
```
Image 1: Detects region (100, 200, 50, 30) as 'text_bubble'
  → Cache stores: key=(hash1, 100, 200, 50, 30) → text="Hello"
  → ROI has type='text_bubble'
  → Region gets bubble_type='text_bubble' ✓

Image 2: Same coordinates but RT-DETR detects as 'free_text'
  → Cache hit with same key!
  → Retrieves text="Hello" 
  → But ROI now has type='free_text'
  → Region gets bubble_type='free_text' ❌ WRONG!
  
Or worse:
  → Cache still has old ROI with type='text_bubble'
  → Region gets bubble_type='text_bubble' ❌ WRONG!
```

## Current Cache Clearing

The code does clear the cache on image hash changes (line 2067-2071):
```python
if hasattr(self, '_current_image_hash') and self._current_image_hash != page_hash:
    if hasattr(self, 'ocr_roi_cache'):
        with self._cache_lock:
            self.ocr_roi_cache.clear()
```

**BUT**: This only works if `page_hash` changes correctly between images. If:
- Images have identical content
- Or hashing fails
- Or cache isn't cleared properly between translations

The wrong type metadata persists.

## Solutions

### Option 1: Include Type in Cache Key ✅ RECOMMENDED
```python
# Change cache key to include region type
cache_key = ("google", page_hash, x, y, w, h, tuple(lang_hints), detection_mode, roi.get('type', 'unknown'))
```

**Pros:**
- Separates cache entries for bubbles vs free text at same coordinates
- Clean separation of concerns
- No false cache hits

**Cons:**
- Slightly more cache entries

### Option 2: Don't Cache Type Metadata
```python
# Remove type from ROI dict, determine bubble_type from fresh RT-DETR detection each time
region.bubble_type = determine_fresh_type(region, current_detections)
```

**Pros:**
- Always uses current detection results
- No stale metadata

**Cons:**
- Loses type information from original detection
- More complex logic

### Option 3: Clear Cache More Aggressively ⚠️ PARTIAL FIX
```python
# Clear cache at the start of every image translation
self.ocr_roi_cache.clear()
```

**Pros:**
- Guaranteed fresh for each image

**Cons:**
- Loses caching benefits within same image (multiple panels)
- Doesn't solve coordinate collision between different images

## Recommended Fix

Implement **Option 1** - include region type in cache key:

```python
# In _google_ocr_rois_batched (line 3878):
cache_key = ("google", page_hash, x, y, w, h, tuple(lang_hints), detection_mode, roi.get('type', 'unknown'))

# In _azure_ocr_rois_concurrent (line 4012):
cache_key = ("azure", page_hash, x, y, w, h, reading_order, roi.get('type', 'unknown'))
```

This ensures:
1. ✅ Bubble text and free text at same coordinates get separate cache entries
2. ✅ No false cache hits between different region types
3. ✅ Type metadata always matches cached text
4. ✅ Maintains caching benefits

## Verification

After fix, test:
1. ✅ Translate image with bubbles and free text at similar coordinates
2. ✅ Translate second image with different classifications
3. ✅ Verify bubble_type matches current RT-DETR detections
4. ✅ Check mask creation logs show correct region type breakdown
5. ✅ Verify rendering applies correct background opacity per type

## Related Code Locations

- **ROI Type Assignment**: Line 3752 (`_prepare_ocr_rois_from_bubbles`)
- **ROI Dict Creation**: Line 3842-3848
- **Google Cache Key**: Line 3878 (`_google_ocr_rois_batched`)
- **Google Cached Region**: Line 3883-3895
- **Azure Cache Key**: Line 4012 (`_azure_ocr_rois_concurrent`)
- **Azure Cached Region**: Line 4017-4029
- **Bubble Type Usage**: Line 5916-5923 (`create_text_mask`)

---

**Date**: 2025-10-04  
**Issue**: Free text/bubble region type mismapping due to incomplete cache key + missing bubble_type in RT-DETR guided OCR  
**Status**: ✅ **FIXED**  
**Priority**: HIGH - Affects rendering and inpainting accuracy

---

## Fix Implemented ✅

### 1. Added Region Type to Cache Keys
Fixed both Google and Azure cache keys to include region type, preventing cache collisions.

### 2. Fixed RT-DETR Guided OCR Path  
The RT-DETR guided OCR (when ROI locality is disabled) was completely losing region type information. Now properly tracks and assigns bubble_type.

### 3. Enhanced Logging
OCR logs now show region types: `✅ Region 1/5 (free_text): Hello world...`
