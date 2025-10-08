# Region Mapping Fix - Complete Summary

## Problem
Free text regions were being mismapped or overlapping with speech bubbles, causing incorrect rendering and inpainting behavior. This was happening **even with ROI locality disabled**.

## Root Causes Found

### 1. Missing bubble_type in RT-DETR Guided OCR ❌
**Location**: Lines 2088-2218 (Google), 2414-2554 (Azure)

When RT-DETR guided OCR is enabled (the default setting now):
- RT-DETR detects separate `text_bubbles` and `text_free` regions
- These are combined into a single `all_regions` list
- **TYPE INFORMATION WAS LOST** when combining
- Regions created without `bubble_type` attribute
- Result: All text treated the same way during masking/rendering

### 2. Incomplete Cache Key in ROI Path ❌
**Location**: Lines 3878 (Google), 4012 (Azure)

Cache keys didn't include region type:
```python
# OLD (BROKEN):
cache_key = ("google", page_hash, x, y, w, h, tuple(lang_hints), detection_mode)
```

This meant:
- Bubble and free text at same coordinates would collide
- Wrong type metadata could be retrieved from cache
- Mismapping between images with similar regions

## Complete Fix Implemented ✅

### Fix 1: Track Region Type in RT-DETR Guided OCR

**Google** (lines 2098-2111):
```python
all_regions = []
region_types = {}  # NEW: Track which type each region is
idx = 0

if 'text_bubbles' in rtdetr_detections:
    for bbox in rtdetr_detections.get('text_bubbles', []):
        all_regions.append(bbox)
        region_types[idx] = 'text_bubble'  # ✅ Track type
        idx += 1
        
if 'text_free' in rtdetr_detections:
    for bbox in rtdetr_detections.get('text_free', []):
        all_regions.append(bbox)
        region_types[idx] = 'free_text'  # ✅ Track type
        idx += 1
```

**Assign Type to Region** (lines 2189-2190):
```python
# Assign bubble_type from RT-DETR detection
region.bubble_type = region_types.get(region_idx, 'text_bubble')
```

**Enhanced Logging** (line 2192):
```python
self._log(f"✅ Region {i}/{len(all_regions)} ({region.bubble_type}): {region_text[:50]}...")
```

**Same fix applied to Azure** (lines 2434-2447, 2532-2535)

### Fix 2: Include Type in Cache Keys

**Google** (line 3901):
```python
# Include region type in cache key to prevent mismapping
cache_key = ("google", page_hash, x, y, w, h, tuple(lang_hints), detection_mode, roi.get('type', 'unknown'))
```

**Azure** (line 4036):
```python
# Include region type in cache key to prevent mismapping
cache_key = ("azure", page_hash, x, y, w, h, reading_order, roi.get('type', 'unknown'))
```

## Files Modified

- **manga_translator.py**
  - Lines 2096-2111: Google RT-DETR type tracking
  - Line 2123-2124: Updated worker function signature
  - Lines 2188-2192: Assign bubble_type + enhanced logging
  - Line 2210: Updated region_data_list
  - Lines 2432-2447: Azure RT-DETR type tracking
  - Lines 2467-2468: Updated worker function signature
  - Lines 2531-2535: Assign bubble_type + enhanced logging
  - Line 2557: Updated region_data_list
  - Line 3901: Google cache key with type
  - Line 4036: Azure cache key with type

## Impact

### Before ❌
- Free text and bubbles could be mismapped
- RT-DETR type information lost
- Cache collisions causing wrong rendering
- Background opacity applied incorrectly
- Inpainting using wrong dilation settings

### After ✅
- Each region correctly identified as bubble or free text
- Type information preserved throughout pipeline
- Cache properly separates different region types
- Correct background opacity per type
- Correct inpainting dilation per type
- Logs show region types for debugging

## Testing

After this fix, verify:

1. ✅ **RT-DETR Guided OCR** (ROI locality disabled, default):
   - Check logs show region types: `(text_bubble)` or `(free_text)`
   - Verify mask creation shows correct breakdown
   - Test free text gets different background opacity if enabled

2. ✅ **ROI Locality** (if enabled):
   - No cache hits between different types
   - Type persists correctly from cache
   - Multiple images don't contaminate each other

3. ✅ **Rendering**:
   - Free text only BG opacity works correctly
   - Bubbles and free text rendered differently
   - No overlap or misclassification

4. ✅ **Inpainting**:
   - Different dilation for bubbles vs free text
   - Mask logs show correct region type counts
   - Clean inpainting results

## Related Documentation

- `REGION_MAPPING_CACHE_ISSUE.md` - Detailed root cause analysis
- `CACHE_ISOLATION_COMPLETE.md` - General cache isolation fixes
- `PYSIDE6_PERSISTENCE_FIX_COMPLETE.md` - GUI persistence fixes

---

**Date**: 2025-10-04  
**Status**: ✅ **COMPLETE FIX**  
**Affects**: Both RT-DETR guided OCR path AND ROI caching path  
**Result**: Proper region type tracking throughout entire pipeline
