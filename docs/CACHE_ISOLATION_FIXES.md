# Cache Isolation Fixes for Manga Translation Pipeline

## Problem
Text data was potentially leaking between images due to shared caches in the manga translation pipeline.

## Changes Made

### 1. manga_translator.py

#### A. OCR ROI Cache (Lines 233-236)
**BEFORE:**
```python
# OCR ROI cache (image_hash + bbox + provider + lang + mode -> text)
self.ocr_roi_cache = {}
```

**AFTER:**
```python
# OCR ROI cache - PER IMAGE ONLY (cleared aggressively to prevent text leakage)
# CRITICAL: This cache MUST be cleared before every new image to prevent text contamination
self.ocr_roi_cache = {}
self._current_image_hash = None  # Track current image to force cache invalidation
```

#### B. Generic Cache Documentation (Lines 261-263)
Added warning documentation that `self.cache = {}` should NOT be used for text data.

#### C. detect_text_regions() - Aggressive Cache Clearing (Lines 1989-2024)
**BEFORE:** Conditional cache clearing based on batch_mode (DANGEROUS!)

**AFTER:** UNCONDITIONAL clearing of ALL text-related caches:
- OCR ROI cache
- OCR manager caches (last_results + generic cache)
- Provider-level caches (for EVERY provider)
- Bubble detector cache

**KEY CHANGE:** Removed batch_mode condition - ALL caches cleared for EVERY image.

#### D. Image Hash Tracking (Lines 2056-2066)
Added automatic cache invalidation when image hash changes:
```python
# CRITICAL: If image hash changed, force clear ROI cache
if hasattr(self, '_current_image_hash') and self._current_image_hash != page_hash:
    if hasattr(self, 'ocr_roi_cache'):
        self.ocr_roi_cache.clear()
self._current_image_hash = page_hash
```

#### E. reset_for_new_image() - Comprehensive Clearing (Lines 8617-8661)
Enhanced to clear:
- OCR ROI cache
- Image hash tracker
- ALL OCR manager caches
- ALL provider-level caches
- Bubble detector cache

#### F. clear_internal_state() - Added ROI Cache Clearing (Lines 9307-9313)
Added explicit clearing of ocr_roi_cache and image hash tracker.

## Cache Clearing Hierarchy

### Before Each Image (detect_text_regions):
1. **OCR ROI cache** → `.clear()`
2. **OCR manager**:
   - `last_results` → `None`
   - `cache` → `.clear()`
   - For each provider:
     - `last_results` → `None`
     - `cache` → `.clear()`
3. **Bubble detector**:
   - `last_detections` → `None`
   - `cache` → `.clear()`

### On Image Hash Change:
1. **OCR ROI cache** → `.clear()` (automatic)

### On Manual Reset (reset_for_new_image):
1. All of the above
2. **Image hash tracker** → `None`

### On Cleanup (clear_internal_state):
1. All of the above

## Key Principles

1. **NO BATCH MODE EXCEPTIONS**: All caches cleared regardless of batch processing mode
2. **EXPLICIT CLEARING**: Use `.clear()` instead of reassignment for better debugging
3. **PROVIDER-LEVEL ISOLATION**: Clear caches at provider level, not just manager level
4. **IMAGE HASH TRACKING**: Automatic invalidation on image change
5. **MULTIPLE SAFEGUARDS**: Clear at multiple points (detect, reset, cleanup) for redundancy

## Impact on ROI-Based OCR

Even though ROI-based OCR is disabled in settings, the cache isolation is still in place to:
- Prevent future bugs if ROI OCR is re-enabled
- Ensure complete isolation of ALL text-related caches
- Provide defense-in-depth against text contamination

## Testing Checklist

- [ ] Process multiple images in sequence
- [ ] Verify no text from Image A appears in Image B
- [ ] Check debug logs for cache clearing messages
- [ ] Test with batch mode enabled
- [ ] Test with batch mode disabled
- [ ] Verify memory usage remains stable across images

## Files Modified
- `manga_translator.py` (7 locations)
- Created: `CACHE_ISOLATION_FIXES.md` (this file)
