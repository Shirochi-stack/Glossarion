# Reverted Fix #4: Removed Result Sorting

## What Was Changed
**Removed** the sorting of OCR results by position that was added in Fix #4.

## Why It Was Removed
**The sorting was causing region mapping problems!**

### The Problem
1. Fix #4 sorted results by position: `results.sort(key=lambda r: (r.bounding_box[1], r.bounding_box[0]))`
2. This caused regions to have **different order** between detection runs
3. The API returns translations as a dictionary with the **original text as keys**
4. But when the region order changes, the **key-based matching fails** because:
   - Azure OCR text: `'大丈夫か !? だいじょう ぶ 顔がマッチョに! かお 顔が! かお'`
   - API returns key: `"顔が！顔がマッチョに！大丈夫か！？"` (cleaned/reordered)
   - Keys don't match → wrong translation goes to wrong bubble!

### Example of the Issue
**Run 1:**
- Regions detected in order: `['必死なんだぞ…', 'わがらないく', ...]`
- After sorting: `['おもろ…', '必死なんだぞ…', ...]` (different order!)

**Run 2:**
- Regions detected in order: `['わがらないく', '必死なんだぞ…', ...]`
- After sorting: `['おもろ…', '必死なんだぞ…', ...]` (same result)

But the translations from the API have keys that don't perfectly match the OCR text (due to furigana, cleaning, etc.), so positional changes break the mapping.

## What Was Removed

### Location 1: Google OCR (Line 4324-4329)
```python
# REMOVED:
# FIX #4: Sort results by position after concurrent processing
# Batch processing may return results out of order
# Sort top-to-bottom, then left-to-right for consistent ordering
if len(results) > 1:
    results.sort(key=lambda r: (r.bounding_box[1], r.bounding_box[0]))
    self._log(f"✅ Sorted {len(results)} Google results by position (top-to-bottom, left-to-right)", "debug")
```

### Location 2: Azure OCR (Line 4507-4512)
```python
# REMOVED:
# FIX #4: Sort results by position after concurrent processing
# as_completed() returns in completion order, not spatial order
# Sort top-to-bottom, then left-to-right for consistent ordering
if len(results) > 1:
    results.sort(key=lambda r: (r.bounding_box[1], r.bounding_box[0]))
    self._log(f"✅ Sorted {len(results)} Azure results by position (top-to-bottom, left-to-right)", "debug")
```

## Current Behavior (After Removal)
- Results are returned in **detection order** (no sorting)
- Order may vary between runs due to concurrent processing
- **But** the key-based matching in `translate_full_page_context` should handle this
- Region order is **more stable** within a single image processing

## The Real Solution
The real fix should be in the **translation mapping logic**, not in sorting:
- Use **fuzzy/similarity matching** between OCR text and API response keys
- Or ensure the API receives and returns text in exactly the same format
- Or use a more robust matching algorithm that handles:
  - Furigana differences (`だいじょう ぶ` vs without)
  - Punctuation differences (`!?` vs `！？`)
  - Whitespace differences
  - Character order differences

## Remaining Fixes Still Active
- ✅ Fix #1: Page hash never None
- ✅ Fix #2: Clear cache before AND after image processing
- ✅ Fix #3: Include image path + region type in cache keys
- ❌ Fix #4: **REMOVED** (was causing problems)
- ✅ Fix #5: Cache hit logging

## Testing Recommendations
1. Translate the same image multiple times
2. Verify text appears in **correct bubbles** consistently
3. Check that order variation doesn't break mapping
4. Monitor for any new cross-contamination issues

## Future Improvements Needed
Consider implementing:
1. **Fuzzy text matching** for API response keys
2. **Stable region ordering** at detection time (not post-processing)
3. **Position-aware matching** when key matching fails
4. **Confidence scores** for mapping quality
