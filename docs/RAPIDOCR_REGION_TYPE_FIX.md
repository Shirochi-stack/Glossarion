# RapidOCR Region Type Preservation Fix

## Problem

Looking at your logs:
```
📊 Parallel OCR complete: 10/11 regions extracted
🎯 Skipping bubble detection merge (regions already aligned with RT-DETR)
✅ Detected 10 text regions after merging
```

Then later:
```
📊 Mask breakdown: 10 text bubbles, 0 empty bubbles, 0 free text regions, 0 skipped
```

**Issue**: All 10 regions classified as `text_bubbles`, even though RT-DETR detected:
- 8 text bubbles
- 3 free text
- 8 empty bubbles

## Why This Matters

Region type affects:
1. **Mask dilation** - Different iterations for different types
2. **Rendering** - Different strategies for text vs free text
3. **Classification** - Proper understanding of what was detected

Without proper types:
- All regions get same dilation (text bubble settings)
- Can cause over-dilation or under-dilation
- Loses RT-DETR's classification intelligence

## Root Cause

When RapidOCR processes RT-DETR regions:

1. **RT-DETR** detects and classifies regions:
   ```python
   region_types = {
       0: 'text_bubble',
       1: 'text_bubble', 
       2: 'free_text',
       ...
   }
   ```

2. **Sequential path** used `region_types` but didn't attach to result ❌
3. **Parallel path** didn't even receive `region_types` ❌❌

4. **Result**: All OCRResults created without `bubble_type` attribute
5. **Masking code** defaults to `'text_bubble'` when attribute missing

## The Fix

### 1. Pass Region Types to Parallel Function

**Before**:
```python
ocr_results = self._parallel_ocr_regions(
    image, all_regions, 'rapidocr', confidence_threshold
)
```

**After**:
```python
ocr_results = self._parallel_ocr_regions(
    image, all_regions, 'rapidocr', confidence_threshold,
    region_types=region_types  # Pass the types!
)
```

### 2. Update Parallel Function Signature

```python
def _parallel_ocr_regions(self, image, regions, provider, 
                          confidence_threshold, region_types=None):
    """
    Args:
        region_types: Dict mapping region index to type 
                      ('text_bubble', 'free_text', 'empty_bubble')
    """
```

### 3. Attach Region Type in Parallel Path

```python
def process_single_region(index: int, bbox: Tuple):
    region_type = region_types.get(index, 'text_bubble') if region_types else 'text_bubble'
    
    # ... OCR processing ...
    
    combined_result.bubble_bounds = (x, y, w, h)
    combined_result.bubble_type = region_type  # Attach it!
    return (index, combined_result)
```

### 4. Attach Region Type in Sequential Path

```python
rtype = region_types.get(i, 'text_bubble')
combined_result.bubble_type = rtype
ocr_results.append(combined_result)
```

## Expected Results

After fix, you should see proper classification:
```
📊 Mask breakdown: 8 text bubbles, 0 empty bubbles, 2 free text regions, 0 skipped
```

This matches RT-DETR's detection:
- 8 text bubbles → text bubble dilation rules
- 2 free text (from 3 detected, maybe 1 had no text) → free text dilation rules
- Empty bubbles without text are skipped from OCR results

## Mask Dilation Impact

With proper classification, mask generation uses correct settings:

### Auto Iterations (B&W images)
```python
text_bubble_iterations = 2    # For speech bubbles
empty_bubble_iterations = 2   # For empty bubbles  
free_text_iterations = 0      # For sound effects (no dilation)
```

### Manual Settings
From Manga Settings → Mask Dilation, each type can have different values.

**Before fix**: All regions got `text_bubble_iterations` (2)
**After fix**: Each region gets its proper iteration count

## Testing

Run the same image again and check logs:
```
📊 Mask breakdown: X text bubbles, Y empty bubbles, Z free text regions
```

Should match RT-DETR's classification, not show "0 empty, 0 free".

## Files Modified

- `manga_translator.py` (lines ~3964, ~4012-4014, ~4182-4197, ~4285-4286, ~4293-4294)

## Related

This fix complements:
- `RAPIDOCR_PARALLEL_FIX.md` - Text combination fix
- `RAPIDOCR_RTDETR_FIX.md` - Boundary preservation fix

Together, these ensure RapidOCR with RT-DETR:
✅ Uses RT-DETR boundaries
✅ Combines all text correctly  
✅ Preserves RT-DETR classification
✅ Applies proper mask dilation per type
