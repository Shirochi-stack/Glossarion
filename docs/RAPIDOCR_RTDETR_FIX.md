# RapidOCR + RT-DETR Region Mapping Fix

## Problem

RapidOCR was messing up region mapping when used with RT-DETR because:

1. **RT-DETR** detects text regions (bubbles, free text) and provides bounding boxes
2. **RapidOCR** was then doing its OWN internal text detection within those regions
3. **Conflict**: RapidOCR's internal detection boundaries didn't match RT-DETR's boundaries
4. **Result**: Wrong regions were being inpainted, text was placed incorrectly

## Solution

**RapidOCR now relies on RT-DETR for region mapping:**

### Before Fix ❌
```
RT-DETR finds:  [Speech Bubble: 320×180 at (150,200)]
                 ↓ Crop and send to RapidOCR
RapidOCR finds: [Line 1: 200×30 at local (10,15)]
                [Line 2: 180×25 at local (15,55)]
                [Line 3: 190×28 at local (12,95)]
                 ↓ Convert to global coordinates
Result:         3 separate regions with RapidOCR boundaries
                ❌ RT-DETR region LOST!
```

### After Fix ✅
```
RT-DETR finds:  [Speech Bubble: 320×180 at (150,200)]
                 ↓ Crop and send to RapidOCR
RapidOCR finds: "こんにちは、世界！これはテスト"
                 (combines all detected text)
                 ↓ Use RT-DETR boundary
Result:         1 region with RT-DETR boundaries
                ✅ Region mapping preserved!
```

## Key Changes

### 1. Text Combination
When RapidOCR detects multiple text lines within an RT-DETR region:
- Combine all text with spaces: `' '.join(texts)`
- Calculate average confidence across all detections
- Use RT-DETR's region boundaries, NOT RapidOCR's internal detections

### 2. Region Preservation
```python
# OLD (wrong):
result[0].bbox = result[0].bbox  # RapidOCR's detection
result[0].vertices = result[0].vertices  # Wrong boundaries

# NEW (correct):
combined_result.bbox = (x, y, w, h)  # RT-DETR region
combined_result.vertices = [(x,y), (x+w,y), (x+w,y+h), (x,y+h)]
combined_result.bubble_bounds = (x, y, w, h)  # For rendering
```

### 3. Debug Logging
When debug mode is enabled, you'll see:
```
✅ Found text: 'こんにちは、世界！これはテスト' (conf: 0.956)
📦 Using RT-DETR region as boundary: 320×180 at (150,200)
🔗 Combined 3 RapidOCR detections into 1 RT-DETR region
```

This confirms:
- Text was successfully recognized
- RT-DETR boundaries are being used
- Multiple RapidOCR detections were combined (if applicable)

## Why This Matters

### Correct Inpainting
- Inpainting mask matches RT-DETR detected bubble exactly
- No partial inpainting or over-inpainting
- Clean bubble removal

### Correct Text Rendering
- Text is rendered within the actual bubble boundaries
- Font sizing uses correct region dimensions
- No text overflow or misplacement

### Better OCR Quality
- RT-DETR is specialized for manga text detection
- RapidOCR is used only for text recognition (its strength)
- Best of both worlds: RT-DETR for "where", RapidOCR for "what"

## Comparison with Other Providers

| Provider | Region Detection | Text Recognition | Region Mapping |
|----------|------------------|------------------|----------------|
| **RapidOCR (Fixed)** | RT-DETR | RapidOCR | RT-DETR boundaries ✅ |
| manga-ocr | RT-DETR | manga-ocr | RT-DETR boundaries ✅ |
| Azure/Google | Full page | Azure/Google | Merged to RT-DETR bubbles |
| PaddleOCR | RT-DETR | PaddleOCR | RT-DETR boundaries ✅ |
| DocTR | RT-DETR | DocTR | RT-DETR boundaries ✅ |

## Testing

To verify the fix is working:

1. **Enable debug mode** (Manga Settings → Advanced → Debug mode)
2. **Enable RT-DETR** (Manga Settings → OCR → Bubble Detection)
3. **Use RapidOCR** as your OCR provider
4. **Translate an image**

Look for these logs:
- `📝 Using RT-DETR text-masked regions for RapidOCR`
- `📦 Using RT-DETR region as boundary: [dimensions]`
- `🔗 Combined X RapidOCR detections into 1 RT-DETR region` (if multiple)

If you see these, the fix is working correctly!

## Edge Cases Handled

1. **Multiple text lines in one bubble**: Combined into single region
2. **Empty RapidOCR results**: Logged as warning, region skipped
3. **Low confidence detections**: Still use RT-DETR boundaries, average confidence calculated
4. **Parallel processing**: Each worker uses RT-DETR regions independently

## Benefits

✅ **Accurate region mapping** - RT-DETR boundaries always used  
✅ **Better text recognition** - RapidOCR does what it's good at  
✅ **Cleaner inpainting** - Masks match actual bubble boundaries  
✅ **Proper rendering** - Text fits within correct regions  
✅ **Debug visibility** - Clear logging shows what's happening  
✅ **Consistent behavior** - Matches other providers (manga-ocr, etc.)  

## Migration Note

If you were using RapidOCR without RT-DETR before:
- **No changes** - Full image mode still works as before
- With RT-DETR enabled, you'll now get **better** region mapping
- Old behavior was buggy, new behavior is correct

## Related Files

- `manga_translator.py`: RapidOCR RT-DETR integration (lines ~3901-4026)
- `ocr_manager.py`: RapidOCR provider with debug logging (lines ~1813-1898)
- `RAPIDOCR_DEBUG_QUICK.md`: Debug mode usage guide
