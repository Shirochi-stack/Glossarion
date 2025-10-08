# Anime Inpainting Text Region Stability Fix

## Issue
The anime inpainting model (both ONNX and JIT versions) was experiencing partial failures in text regions, showing less stability compared to other inpainting methods like AOT.

## Root Cause
The anime/manga inpainting models were missing a **critical preprocessing step** that was present in AOT models but not in the standard LaMa implementation.

### What Was Missing
**Input masking** - The AOT model masks out the input regions before feeding them to the model:
```python
img = img * (1 - mask)
```

This preprocessing step:
1. **Zeros out the masked regions** in the input image
2. **Forces the model to focus on generation** rather than being influenced by existing content
3. **Improves stability for text-heavy regions** where artifacts can be particularly visible

### Why This Matters for Text Regions
Text regions are particularly challenging for inpainting because:
- Text has high-frequency patterns that can create artifacts
- Partial text characters can confuse the model
- The model might try to "complete" or "fix" visible text fragments instead of fully replacing them
- Text edges are sharp and any bleeding/ghosting is immediately visible

By masking out the input, we ensure the model:
- Doesn't see any text fragments
- Generates clean content from context only
- Produces more stable and consistent results

## The Fix

Added input masking specifically for anime/manga models in three locations:

### 1. ONNX Fixed-Size Model Path (lines ~1992-2020)
```python
elif 'anime' in str(self.current_method).lower():
    # Anime/Manga LaMa normalization: [0, 1] range with optional input masking for stability
    logger.debug("Using Anime/Manga LaMa normalization [0, 1] with input masking")
    img_np = image_resized.astype(np.float32) / 255.0
    mask_np = mask_resized.astype(np.float32) / 255.0
    mask_np = (mask_np > 0.5) * 1.0  # Binary mask
    # CRITICAL: Mask out input regions for better text region stability
    # This helps the model focus on generating content rather than being influenced by text artifacts
    img_np = img_np * (1 - mask_np[:, :, np.newaxis])
```

### 2. ONNX Padded Model Path (lines ~2054-2085)
Same preprocessing applied for variable-size ONNX models with padding.

### 3. JIT Model Path (lines ~2185-2210)
```python
# For anime models: mask out input regions for better text stability
if 'anime' in str(self.current_method).lower():
    logger.debug("Applying input masking for anime model (text region stability)")
    image_norm = image_norm * (1 - mask_binary[:, :, np.newaxis])
```

## Comparison: Before vs After

### Before
- Anime ONNX/JIT models only normalized to [0, 1] range
- Input image with text was sent as-is to the model
- Model could see partial text characters in masked regions
- Text artifacts could bleed through or partially persist

### After  
- Anime models now mask out input regions (like AOT)
- Masked regions are zeroed before inference
- Model only sees clean context around the masked area
- More stable and consistent text region inpainting

## Related Models

This fix applies to:
- **anime_onnx** - Dynamic ONNX model from `ogkalu/lama-manga-onnx-dynamic`
- **anime** (JIT) - JIT model from Sanster's anime-manga-big-lama
- Any method with 'anime' in the name

Standard LaMa models do NOT use input masking as they are trained differently and work well without it.

## RTEDR Model
If RTEDR is based on similar architecture to anime/manga models, it may benefit from the same preprocessing. The fix uses a simple check:
```python
if 'anime' in str(self.current_method).lower():
```

To apply this to RTEDR, you could either:
1. Rename the method to include 'anime' in the string
2. Add RTEDR to the condition: `if 'anime' in str(...) or 'rtedr' in str(...)`

## Testing Recommendations

Test with:
1. Dense text regions (manga panels with lots of dialogue)
2. Small text (furigana, sound effects)
3. Text with complex backgrounds
4. Overlapping text bubbles
5. Text at various angles

Compare stability and quality between:
- Anime ONNX vs Anime JIT (should now be consistent)
- Anime vs Standard LaMa (anime should be better for text-heavy content)
- Anime vs AOT (should have similar stability characteristics)

## Performance Impact

**Minimal** - The masking operation is a simple element-wise multiplication that happens once during preprocessing:
```python
img = img * (1 - mask)  # O(width × height × channels)
```

This is negligible compared to the model inference time.

## Credits

This fix aligns the anime inpainting preprocessing with the AOT model's approach, which has proven effective for text region stability. The insight comes from analyzing the differences between model preprocessing pipelines in `local_inpainter.py`.
