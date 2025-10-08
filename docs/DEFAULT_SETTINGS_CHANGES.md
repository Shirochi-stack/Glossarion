# Default Settings Changes

## Summary
Updated default settings in the manga translation pipeline to enable better OCR accuracy out of the box.

## Changes Made

### 1. ✅ Free Text Only Background Opacity
**Status**: Already enabled by default ✓

**Location**: `manga_integration.py` line 4274

**Current Default**: `True`

**What it does**: 
- When enabled, background opacity is applied **only to free text regions** (text outside bubbles)
- Text inside speech bubbles remains with minimal/no background
- Provides cleaner look for bubble text while ensuring free text readability

**Impact**: 
- Better visual quality for translated manga
- Free text gets proper background for readability
- Bubble text remains clean without unnecessary backgrounds

---

### 2. ✅ RT-DETR to Guide OCR
**Status**: **NOW ENABLED** by default ✓

**Location**: `manga_settings_dialog.py` lines 93, 2735

**Previous Default**: `False`  
**New Default**: `True`

**What it does**:
- Uses RT-DETR (Real-Time Detection Transformer) to detect all text regions first
- Then guides Google Cloud Vision or Azure Computer Vision to OCR each detected region
- Other OCR providers (Qwen2-VL, EasyOCR, PaddleOCR, etc.) already use this approach natively

**Benefits**:
- 🎯 **More accurate text detection** - RT-DETR is trained specifically for manga/comics
- 📍 **Better region isolation** - Separates overlapping text and different text types
- 🔍 **Focused OCR** - Each text region is OCR'd individually, improving accuracy
- 🎨 **Type-aware** - Distinguishes between speech bubbles, thought bubbles, and free text
- ⚡ **Faster processing** - OCR only runs on actual text regions, not entire image

**Applies to**: 
- ✅ Google Cloud Vision
- ✅ Azure Computer Vision

**Already enabled by default in**:
- Qwen2-VL (built-in)
- Custom API endpoints (built-in)
- EasyOCR (built-in)
- PaddleOCR (built-in)
- DocTR (built-in)
- manga-ocr (built-in)

---

## Files Modified

1. **manga_settings_dialog.py**
   - Line 93: Changed default dict value from `False` to `True`
   - Line 2735: Changed checkbox default fallback from `False` to `True`

2. **manga_integration.py**
   - Line 4274: Confirmed already defaults to `True` (no change needed)

---

## User Impact

### For New Users
- Better OCR accuracy immediately without needing to configure settings
- RT-DETR guidance provides manga-optimized text detection for Google/Azure
- Free text backgrounds ensure readability while keeping bubbles clean

### For Existing Users
- If you previously disabled RT-DETR guidance, your setting is preserved
- New installations or reset configs will get the improved defaults
- You can still toggle it off in Advanced Settings > OCR > AI Bubble Detection

---

## Configuration Location

These settings are stored in the config file under:

```json
{
  "manga_free_text_only_bg_opacity": true,
  "manga_settings": {
    "ocr": {
      "use_rtdetr_for_ocr_regions": true
    }
  }
}
```

---

## Related Documentation

- `AZURE_RATE_LIMITING_FIX.md` - OCR rate limiting protection
- `CACHE_ISOLATION_COMPLETE.md` - Cache isolation fixes
- `PARALLEL_PANEL_SAFETY.md` - Thread safety for parallel processing
- RT-DETR bubble detection settings in Advanced Settings dialog

---

## Testing Recommendations

After these changes, test with:
1. ✅ Manga pages with both bubble text and free text
2. ✅ Google Cloud Vision OCR
3. ✅ Azure Computer Vision OCR
4. ✅ Complex layouts with overlapping text
5. ✅ Pages with mixed text types (speech bubbles, thought bubbles, narration)

Expected improvements:
- More accurate text region detection
- Better separation of overlapping text
- Improved handling of free text vs. bubble text
- Cleaner visual output with appropriate backgrounds

---

**Date**: 2025-10-04  
**Version**: Latest
