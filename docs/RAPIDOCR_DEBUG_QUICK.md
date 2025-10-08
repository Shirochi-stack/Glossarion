# RapidOCR Debug Quick Guide

## Enable Debug Logging

Go to **Manga Settings → Advanced Tab** and enable **EITHER**:

1. **"Enable debug mode (verbose logging)"** - OR -
2. **"Save intermediate images (preprocessed, detection overlays)"**

Both options will now enable detailed RapidOCR debugging.

**Note**: If you're using parallel processing, you may need to **disable** "Enable parallel processing" temporarily to see the full debug output. Parallel mode shows less verbose logging for performance.

## What You'll See

### RT-DETR Integration Status
```
📝 Using RT-DETR text-masked regions for RapidOCR...
🔍 [RT-DETR] Detection results:
   Text bubbles: 5
   Free text: 2
📊 Processing 7 RT-DETR text-masked regions with RapidOCR
```

### Per-Region Details
```
🔍 [Region 1/7] Type: text_bubble, BBox: 320×180 at (150,200)
   Cropped size: 320×180
🔍 [RapidOCR DEBUG] Input image size: 320x180
🔍 [RapidOCR DEBUG] OCR inference time: 0.234s
🔍 [RapidOCR DEBUG] Raw OCR results count: 3
```

### Text Detection Results
```
🔍 [RapidOCR DEBUG] Region 1:
   Text: 'こんにちは、世界！'
   Confidence: 0.956
   BBox (x,y,w,h): (10, 15, 280, 45)
   Area: 12600 px²
   ✅ Found text: 'こんにちは、世界！' (conf: 0.956)
```

### Summary
```
🔍 [RapidOCR DEBUG] Final results: 3 valid text regions
🔍 [RapidOCR DEBUG] Total text length: 147 chars
```

## Region Mapping Fix

**RapidOCR now correctly uses RT-DETR region boundaries:**

- When RT-DETR provides a text region, RapidOCR recognizes text within it
- All text found by RapidOCR is combined into **one region with RT-DETR boundaries**
- This prevents region mismatch issues where RapidOCR's internal detection conflicts with RT-DETR

You'll see logs like:
```
✅ Found text: 'こんにちは、世界！' (conf: 0.956)
📦 Using RT-DETR region as boundary: 320×180 at (150,200)
🔗 Combined 3 RapidOCR detections into 1 RT-DETR region
```

## Common Issues

**No RT-DETR regions being used:**
```
📊 Processing full image with RapidOCR (NO RT-DETR GUIDANCE)
   ⚠️ WARNING: RT-DETR guidance disabled - OCR quality may be degraded
```
→ Enable "Bubble Detection" in OCR settings

**No text detected:**
```
⚠️ No text detected in region 3
```
→ Check RT-DETR confidence threshold, try lowering it

**Low confidence:**
```
   Confidence: 0.245  [Too low!]
```
→ Check language setting, image quality, or RT-DETR region accuracy

## Disable Debug

Uncheck both "Debug mode" and "Save intermediate" when done debugging.
