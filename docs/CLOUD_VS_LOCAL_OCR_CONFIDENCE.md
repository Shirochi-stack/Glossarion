# Cloud vs Local OCR Confidence Settings

## Overview

Implemented separation between **cloud OCR confidence** (Google/Azure) and **local OCR** (RapidOCR, PaddleOCR, etc.) based on comic-translate's approach.

## The Change

### Before
```
Confidence Threshold: [slider] 0.00
  ↓
Applied to ALL OCR providers equally
```

### After (Option B)
```
☁️ Cloud OCR Confidence: [slider] 0.00
  ↓
• Google Cloud Vision: Uses slider value
• Azure Computer Vision: Uses slider value
• RapidOCR: Uses 0.0 (no filtering)
• PaddleOCR: Uses 0.0 (no filtering)
• EasyOCR: Uses 0.0 (no filtering)
• DocTR: Uses 0.0 (no filtering)
```

## Rationale

### Cloud Providers Need Filtering
**Google and Azure** return confidence scores that are useful for filtering:
```python
# Google Cloud Vision result
line.confidence = 0.95  # Very confident
line.confidence = 0.32  # Not confident - probably wrong
```

**Benefit**: Can filter out low-quality results to improve accuracy

### Local Providers Don't Need Filtering
**RapidOCR, PaddleOCR, EasyOCR, DocTR** either:
- Don't return meaningful confidence scores
- Return scores that aren't calibrated
- Always return high confidence even when wrong

**Solution (Comic-Translate Approach)**: 
- RT-DETR already filtered regions at 30% confidence
- Trust those regions and accept all text found by local OCR
- No additional confidence filtering needed

## Implementation Details

### 1. Settings Storage

```python
# New setting (preferred)
settings['ocr']['cloud_ocr_confidence'] = 0.0  # Applies to Google/Azure only

# Old setting (kept for backward compatibility)
settings['ocr']['confidence_threshold'] = 0.0  # Mirrors cloud_ocr_confidence
```

### 2. UI Changes (manga_settings_dialog.py)

**Label changed:**
```python
"Confidence Threshold:" 
  ↓
"☁️ Cloud OCR Confidence:"
```

**Tooltip added:**
```
"Applies to Google Cloud Vision and Azure OCR only.
Local OCR (RapidOCR, PaddleOCR, etc.) uses RT-DETR confidence only (comic-translate approach)."
```

**Info label added below slider:**
```
ℹ️ Local OCR providers (RapidOCR, PaddleOCR, EasyOCR, DocTR) don't use this
   - they rely on RT-DETR confidence only
```

### 3. Runtime Logic (manga_translator.py)

```python
# Confidence threshold: Cloud providers vs Local OCR
if self.ocr_provider in ['google', 'azure']:
    # Cloud providers: use configurable threshold
    confidence_threshold = ocr_settings.get('cloud_ocr_confidence', 0.0)
else:
    # Local OCR: no filtering (trust RT-DETR regions)
    confidence_threshold = 0.0
```

## User Experience

### What Users See

**In Settings Dialog:**
```
OCR Parameters
├── ☁️ Cloud OCR Confidence: [========] 0.00
│   ℹ️ Local OCR providers don't use this - they rely on RT-DETR confidence only
└── Detection Mode: [document ▼]
```

**Tooltip on hover:**
```
Applies to Google Cloud Vision and Azure OCR only.
Local OCR (RapidOCR, PaddleOCR, etc.) uses RT-DETR confidence only
(comic-translate approach).

0 = accept all (recommended, like comic-translate)
Higher values filter low-confidence cloud OCR results
```

### What Happens at Runtime

**Google/Azure:**
```python
# User sets cloud_ocr_confidence = 0.5 (50%)
confidence_threshold = 0.5

for line in google_results:
    if line.confidence >= 0.5:  # Apply filtering
        keep_text()
    else:
        skip_text()
```

**RapidOCR/PaddleOCR/etc:**
```python
# Automatically uses 0.0 regardless of slider
confidence_threshold = 0.0

for line in rapidocr_results:
    if line.confidence >= 0.0:  # Always passes
        keep_text()
```

## Backward Compatibility

- ✅ Existing configs still work
- ✅ `confidence_threshold` migrates to `cloud_ocr_confidence`
- ✅ Old setting kept as mirror for compatibility
- ✅ No breaking changes

Migration on first load:
```python
# In UI initialization
cloud_conf = settings['ocr'].get('cloud_ocr_confidence', 
                                   settings['ocr'].get('confidence_threshold', 0.0))
```

## Comic-Translate Alignment

This matches comic-translate's approach:

### Comic-Translate
```python
# RT-DETR detects regions with 0.3 confidence
text_blocks = rtdetr.detect(image, confidence=0.3)

# OCR processes regions without confidence filtering
for block in text_blocks:
    ocr_lines = paddle_ocr.ocr(image)  # No confidence filter!
    block.text = match_and_join(ocr_lines, block)
```

### Our Implementation
```python
# RT-DETR detects regions with 0.3 confidence (configurable via RT-DETR slider)
rt_detr_regions = bubble_detector.detect(confidence=0.3)

# Local OCR processes without confidence filtering
if provider in ['rapidocr', 'paddleocr', 'easyocr', 'doctr']:
    confidence_threshold = 0.0  # No filtering, trust RT-DETR
    
# Cloud OCR can optionally filter
elif provider in ['google', 'azure']:
    confidence_threshold = cloud_ocr_confidence  # User configurable
```

## Benefits

✅ **Clearer UI**: Users understand cloud vs local distinction  
✅ **Better defaults**: Local OCR uses comic-translate approach (no filtering)  
✅ **More control**: Cloud users can still adjust if needed  
✅ **Simpler logic**: One slider, clear behavior per provider  
✅ **Proven approach**: Based on comic-translate's successful implementation

## Settings Summary

### RT-DETR Confidence (separate slider)
- **Location**: Bubble Detection section
- **Default**: 0.3 (30%)
- **Purpose**: Filter which regions RT-DETR detects
- **Applies to**: All providers (when bubble detection enabled)

### Cloud OCR Confidence (this slider)
- **Location**: OCR Parameters section
- **Default**: 0.0 (0% - accept all)
- **Purpose**: Filter low-confidence cloud OCR results
- **Applies to**: Google Cloud Vision, Azure Computer Vision only

### Local OCR
- **No slider**: Uses 0.0 internally (no filtering)
- **Purpose**: Trust RT-DETR regions completely
- **Applies to**: RapidOCR, PaddleOCR, EasyOCR, DocTR

## Recommended Settings

### For Most Users
```
RT-DETR Confidence: 0.30 (30%)
Cloud OCR Confidence: 0.00 (0% - accept all)
```

This matches comic-translate's approach and provides best results.

### For Strict Quality
```
RT-DETR Confidence: 0.40 (40%) - fewer but more accurate regions
Cloud OCR Confidence: 0.50 (50%) - filter uncertain cloud results
```

Use if you're getting too much noise/false positives.

### For Maximum Recall
```
RT-DETR Confidence: 0.20 (20%) - catch more regions
Cloud OCR Confidence: 0.00 (0%) - accept all text
```

Use if you're missing text and want to catch everything.

## Summary

✅ **Cloud providers (Google/Azure)**: Use `cloud_ocr_confidence` slider  
✅ **Local providers (RapidOCR/PaddleOCR/etc.)**: Use `0.0` (no filtering)  
✅ **RT-DETR**: Has its own confidence slider (already exists)  
✅ **Comic-translate aligned**: Local OCR trusts RT-DETR regions only  
✅ **User-friendly**: Clear labels and tooltips explain behavior

🎉 **Option B successfully implemented!**
