# Conditional Merge Logic for OCR Providers

## Overview

Different OCR providers need different merging strategies depending on:
1. Which provider is used
2. Whether bubble detection is enabled
3. Whether RT-DETR guidance is enabled (for cloud providers)

## The Merging Decision Tree

```
Is bubble_detection_enabled?
├─ NO → Apply proximity-based merging
│         (merge nearby regions within threshold)
│
└─ YES → Check provider type:
          │
          ├─ Local OCR (RapidOCR, PaddleOCR, etc.)
          │  → SKIP merge (regions already aligned)
          │
          └─ Cloud OCR (Google, Azure)
             │
             ├─ Is use_rtdetr_for_ocr_regions enabled?
             │  │
             │  ├─ YES → SKIP merge
             │  │         (processed regions individually)
             │  │
             │  └─ NO → APPLY bubble merge
             │           (full-image lines need grouping)
```

## Provider Behavior Details

### Local OCR Providers (Always Skip Merge When Bubble Detection Enabled)

**Providers:** RapidOCR, PaddleOCR, EasyOCR, DocTR, manga-ocr, Qwen2-VL, custom-api

**Why skip merge:**
- **RapidOCR (with comic-translate approach)**: 
  - Runs full-image OCR
  - Matches lines to RT-DETR blocks
  - Each result already has correct `bubble_bounds`
  
- **Others**:
  - Process RT-DETR regions individually
  - Each result already has `bubble_bounds` from RT-DETR

**Code:**
```python
skip_merge_providers = ['rapidocr', 'manga-ocr', 'Qwen2-VL', 
                        'custom-api', 'easyocr', 'paddleocr', 'doctr']
```

### Cloud OCR Providers (Conditional Merge)

**Providers:** Google Cloud Vision, Azure Computer Vision

#### Scenario A: RT-DETR Guidance Enabled (Default)

**Setting:** `use_rtdetr_for_ocr_regions = True`

**Behavior:**
1. RT-DETR detects text regions
2. Cloud OCR processes each region individually
3. Each result already aligned to RT-DETR region
4. **Skip merge** (already aligned)

**Code:**
```python
use_rtdetr_guidance = ocr_settings.get('use_rtdetr_for_ocr_regions', True)
if use_rtdetr_guidance:
    if self.ocr_provider in ['google', 'azure']:
        skip_merge_providers.extend(['google', 'azure'])
```

#### Scenario B: RT-DETR Guidance Disabled

**Setting:** `use_rtdetr_for_ocr_regions = False`

**Behavior:**
1. Cloud OCR processes full image
2. Returns many line-level results scattered across page
3. Lines need to be grouped into speech bubbles
4. **Apply merge** (group lines into bubbles)

**Code:**
```python
# Cloud providers not in skip list
# → Falls through to merge logic
regions = self._merge_with_bubble_detection(regions, image_path)
```

## Implementation

### Code Location
`manga_translator.py`, lines 4316-4338

### Full Logic
```python
# MERGING SECTION (applies to all providers)
if ocr_settings.get('bubble_detection_enabled', False):
    # Build list of providers that should skip merging
    skip_merge_providers = ['rapidocr', 'manga-ocr', 'Qwen2-VL', 
                           'custom-api', 'easyocr', 'paddleocr', 'doctr']
    
    # If RT-DETR guidance is enabled for cloud providers, they also skip merging
    use_rtdetr_guidance = ocr_settings.get('use_rtdetr_for_ocr_regions', True)
    if use_rtdetr_guidance:
        if self.ocr_provider in ['google', 'azure']:
            skip_merge_providers.extend(['google', 'azure'])
    
    if self.ocr_provider in skip_merge_providers:
        # Skip merge - regions already aligned
        self._log("🎯 Skipping bubble detection merge (regions already aligned with RT-DETR)")
    else:
        # Apply bubble merge
        self._log("🤖 Using AI bubble detection for merging")
        regions = self._merge_with_bubble_detection(regions, image_path)
else:
    # Bubble detection disabled - use proximity merging
    merge_threshold = ocr_settings.get('merge_nearby_threshold', 20)
    regions = self._merge_nearby_regions(regions, threshold=merge_threshold)
```

## Examples

### Example 1: RapidOCR (Comic-Translate Approach)

**Settings:**
- Provider: RapidOCR
- bubble_detection_enabled: True

**Flow:**
```
1. RT-DETR detects 10 text regions
2. RapidOCR processes full image → 45 text lines
3. Match 45 lines to 10 RT-DETR blocks (comic-translate)
4. Each of 10 results has bubble_bounds from matching
5. Skip merge ✓ (already aligned)
```

**Result:** 10 properly aligned text regions

### Example 2: Google with RT-DETR Guidance

**Settings:**
- Provider: Google Cloud Vision
- bubble_detection_enabled: True
- use_rtdetr_for_ocr_regions: True

**Flow:**
```
1. RT-DETR detects 10 text regions
2. Google OCR processes each of 10 regions individually
3. Each region returns text for that bubble
4. Each result has bubble bounds from RT-DETR
5. Skip merge ✓ (already aligned)
```

**Result:** 10 properly aligned text regions

### Example 3: Google without RT-DETR Guidance

**Settings:**
- Provider: Google Cloud Vision
- bubble_detection_enabled: True
- use_rtdetr_for_ocr_regions: False

**Flow:**
```
1. Google OCR processes full image → 45 text lines
2. Lines scattered across entire page
3. Need to group lines into speech bubbles
4. Apply bubble merge (group nearby lines)
```

**Result:** Lines grouped into bubbles based on proximity

### Example 4: Azure without Bubble Detection

**Settings:**
- Provider: Azure Computer Vision
- bubble_detection_enabled: False

**Flow:**
```
1. Azure OCR processes full image → 45 text lines
2. No bubble detection to guide grouping
3. Apply proximity-based merging (20px threshold)
4. Lines within 20px merged together
```

**Result:** Lines grouped by proximity

## Benefits of Conditional Logic

✅ **Prevents double-merging**: RapidOCR's comic-translate matching isn't ruined by re-merging  
✅ **Efficient for cloud**: When RT-DETR guides cloud OCR, skip unnecessary merge  
✅ **Flexible**: When RT-DETR guidance disabled, still group lines properly  
✅ **Consistent**: All providers that process regions individually skip merge  

## Settings Summary

### Bubble Detection Toggle
- **Location**: Manga Settings → OCR → Bubble Detection
- **Default**: True (enabled)
- **Effect**: Enables RT-DETR text region detection

### RT-DETR Guidance Toggle (for Cloud Providers)
- **Location**: Manga Settings → OCR → "Use RT-DETR to guide Google/Azure OCR"
- **Setting name**: `use_rtdetr_for_ocr_regions`
- **Default**: True (enabled)
- **Effect**: 
  - When True: Cloud OCR processes RT-DETR regions individually (skip merge)
  - When False: Cloud OCR processes full image (apply merge)

## Testing Scenarios

### Verify Skip Merge (RapidOCR)
```
Settings:
- Provider: RapidOCR
- bubble_detection_enabled: True

Expected log:
🎯 Skipping bubble detection merge (regions already aligned with RT-DETR)
```

### Verify Skip Merge (Google with RT-DETR)
```
Settings:
- Provider: Google
- bubble_detection_enabled: True
- use_rtdetr_for_ocr_regions: True

Expected log:
🎯 Skipping bubble detection merge (regions already aligned with RT-DETR)
```

### Verify Apply Merge (Google without RT-DETR)
```
Settings:
- Provider: Google
- bubble_detection_enabled: True
- use_rtdetr_for_ocr_regions: False

Expected log:
🤖 Using AI bubble detection for merging
```

## Summary

The merging logic now correctly handles:

1. ✅ **Local OCR** → Always skip merge (regions aligned to RT-DETR)
2. ✅ **Cloud OCR + RT-DETR guidance** → Skip merge (regions processed individually)
3. ✅ **Cloud OCR without RT-DETR guidance** → Apply merge (group full-image lines)
4. ✅ **No bubble detection** → Proximity merge for all providers

This ensures:
- RapidOCR's comic-translate matching isn't disrupted
- Cloud providers with RT-DETR guidance don't double-merge
- Cloud providers without RT-DETR guidance still group lines properly
- All providers work optimally based on their processing approach

🎉 **Conditional merge logic successfully implemented!**
