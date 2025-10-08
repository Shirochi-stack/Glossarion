# Bubble/Free Text Overlap Fix - COMPLETE SOLUTION

## Problem
Free text and speech bubbles were overlapping, causing:
- Auction text merging with speech bubble text  
- Free text being included in bubble translations
- Incorrect rendering with bubble-style formatting applied to free text

## Visual Example
```
┌─────────────────┐
│  Speech Bubble  │ ← RT-DETR detects as "text_bubble" (Blue box)
│  "ANY OTHER     │
│   BIDDERS??"    │
└─────────────────┘

    ╔═════════════════════╗
    ║ ONE MILLION!        ║ ← RT-DETR detects large "text_free" region (Green box)
    ║ AN PACHI            ║    BUT also detects small "text_bubble" inside it!
    ║ HAS BEEN MADE!      ║
    ╚═════════════════════╝
         ↓ (overlaps)
    ┌─────────────┐
    │ ONE MILLION!│ ← Incorrectly classified as text_bubble (Blue box)
    └─────────────┘
```

## Root Cause Analysis

### Issue #1: RT-DETR Misclassification
RT-DETR may detect:
1. A large `text_free` region covering all free text (green box)
2. **ALSO** smaller `text_bubble` regions inside that free text area (blue boxes)

This creates overlapping classifications where the same text is marked as both bubble and free text.

### Issue #2: Bypassed Merging Logic
When using **RT-DETR Guided OCR** (the recommended mode):
- OCR is performed on each RT-DETR detected region separately
- Results are returned directly (line 2615/2266)
- The bubble merging logic at line 3554 is **never reached**
- Therefore, the free text exclusion logic (lines 1538-1640) is **never executed**

### Issue #3: Traditional Merging Path
For non-RT-DETR guided paths, the merging logic was:
```python
for bubble_idx, (bx, by, bw, bh) in enumerate(bubbles):
    for idx, region in enumerate(regions):
        region_center_x = rx + rw / 2
        region_center_y = ry + rh / 2
        
        if (bx <= region_center_x <= bx + bw and 
            by <= region_center_y <= by + bh):
            bubble_regions.append(region)  # ❌ MERGES EVERYTHING!
```

It didn't check if the region was also in a free text area.

## Complete Solution ✅

### PRIMARY FIX: RT-DETR Guided OCR Reclassification

Added post-processing after RT-DETR guided OCR to reclassify overlapping regions:

**Location**: Lines 2615-2638 (Azure) and 2268-2291 (Google)

```python
# POST-PROCESS: Check for text_bubbles that overlap with free_text regions
# If a text_bubble's center is within a free_text bbox, reclassify it as free_text
free_text_bboxes = rtdetr_detections.get('text_free', [])
if free_text_bboxes:
    reclassified_count = 0
    for region in regions:
        if getattr(region, 'bubble_type', None) == 'text_bubble':
            # Get region center
            x, y, w, h = region.bounding_box
            cx = x + w / 2
            cy = y + h / 2
            
            # Check if center is in any free_text bbox
            for (fx, fy, fw, fh) in free_text_bboxes:
                if fx <= cx <= fx + fw and fy <= cy <= fy + fh:
                    # Reclassify as free text
                    old_type = region.bubble_type
                    region.bubble_type = 'free_text'
                    reclassified_count += 1
                    self._log(f"🔄 Reclassified region '{region.text[:30]}...' from {old_type} to free_text", "debug")
                    break
    
    if reclassified_count > 0:
        self._log(f"🔄 Reclassified {reclassified_count} overlapping regions as free_text", "info")
```

**How it works:**
1. After RT-DETR guided OCR completes, iterate through all regions
2. For each region classified as `text_bubble`:
   - Calculate its center point (cx, cy)
   - Check if center falls inside any `text_free` bounding box
   - If yes → **reclassify as `free_text`**
3. Log the reclassification for debugging

### SECONDARY FIX: Traditional Merging Path Enhancement

Added free text exclusion logic for non-RT-DETR guided paths:

**Location**: Lines 1538-1640

```python
# Build lookup of free text regions for exclusion
free_text_bboxes = free_text_regions if detector_type in ('rtdetr', 'rtdetr_onnx') else []

# Helper to check if a point is in any free text region
def _point_in_free_text(cx, cy, free_boxes):
    for (fx, fy, fw, fh) in free_boxes or []:
        if fx <= cx <= fx + fw and fy <= cy <= fy + fh:
            return True
    return False

# During bubble merging
for bubble_idx, (bx, by, bw, bh) in enumerate(bubbles):
    for idx, region in enumerate(regions):
        region_center_x = rx + rw / 2
        region_center_y = ry + rh / 2
        
        if (bx <= region_center_x <= bx + bw and 
            by <= region_center_y <= by + bh):
            
            # ✅ NEW: Don't merge if this region is in a free text area
            if _point_in_free_text(region_center_x, region_center_y, free_text_bboxes):
                continue  # Skip merging
            
            bubble_regions.append(region)
            used_indices.add(idx)
```

### TERTIARY FIX: Enhanced Debug Logging

Added detailed logging to help diagnose issues:

**Location**: Lines 1541-1558, 1562-1586

```python
# Log free text exclusion zones
if free_text_bboxes:
    self._log(f"🔍 Free text exclusion zones: {len(free_text_bboxes)} regions", "debug")
    for idx, (fx, fy, fw, fh) in enumerate(free_text_bboxes):
        self._log(f"   Free text zone {idx + 1}: x={fx:.0f}, y={fy:.0f}, w={fw:.0f}, h={fh:.0f}", "debug")
else:
    self._log(f"⚠️ No free text exclusion zones detected by RT-DETR", "warning")

# Log bubble processing
for bubble_idx, (bx, by, bw, bh) in enumerate(bubbles):
    self._log(f"\n   Processing bubble {bubble_idx + 1}: x={bx:.0f}, y={by:.0f}, w={bw:.0f}, h={bh:.0f}", "debug")
    
    # Log region checks
    self._log(f"      Region '{region.text[:20]}...' center ({region_center_x:.0f}, {region_center_y:.0f}) is in bubble", "debug")
    
    if _point_in_free_text(region_center_x, region_center_y, free_text_bboxes):
        self._log(f"      ❌ SKIPPING: Region overlaps with free text area", "debug")
        continue
    
    self._log(f"      ✓ Adding region to bubble {bubble_idx + 1}", "debug")
```

## How It Works Now

### RT-DETR Guided OCR Path (Recommended)
1. **Detection Phase:**
   - RT-DETR detects `text_bubbles` (blue) and `text_free` (green) regions
   - May detect overlapping regions (e.g., small blue box inside large green box)

2. **OCR Phase:**
   - Azure/Google OCR processes each detected region
   - Regions are initially classified based on RT-DETR detection type

3. **🆕 Post-Processing Phase:**
   - Check all `text_bubble` regions for overlap with `text_free` areas
   - If a text_bubble's center is inside a text_free bbox → **reclassify as free_text**
   - This fixes RT-DETR misclassifications

4. **Result:**
   - ✅ Correctly classified free text (no bubble formatting)
   - ✅ Correctly classified bubble text (bubble formatting)
   - ✅ No overlap or contamination

### Traditional Merging Path (Fallback)
1. **Detection Phase:**
   - RT-DETR detects bubble and free text regions
   - OCR detects all text (doesn't know types)

2. **Merging Phase:**
   - For each bubble bbox:
     - Find OCR regions with center inside bubble
     - **🆕 Check:** Is center also in a free_text bbox?
       - YES → **SKIP** (don't merge into bubble)
       - NO → Merge into bubble
   
3. **Free Text Assignment:**
   - All unmerged regions → mark as `bubble_type = 'free_text'`

4. **Result:**
   - ✅ Bubble text merged together
   - ✅ Free text stays separate

## Files Modified

- **manga_translator.py**
  - **Lines 2615-2638**: Azure RT-DETR guided OCR - post-process reclassification
  - **Lines 2268-2291**: Google RT-DETR guided OCR - post-process reclassification
  - Lines 1538-1558: Free text lookup helper and debug logging
  - Lines 1562-1586: Enhanced bubble merging with exclusion checks
  - Lines 1621-1640: Improved free text region marking

## Testing

After this fix, verify:

### 1. RT-DETR Guided OCR (Primary Path)
Look for reclassification logs:
```
✅ RT-DETR + Azure Vision: 10 text regions detected
🔄 Reclassified region 'ONE MILLION!...' from text_bubble to free_text (overlaps with free text area)
🔄 Reclassified 2 overlapping regions as free_text
```

### 2. Traditional Merging (Fallback Path)
Look for exclusion logs:
```
🔍 Free text exclusion zones: 1 regions
   Free text zone 1: x=100, y=150, w=200, h=80
   
   Processing bubble 1: x=50, y=50, w=200, h=100
      Region 'ONE MILLION...' center (150, 180) is in bubble
      ✓ Point (150, 180) is in free text zone 1
      ❌ SKIPPING: Region overlaps with free text area
```

### 3. Mask Creation
```
📊 Mask breakdown: 8 text bubbles, 0 empty bubbles, 2 free text regions, 0 skipped
```
- Text bubbles and free text should be properly separated
- Counts should match RT-DETR detections

### 4. Final Rendering
- ✅ Bubble text: Rendered inside bubble with proper formatting
- ✅ Free text: Rendered with background opacity (if enabled)
- ✅ No text mixing between bubble and free text

## Related Issues

This fix complements:
- `REGION_MAPPING_FIX_SUMMARY.md` - Region type tracking in RT-DETR path
- `REGION_MAPPING_CACHE_ISSUE.md` - Cache key separation by type

Together these ensure:
- ✅ Proper type detection (RT-DETR)
- ✅ Proper type preservation (cache)
- ✅ **Proper type correction (reclassification)** ← **THIS FIX**
- ✅ Proper type separation (merging)
- ✅ Proper type application (rendering/masking)

---

**Date**: 2025-01-04  
**Issue**: Free text being misclassified as bubbles due to RT-DETR overlap  
**Status**: ✅ **FIXED**  
**Result**: Overlapping regions are correctly reclassified, bubble and free text remain separate
