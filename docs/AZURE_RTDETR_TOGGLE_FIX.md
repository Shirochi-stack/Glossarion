# Azure Providers - RT-DETR Toggle Fix

## ✅ **FIXED: Azure Providers Now Respect "RT-DETR to Guide" Toggle**

### Problem You Reported

When you **disable the "RT-DETR to guide" toggle** (`use_rtdetr_for_ocr_regions = False`), Azure providers were still doing **per-region cropping** instead of simple full-page OCR.

### Root Causes Found

#### **Issue #1: Azure Document Intelligence**
- Line 4111 only checked `bubble_detection_enabled`
- Did NOT check `use_rtdetr_for_ocr_regions`
- Result: Even when toggle was off, it still tried RT-DETR matching

#### **Issue #2: Azure Vision ROI-Based Path**
- Line 3255 (ROI-based concurrent OCR) only checked `bubble_detection_enabled` and `roi_locality_enabled`
- Did NOT check `use_rtdetr_for_ocr_regions`
- Result: When toggle was off, it fell through to per-region cropping instead of full-page OCR

---

## ✅ **Solutions Implemented**

### Fix #1: Azure Document Intelligence Check (Line 4111)

**Before:**
```python
if ocr_settings.get('bubble_detection_enabled', False):
    # Always tried to use RT-DETR matching if bubble detection enabled
```

**After:**
```python
if ocr_settings.get('bubble_detection_enabled', False) and ocr_settings.get('use_rtdetr_for_ocr_regions', True):
    # Now respects the RT-DETR toggle!
```

### Fix #2: Azure Vision ROI-Based Check (Line 3255-3257)

**Before:**
```python
use_roi_locality = (ocr_settings.get('bubble_detection_enabled', False) and 
                   ocr_settings.get('roi_locality_enabled', False))
```

**After:**
```python
use_roi_locality = (ocr_settings.get('bubble_detection_enabled', False) and 
                   ocr_settings.get('roi_locality_enabled', False) and 
                   ocr_settings.get('use_rtdetr_for_ocr_regions', True))
```

---

## 🎯 **Expected Behavior After Fix**

### ✅ **When "RT-DETR to guide" is ENABLED** (`use_rtdetr_for_ocr_regions = True`)

**Both Azure providers:**
```
🎯 Azure [Vision/Doc Intelligence] full image → match to RT-DETR blocks
📊 Step 1: Running Azure [Vision/Doc Intelligence] on full image
✅ Azure detected X text lines in full image
🔗 Step 2: Matching X OCR lines to Y RT-DETR blocks
✅ Matched text to Y RT-DETR blocks (comic-translate style)
```

**Result:** Full-image OCR + RT-DETR matching approach (optimal)

---

### ✅ **When "RT-DETR to guide" is DISABLED** (`use_rtdetr_for_ocr_regions = False`)

**Azure Vision:**
```
📝 Using Azure Image Analysis API for OCR
🌐 Azure language: ja
[Processes full page without RT-DETR]
✅ Detected X text regions after merging
```

**Azure Document Intelligence:**
```
📋 Azure Document Intelligence OCR (successor to Azure AI Vision)
📝 Processing full image with Azure Document Intelligence
[Processes full page without RT-DETR]
✅ Detected X text regions after merging
```

**Result:** Simple full-page OCR without RT-DETR (clean fallback)

---

## 📊 **Behavior Matrix**

| Toggle State | `bubble_detection_enabled` | `use_rtdetr_for_ocr_regions` | Azure Behavior |
|--------------|---------------------------|------------------------------|----------------|
| ✅ RT-DETR ON | True | True | Full-image OCR + RT-DETR matching |
| ❌ RT-DETR OFF | True | False | Full-page OCR (no RT-DETR) |
| ❌ No Bubbles | False | N/A | Full-page OCR (no RT-DETR) |

---

## 🐛 **Before vs After**

### Before the Fix:

#### Scenario: Disable "RT-DETR to guide"

**Azure Document Intelligence:**
```
✗ Still entered RT-DETR matching block
✗ Tried to match OCR to RT-DETR regions
✗ Confusing behavior
```

**Azure Vision:**
```
✗ Fell through to ROI-based concurrent path
✗ Did per-region cropping (lines 3267-3291)
✗ Multiple API calls to crop regions
✗ Not the expected "simple full-page OCR"
```

### After the Fix:

#### Scenario: Disable "RT-DETR to guide"

**Azure Document Intelligence:**
```
✓ Skips RT-DETR matching block
✓ Goes to else block (line 4178)
✓ Does simple full-page OCR (line 4183)
✓ Clean, expected behavior
```

**Azure Vision:**
```
✓ Skips RT-DETR matching block (line 3146)
✓ Skips ROI-based path (line 3255 condition fails)
✓ Falls through to full-page Azure Vision OCR (line 3409+)
✓ Clean, expected behavior
```

---

## 🔍 **What Each Code Path Does**

### Path 1: RT-DETR Matching (Lines 3146-3241 for Azure Vision)
**When:** `bubble_detection_enabled = True` AND `use_rtdetr_for_ocr_regions = True`

**What it does:**
1. Runs RT-DETR to detect bubble regions
2. Runs Azure OCR on FULL image
3. Matches Azure OCR text lines to RT-DETR bubble regions
4. Returns results aligned to bubbles

**Use case:** Optimal manga OCR with bubble detection

---

### Path 2: ROI-Based Concurrent (Lines 3255-3291 for Azure Vision)
**When:** `bubble_detection_enabled = True` AND `roi_locality_enabled = True` AND `use_rtdetr_for_ocr_regions = True` AND batching enabled

**What it does:**
1. Runs RT-DETR to detect bubble regions
2. Crops each region with padding
3. Runs Azure OCR on each cropped region concurrently
4. Returns results from multiple API calls

**Use case:** Advanced batched processing (rarely used)

---

### Path 3: Full-Page OCR (Lines 3409+ for Azure Vision)
**When:** Neither Path 1 nor Path 2 conditions are met

**What it does:**
1. Runs Azure OCR on the full page
2. Returns all detected text regions
3. No RT-DETR involvement

**Use case:** Simple full-page OCR, or when RT-DETR toggle is disabled

---

## ⚙️ **Configuration Examples**

### Example 1: Optimal Manga OCR (RT-DETR + Azure)

```json
{
  "bubble_detection_enabled": true,
  "use_rtdetr_for_ocr_regions": true,
  "roi_locality_enabled": false
}
```

**Result:** Full-image OCR + RT-DETR matching (Path 1) ✅

---

### Example 2: Simple Full-Page OCR (No RT-DETR)

```json
{
  "bubble_detection_enabled": true,
  "use_rtdetr_for_ocr_regions": false,
  "roi_locality_enabled": false
}
```

**Result:** Simple full-page OCR (Path 3) ✅

---

### Example 3: No Bubble Detection

```json
{
  "bubble_detection_enabled": false,
  "use_rtdetr_for_ocr_regions": false,
  "roi_locality_enabled": false
}
```

**Result:** Simple full-page OCR (Path 3) ✅

---

## 📝 **Files Modified**

1. **manga_translator.py line 3255-3257**: Added `use_rtdetr_for_ocr_regions` check to ROI-based path
2. **manga_translator.py line 4111**: Added `use_rtdetr_for_ocr_regions` check to Azure Doc Intelligence
3. **manga_translator.py line 3249-3253**: Added clarifying comments

---

## ✅ **Summary**

**The Fix:**
- Both Azure providers now **consistently check** the `use_rtdetr_for_ocr_regions` toggle
- Disabling "RT-DETR to guide" now **correctly** results in simple full-page OCR
- No more unexpected per-region cropping when the toggle is off

**Test It:**
1. Enable bubble detection, disable "RT-DETR to guide"
2. Run Azure OCR
3. Check logs for "📝 Processing full image with Azure..." (not region cropping messages)
4. Should see simple full-page OCR behavior ✅

**Your Azure providers now behave predictably based on the toggle state!** 🚀
