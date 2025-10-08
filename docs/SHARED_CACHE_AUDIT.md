# Shared Cache Audit Report - Manga Translation Pipeline

## Executive Summary

**ISSUE**: Text-related caches were being shared across images, potentially causing text contamination between different manga pages.

**STATUS**: ✅ **FIXED** - All text-related caches are now isolated per image with aggressive clearing.

---

## Caches Found & Fixed

### 🔴 CRITICAL - Text Contamination Risk (NOW FIXED)

#### 1. **OCR ROI Cache** (`manga_translator.py` line 234)
- **Type**: `dict` - Maps (image_hash, bbox, provider, lang, mode) → text
- **Risk**: HIGH - Directly stores OCR text results
- **Used in**: Google/Azure ROI-based OCR batching
- **Fix Applied**: 
  - Cleared before every image (line 1997-1999)
  - Cleared on image hash change (line 2059-2062)
  - Cleared in reset_for_new_image (line 8628-8630)
  - Cleared in clear_internal_state (line 9311-9312)

#### 2. **OCR Manager Caches** (`manga_translator.py` via OCR manager)
- **Locations**:
  - `ocr_manager.last_results` - Last OCR results
  - `ocr_manager.cache` - Generic cache dict
  - `provider.last_results` - Per-provider last results
  - `provider.cache` - Per-provider cache dict
- **Risk**: HIGH - Can store text from previous images
- **Fix Applied**:
  - All cleared before every image (lines 2002-2016)
  - All cleared in reset_for_new_image (lines 8633-8644)

#### 3. **Bubble Detector Cache** (`manga_translator.py` via bubble detector)
- **Locations**:
  - `bubble_detector.last_detections` - Last detection results
  - `bubble_detector.cache` - Generic cache dict
- **Risk**: MEDIUM - Stores text region locations/metadata
- **Fix Applied**:
  - Cleared before every image (lines 2019-2024)
  - Cleared in reset_for_new_image (lines 8647-8651)

---

## Other Caches Identified

### 🟡 MEDIUM RISK - Monitored but Not Text-Related

#### 4. **Generic Cache** (`manga_translator.py` line 260)
- **Type**: `dict` - Purpose unclear
- **Risk**: MEDIUM - Could be used for anything
- **Status**: Documented as "DO NOT USE FOR TEXT DATA"
- **Action**: Added warning comments, cleared in clear_internal_state

#### 5. **Translation History** (`manga_translator.py`)
- **Location**: `self.translation_context`
- **Risk**: LOW - This is INTENDED to be shared (rolling history feature)
- **Status**: Intentionally preserved when `rolling_history_enabled=True`
- **Action**: Only cleared when NOT in rolling history mode

---

## Singleton/Shared Model Instances (NOT CACHES)

### 🟢 LOW RISK - These are models, not text caches

These are intentionally shared for memory efficiency and do NOT store text:

1. **Bubble Detector Shared Models** (`bubble_detector.py`)
   - `BubbleDetector._rtdetr_shared_model` (RT-DETR model weights)
   - `BubbleDetector._rtdetr_shared_processor` (Image processor)
   - `BubbleDetector._rtdetr_onnx_shared_session` (ONNX session)
   - **Safe**: These are neural network models, not text caches

2. **Inpainter Pool** (`manga_translator.py`)
   - `MangaTranslator._inpaint_pool` (Shared inpainter instances)
   - **Safe**: These are image processing models, not text caches

3. **OCR Provider Models** (`ocr_manager.py`)
   - Various provider models (manga-ocr, easyocr, etc.)
   - **Safe**: These are OCR models themselves, not result caches

---

## Cache Clearing Strategy

### Three-Tier Defense System:

#### Tier 1: Pre-Processing (Every Image)
```python
# In detect_text_regions() - Line 1989+
- Clear ocr_roi_cache
- Clear ocr_manager caches (all providers)
- Clear bubble_detector caches
```

#### Tier 2: Image Change Detection
```python
# In detect_text_regions() - Line 2058+
if image_hash_changed:
    - Clear ocr_roi_cache
```

#### Tier 3: Manual Reset
```python
# In reset_for_new_image() - Line 8617+
# In clear_internal_state() - Line 9243+
- Clear ALL caches comprehensively
```

---

## Architecture Decisions

### ✅ What We Changed

1. **Removed Batch Mode Exception**
   - OLD: Skip cache clearing in batch mode (DANGEROUS!)
   - NEW: Clear ALL caches for EVERY image, regardless of mode

2. **Added Provider-Level Clearing**
   - OLD: Only cleared OCR manager level
   - NEW: Clear each provider's caches individually

3. **Added Image Hash Tracking**
   - NEW: Automatic detection of image changes
   - NEW: Auto-invalidation of ROI cache on change

4. **Multiple Clearing Points**
   - Redundancy ensures no cache survives
   - Defense-in-depth approach

### ❌ What We Did NOT Change

1. **Model Instances** - Still shared for memory efficiency
2. **Translation History** - Still shared when rolling history enabled
3. **Model Pools** - Still shared for performance

---

## Verification Steps

To verify cache isolation is working:

1. **Check Debug Logs** - Look for these messages:
   ```
   🧹 Cleared OCR ROI cache
   🧹 Cleared OCR manager caches
   🧹 Cleared bubble detector cache
   🧹 Image changed - cleared ROI cache
   🔄 Reset translator state for new image (ALL text caches cleared)
   ```

2. **Test Sequence**:
   - Process Image A with text "안녕하세요"
   - Process Image B with text "こんにちは"
   - Verify Image B does NOT contain "안녕하세요"

3. **Monitor Memory**:
   - Cache clearing should prevent memory growth
   - Models can grow memory (expected)
   - Text caches should NOT accumulate

---

## Parallel Panel Translation Safety

### ✅ COMPLETELY SAFE

When `parallel_panel_translation=True`, each panel gets its **own isolated MangaTranslator instance**:

```python
# Each thread creates its own translator
translator = MangaTranslator(ocr_config, self.main_gui.client, self.main_gui, log_callback=self._log)
```

**This means**:
- ✅ Each panel has its **own separate `ocr_roi_cache`** (no sharing!)
- ✅ Each panel has its **own separate `ocr_manager`** (with its own providers)
- ✅ All cache operations use `threading.Lock()` for atomicity
- ✅ Shared model weights are **read-only** (safe to share)
- ✅ Model pools use **locks** (safe to share)

**See**: `PARALLEL_PANEL_SAFETY.md` for detailed analysis.

---

## Files Modified

1. `manga_translator.py` - 14 edit locations (7 cache isolation + 7 thread safety)
2. `CACHE_ISOLATION_FIXES.md` - Created (detailed changes)
3. `SHARED_CACHE_AUDIT.md` - Created (this file)  
4. `PARALLEL_PANEL_SAFETY.md` - Created (parallel mode analysis)
5. `CACHE_ISOLATION_QUICKREF.md` - Created (developer guide)

---

## Conclusion

✅ **ALL TEXT-RELATED CACHES ARE NOW ISOLATED PER IMAGE**

- OCR ROI cache: Isolated ✅
- OCR manager caches: Isolated ✅
- Provider-level caches: Isolated ✅
- Bubble detector cache: Isolated ✅

**NO SHARING. NO EXCEPTIONS. NO TEXT LEAKAGE.**

The pipeline now has multiple redundant safeguards to prevent any text data from leaking between images, even if ROI-based OCR is re-enabled in the future.
