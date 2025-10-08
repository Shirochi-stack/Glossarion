# Quick Monitoring Guide - Azure OCR Fixes

## What to Watch For

### ✅ Signs the Fixes are Working

When processing 10 images, you should see in the logs:

```
🔍 [1/10] image_001.png
📊 Azure ROI Cache: 0/5 hits (0.0%), page_hash=a3f5d8c1...
✅ Sorted 5 Azure results by position (top-to-bottom, left-to-right)
🧹 Cleared ROI cache after processing

🔍 [2/10] image_002.png
📊 Azure ROI Cache: 0/5 hits (0.0%), page_hash=b7e2f9d4...  ← Different hash!
✅ Sorted 5 Azure results by position (top-to-bottom, left-to-right)
🧹 Cleared ROI cache after processing

🔍 [3/10] image_003.png
📊 Azure ROI Cache: 0/5 hits (0.0%), page_hash=c9a1b5e6...  ← Different hash!
...
```

**Key indicators:**
- ✅ `page_hash` changes for each image (first 8 chars differ)
- ✅ Cache hit rate is 0% for each NEW image
- ✅ "Cleared ROI cache after processing" appears after each image
- ✅ Results are sorted by position

---

### ⚠️ Warning Signs (Investigate)

```
⚠️ Using fallback page hash for cache isolation
```
→ SHA1 computation returned None somehow (rare)  
→ Check if `processed_image_data` is valid  
→ Fallback is working, but investigate root cause

```
⚠️ Page hash generation failed: {error}, using UUID fallback
```
→ Exception during hash computation  
→ Emergency fallback activated  
→ Check the error message for clues

```
📊 Azure ROI Cache: 5/5 hits (100.0%), page_hash=a3f5d8c1...
📊 Azure ROI Cache: 5/5 hits (100.0%), page_hash=a3f5d8c1...  ← SAME hash!
```
→ Two different images have identical hash (should be rare)  
→ Verify images are actually different  
→ Or compression settings changed image to identical bytes

---

### ❌ Red Flags (BUG - Report Immediately)

```
📊 Azure ROI Cache: 3/5 hits (60.0%), page_hash=None...
```
→ **CRITICAL:** page_hash is None despite fixes!  
→ Both Fix #1 fallbacks failed  
→ Report immediately with full log context

```
🧹 Cleared OCR ROI cache
[No "Cleared ROI cache after processing" at end]
```
→ Fix #2 not executing (finally block skipped somehow)  
→ Check for early returns or exceptions  
→ Cache may persist across images

```
💾 Cached[0]: 'Text from previous image...'
```
(On a fresh image that should have 0% cache hit)  
→ Cache not cleared between images  
→ Text contamination occurring  
→ Report with full log and images

---

## Quick Debug Commands

### Check page_hash uniqueness:
```powershell
# Extract all page_hash values from log
Select-String -Path "translation.log" -Pattern "page_hash=(\w+)" | 
    ForEach-Object { $_.Matches.Groups[1].Value } | 
    Group-Object | 
    Where-Object { $_.Count -gt 1 }
```
If any hash appears more than once, investigate those images.

### Check cache hit rates:
```powershell
# Extract cache statistics
Select-String -Path "translation.log" -Pattern "Cache: (\d+)/(\d+) hits"
```
Expected: 0% for each new image, >0% only if re-processing same image.

### Check cache clearing:
```powershell
# Count cache clears
(Select-String -Path "translation.log" -Pattern "Cleared ROI cache").Count
```
Expected: At least 2× the number of images (once at start, once at end).

---

## Testing Scenarios

### Scenario 1: Identical Bubble Positions
**Purpose:** Test cache key uniqueness

1. Create 3 images with bubble at (100, 200, 300, 150)
2. Each has different text: "Test 1", "Test 2", "Test 3"
3. Process all 3
4. **Expected:** Each outputs correct text
5. **Log should show:** 3 different page_hash values

### Scenario 2: Rapid Sequential Processing
**Purpose:** Test cache clearing between images

```python
translator.detect_text_regions("image_a.png")
# Immediately process different image
translator.detect_text_regions("image_b.png")
```

**Expected:** 
- Image B has 0% cache hit rate
- No text from Image A appears in Image B
- Two "Cleared ROI cache after processing" messages

### Scenario 3: Batch Processing
**Purpose:** Test full pipeline with 10 images

Process 10 different manga pages:
- **Expected:** 10 different page_hash values
- **Expected:** No cache hits across images
- **Expected:** 20 cache clear messages (2 per image)

---

## Performance Benchmarks

### Typical Processing Time (per image)
- **ROI Preparation:** 0.1-0.5s
- **Azure OCR (5 ROIs):** 2-5s
- **Cache Operations:** <0.01s (negligible)
- **Sorting Results:** <0.01s (negligible)

### Cache Hit Rate Expectations
| Scenario | Expected Hit Rate |
|----------|------------------|
| First image | 0% |
| Re-processing same image | 100% |
| Different images | 0% |
| Identical images (same content) | 0% first, 100% second |

---

## Common False Positives

### "Same text in multiple panels"
If the **actual manga** has repeated text:
- ✅ EXPECTED: Same text appears in multiple panels
- ❌ NOT A BUG: This is correct OCR behavior

Example: Character saying "はい" (yes) in multiple panels → correct to see "はい" multiple times.

### "Cache hit rate >0% on second image"
If images are truly identical (duplicate files):
- ✅ EXPECTED: Cache reuse for identical content
- ❌ NOT A BUG: Optimization working as intended

---

## Emergency Rollback

If contamination still occurs after fixes:

1. **Quick disable cache:**
   ```python
   # In manga_translator.py __init__, add:
   self.ocr_roi_cache = None  # Disable cache entirely
   ```

2. **Check Git status:**
   ```powershell
   git diff manga_translator.py | Select-String -Pattern "page_hash|cache|sort"
   ```

3. **Revert fixes:**
   ```powershell
   git checkout HEAD -- manga_translator.py
   ```

---

## Support Checklist

When reporting issues, provide:

- [ ] Full log file (with debug enabled)
- [ ] Sample images that reproduce issue (2-3 minimum)
- [ ] Output showing wrong text placement
- [ ] Cache statistics from logs
- [ ] page_hash values for affected images
- [ ] Parallel panel translation setting (enabled/disabled)
- [ ] Azure tier (Free/Standard)
- [ ] Number of images processed simultaneously

---

## Success Criteria

✅ **Fixes are working if:**
1. Each image has unique page_hash (first 8 chars differ)
2. Cache hit rate is 0% for each new image
3. No text from Image N appears in Image N+1
4. Results are consistently ordered top-to-bottom
5. No "None" page_hash warnings

✅ **Issue is resolved if:**
- Processing 10 images → Each has correct text
- No cross-contamination observed in 100+ image batch
- Cache statistics show proper isolation
