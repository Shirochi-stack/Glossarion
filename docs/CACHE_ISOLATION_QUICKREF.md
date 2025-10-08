# Cache Isolation - Quick Reference Guide

## ⚠️ CRITICAL RULE: NO SHARED TEXT CACHES

**Any cache that stores OCR text, translations, or text regions MUST be cleared between images.**

---

## 🚫 DON'T DO THIS (Anti-Patterns)

```python
# ❌ WRONG: Persistent text cache across images
class MyOCRProvider:
    def __init__(self):
        self.text_cache = {}  # This will leak between images!
    
    def ocr(self, image):
        if image_hash in self.text_cache:
            return self.text_cache[image_hash]  # DANGEROUS
```

```python
# ❌ WRONG: Conditional cache clearing
if not self.batch_mode:  # DANGEROUS - batch mode needs clearing too!
    self.text_cache.clear()
```

```python
# ❌ WRONG: Only clearing manager-level cache
self.ocr_manager.cache.clear()
# Forgot to clear provider.cache! Text can still leak.
```

---

## ✅ DO THIS (Correct Patterns)

```python
# ✅ CORRECT: Clear ALL caches before processing
def process_new_image(self, image_path):
    # ALWAYS clear - no exceptions
    if hasattr(self, 'text_cache'):
        self.text_cache.clear()
    
    # Clear manager AND provider caches
    if hasattr(self, 'ocr_manager'):
        if hasattr(self.ocr_manager, 'cache'):
            self.ocr_manager.cache.clear()
        # Don't forget providers!
        if hasattr(self.ocr_manager, 'providers'):
            for provider in self.ocr_manager.providers.values():
                if hasattr(provider, 'cache'):
                    provider.cache.clear()
```

```python
# ✅ CORRECT: Track image changes
def process_image(self, image_data):
    new_hash = hashlib.sha1(image_data).hexdigest()
    
    if self._current_hash != new_hash:
        # Image changed - invalidate ALL caches
        self._clear_all_text_caches()
        self._current_hash = new_hash
```

```python
# ✅ CORRECT: Multiple clearing points
def detect_text(self, image):
    self._clear_text_caches()  # Before processing
    # ... do OCR ...
    
def reset_for_new_image(self):
    self._clear_text_caches()  # Manual reset
    
def cleanup(self):
    self._clear_text_caches()  # Final cleanup
```

---

## 📋 Checklist for Adding New OCR/Text Features

Before committing code that touches text/OCR:

- [ ] Does it create any `dict`, `list`, or variable that stores text?
- [ ] Is that storage shared across multiple images?
- [ ] Have you added clearing logic in `detect_text_regions()`?
- [ ] Have you added clearing logic in `reset_for_new_image()`?
- [ ] Have you added clearing logic in `clear_internal_state()`?
- [ ] Does it work in BOTH batch mode and single-image mode?
- [ ] Have you tested with 3+ sequential images to verify no leakage?

---

## 🔍 Where to Clear Caches

### In `manga_translator.py`:

1. **detect_text_regions()** (line ~1989)
   - Clear BEFORE processing each image
   - No exceptions for batch mode

2. **reset_for_new_image()** (line ~8617)
   - Clear when manually resetting state
   - Clear image hash tracker

3. **clear_internal_state()** (line ~9243)
   - Clear during comprehensive cleanup
   - Clear image hash tracker

### In `ocr_manager.py`:

- Clear in provider's `detect_text()` if needed
- Never persist text results in instance variables

### In `bubble_detector.py`:

- Clear `last_detections` before each detection
- Don't cache text region content, only metadata

---

## 🧪 How to Test Your Changes

```python
# Test sequence to verify isolation
def test_cache_isolation():
    translator = MangaTranslator(...)
    
    # Process Image 1 with unique text
    result1 = translator.translate_image("image1_with_korean.png")
    korean_text = "안녕하세요"  # Expected in result1
    
    # Process Image 2 with different text
    result2 = translator.translate_image("image2_with_japanese.png")
    japanese_text = "こんにちは"  # Expected in result2
    
    # CRITICAL: result2 should NOT contain korean_text
    assert korean_text not in str(result2), "TEXT LEAKED FROM IMAGE 1!"
    
    # Process Image 3
    result3 = translator.translate_image("image3_with_chinese.png")
    
    # Verify no leakage from previous images
    assert korean_text not in str(result3), "Korean text leaked!"
    assert japanese_text not in str(result3), "Japanese text leaked!"
```

---

## 🎯 Key Locations in Code

| File | Lines | What Gets Cleared |
|------|-------|-------------------|
| `manga_translator.py` | 1997-1999 | OCR ROI cache |
| `manga_translator.py` | 2002-2016 | OCR manager + all providers |
| `manga_translator.py` | 2019-2024 | Bubble detector cache |
| `manga_translator.py` | 2059-2062 | Auto-clear on image change |
| `manga_translator.py` | 8628-8651 | Comprehensive reset |
| `manga_translator.py` | 9311-9313 | Final cleanup |

---

## 🆘 Debugging Cache Issues

If you suspect text is leaking between images:

1. **Enable Debug Logging**:
   ```python
   self._log("🧹 Cleared OCR ROI cache", "debug")
   ```

2. **Check for These Messages**:
   - "🧹 Cleared OCR ROI cache"
   - "🧹 Cleared OCR manager caches"
   - "🧹 Cleared bubble detector cache"
   - "🧹 Image changed - cleared ROI cache"

3. **Add Assertions**:
   ```python
   assert len(self.ocr_roi_cache) == 0, "Cache not cleared!"
   ```

4. **Check All Providers**:
   ```python
   for name, provider in self.ocr_manager.providers.items():
       if hasattr(provider, 'cache'):
           print(f"Provider {name} cache size: {len(provider.cache)}")
   ```

---

## 📚 Related Documentation

- `CACHE_ISOLATION_FIXES.md` - Detailed changes made
- `SHARED_CACHE_AUDIT.md` - Complete audit of all caches

---

**Remember**: When in doubt, CLEAR THE CACHE. Better safe than sorry!
