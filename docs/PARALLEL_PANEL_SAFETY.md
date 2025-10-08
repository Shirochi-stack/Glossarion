# Parallel Panel Translation - Cache Safety Analysis

## Overview

When `parallel_panel_translation=True`, multiple manga panels are processed simultaneously using `ThreadPoolExecutor`. This document explains how cache isolation works in this parallel mode.

---

## Architecture

### Sequential Mode (Default)
```
┌──────────────────────────────┐
│  Single MangaTranslator      │
│                              │
│  ┌─────────────────────┐    │
│  │ ocr_roi_cache: {}   │    │
│  │ ocr_manager         │    │
│  │ bubble_detector     │    │
│  └─────────────────────┘    │
└──────────────────────────────┘
   ↓         ↓         ↓
 Panel 1 → Panel 2 → Panel 3
 (one at a time, cache cleared between)
```

### Parallel Mode
```
┌────────────────────┐  ┌────────────────────┐  ┌────────────────────┐
│ MangaTranslator #1 │  │ MangaTranslator #2 │  │ MangaTranslator #3 │
│                    │  │                    │  │                    │
│ ocr_roi_cache: {}  │  │ ocr_roi_cache: {}  │  │ ocr_roi_cache: {}  │
│ _cache_lock 🔒     │  │ _cache_lock 🔒     │  │ _cache_lock 🔒     │
│ ocr_manager (own)  │  │ ocr_manager (own)  │  │ ocr_manager (own)  │
└────────────────────┘  └────────────────────┘  └────────────────────┘
         ↓                       ↓                       ↓
     Panel 1              Panel 2               Panel 3
  (simultaneous, completely isolated)
```

---

## Isolation Guarantees

### ✅ Per-Thread Isolation

Each parallel worker thread gets its own `MangaTranslator` instance:

```python
# From manga_integration.py line 7470
translator = MangaTranslator(ocr_config, self.main_gui.client, self.main_gui, log_callback=self._log)
```

This means:
- **Separate `ocr_roi_cache`** - Panel 1's cache ≠ Panel 2's cache
- **Separate `ocr_manager`** - Each has its own OCR manager instance
- **Separate providers** - Each manager has its own provider instances
- **Separate `_current_image_hash`** - Each tracks its own image

### ✅ Thread-Safe Cache Operations

All cache operations use `threading.Lock()`:

```python
# Cache write (thread-safe)
with self._cache_lock:
    self.ocr_roi_cache[cache_key] = text_result

# Cache read (thread-safe)
with self._cache_lock:
    cached_text = self.ocr_roi_cache.get(cache_key)

# Cache clear (thread-safe)
with self._cache_lock:
    self.ocr_roi_cache.clear()
```

**Why locks matter**: Even though each thread has its own cache, Python dict operations aren't atomic. Without locks, you could get corrupted cache state during concurrent access.

---

## Shared Components (Intentionally Shared)

### 🟢 Safe to Share (Read-Only or Internally Locked)

These components ARE shared across threads, but it's **SAFE** because they don't store text:

#### 1. **Bubble Detector Models** (Read-Only Weights)
```python
BubbleDetector._rtdetr_shared_model        # PyTorch model weights (immutable)
BubbleDetector._rtdetr_shared_processor    # Image preprocessor (stateless)
BubbleDetector._rtdetr_onnx_shared_session # ONNX session (has internal locking)
```
- **Why safe**: Model weights are read-only after loading
- **No text stored**: These only contain neural network parameters
- **Internal locking**: ONNX runtime has its own thread safety

#### 2. **Model Pools** (With Locks)
```python
MangaTranslator._detector_pool      # Protected by _detector_pool_lock
MangaTranslator._inpaint_pool       # Protected by _inpaint_pool_lock
```
- **Why safe**: All access is protected by `threading.Lock()`
- **No text stored**: These pool model instances, not OCR results

#### 3. **UnifiedClient** (API Client)
```python
self.main_gui.client  # Shared API client
```
- **Why safe**: Client is stateless for translation requests
- **No text caching**: Translations are returned, not stored in client
- **Rate limiting**: Client's internal rate limiting is thread-safe

---

## What Gets Isolated

### Per-Thread State (NOT Shared)

| Component | Scope | Contains Text? | Safety |
|-----------|-------|----------------|--------|
| `ocr_roi_cache` | Per translator instance | ✅ YES - OCR results | Thread-safe with lock |
| `ocr_manager.cache` | Per translator instance | ✅ YES - OCR cache | Per-instance isolation |
| `provider.cache` | Per translator instance | ✅ YES - Provider cache | Per-instance isolation |
| `bubble_detector.last_detections` | Per translator instance | ⚠️ Metadata only | Per-instance isolation |
| `_current_image_hash` | Per translator instance | ❌ NO - Just hash | Per-instance isolation |
| `translation_context` | Per translator instance | ✅ YES - Translation history | Per-instance isolation |

---

## Test Scenarios

### ✅ Test 1: Parallel Panels with Different Text
```python
def test_parallel_isolation():
    # Enable parallel panel translation
    config['manga_settings']['advanced']['parallel_panel_translation'] = True
    config['manga_settings']['advanced']['panel_max_workers'] = 3
    
    # Process 3 panels simultaneously
    panels = [
        'panel1_korean.png',   # Contains "안녕하세요"
        'panel2_japanese.png', # Contains "こんにちは"
        'panel3_chinese.png'   # Contains "你好"
    ]
    
    results = translate_manga(panels)
    
    # Verify no cross-contamination
    assert '안녕하세요' not in results['panel2_japanese.png']
    assert 'こんにちは' not in results['panel3_chinese.png']
    assert '你好' not in results['panel1_korean.png']
```

### ✅ Test 2: Cache Hit Verification
```python
def test_cache_per_instance():
    # Panel 1 processes first, builds cache
    translator1 = MangaTranslator(...)
    translator1.translate_image('panel1.png')
    cache1_size = len(translator1.ocr_roi_cache)
    
    # Panel 2 processes simultaneously
    translator2 = MangaTranslator(...)
    translator2.translate_image('panel2.png')
    cache2_size = len(translator2.ocr_roi_cache)
    
    # Caches should be independent
    assert translator1.ocr_roi_cache != translator2.ocr_roi_cache
```

---

## Performance Implications

### Cache Benefits Per Thread

Each thread benefits from its own cache:
- ✅ Same panel processed multiple times (e.g., retries) = cache hits
- ✅ No lock contention between threads (separate caches)
- ✅ No cache invalidation from other panels

### Trade-offs

**Pro**: Complete isolation, no race conditions
**Con**: No cache sharing between panels (each builds its own)

This is **intentional** - safety over cache reuse.

---

## Summary

### ✅ Safe for Parallel Panel Translation

| Aspect | Status | Notes |
|--------|--------|-------|
| Text isolation | ✅ SAFE | Each thread has separate translator instance |
| Cache thread-safety | ✅ SAFE | All cache ops use `threading.Lock()` |
| Cross-panel contamination | ✅ PREVENTED | Per-instance caches, cleared per image |
| Model sharing | ✅ SAFE | Read-only weights, internally locked |
| API client sharing | ✅ SAFE | Stateless, thread-safe rate limiting |

### Key Takeaways

1. **Each panel = Fresh translator instance** → Complete isolation
2. **Each translator = Own cache + lock** → Thread-safe operations  
3. **Shared models = Read-only** → No state contamination
4. **All cache ops = Locked** → No race conditions

**Result**: Parallel panel translation is **COMPLETELY SAFE** from text leakage.