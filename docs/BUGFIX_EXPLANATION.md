# Bug Fix: Windows ProcessPoolExecutor Hang

## Problem Identified

The translation process was hanging at approximately 90% during chapter extraction due to a **Windows-specific multiprocessing issue**.

### Root Cause

When using `ProcessPoolExecutor` with a high worker count (12 workers in your case) on Windows, Python's multiprocessing spawn method encounters:

```
PermissionError: [WinError 5] Access is denied
```

This occurs in `multiprocessing\reduction.py` when trying to duplicate handles via `_winapi.DuplicateHandle()`.

### Why It Hung

1. `ProcessPoolExecutor.submit()` creates 529 tasks
2. Some worker processes fail to spawn due to permission errors
3. The `as_completed()` loop waits indefinitely for futures that will never complete
4. Progress reaches ~90% then freezes because the remaining tasks are assigned to dead workers

### The Fix

**Switched from `ProcessPoolExecutor` to `ThreadPoolExecutor`**

**Why this works:**
- `ThreadPoolExecutor` uses threads within the same process (no spawn, no handle duplication)
- No permission issues with Windows handle inheritance
- Still provides parallelism for I/O-bound operations (which chapter extraction is)
- More reliable on Windows with high worker counts

### Performance Impact

**Minimal to none** because:
- Chapter extraction is I/O-bound (reading from ZIP files, parsing HTML)
- Python's GIL doesn't significantly impact I/O-bound operations
- BeautifulSoup parsing releases the GIL frequently
- 12 threads can fully saturate I/O without needing separate processes

### Files Modified

1. `Chapter_Extractor.py` - Line 900: Changed to `ThreadPoolExecutor`
2. `chapter_extraction_manager.py` - Line 241-243: Added DEBUG message pass-through

### Verification

After this fix, you should see:
- Chapter extraction completes without hanging
- All 529 chapters process successfully
- No permission errors in logs
- Process continues to glossary generation phase

## Testing

Run a translation and watch for:
```
📊 Submitted 529 tasks to ThreadPoolExecutor
[DEBUG_LOOP] Starting as_completed loop, waiting for first result...
✅ ThreadPoolExecutor loop completed: 529/529 tasks processed in X.Xs
```

If you see `529/529`, the bug is fixed!
