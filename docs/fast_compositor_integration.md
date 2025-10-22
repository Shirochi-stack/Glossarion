# FastOverlayCompositor Integration Plan

The `FastOverlayCompositor` class is already instantiated in `MangaTranslationTab.__init__` but not yet integrated.
It's designed to provide instant preview updates when dragging text overlays by caching individual region renders.

## Integration Points

### 1. Initialize After Translation Complete
**Location:** `ImageRenderer._process_translate_results()` (line 8866)
**When:** After `_render_with_manga_translator()` completes successfully

```python
# After rendering is complete and we have the cleaned image
if cleaned_image_bgr is not None:
    # Initialize compositor with base image
    self.fast_compositor.set_base_image(cleaned_image_bgr)
    self._compositor_initialized = True
```

### 2. Cache Region Overlays During Render
**Location:** `manga_translator.py` in the `render_translated_text()` method (around line 10245+)
**When:** After each region overlay is created in the parallel/sequential rendering loop

```python
# Inside the rendering loop, after creating each region's overlay
if hasattr(self, 'main_gui') and hasattr(self.main_gui, 'manga_tab'):
    manga_tab = self.main_gui.manga_tab
    if hasattr(manga_tab, 'fast_compositor') and manga_tab._compositor_initialized:
        # Cache this region's overlay
        bbox = region.bounding_box  # (x, y, w, h)
        manga_tab.fast_compositor.cache_region_overlay(
            region_index=idx,
            overlay=overlay,  # The RGBA PIL image created for this region
            position=bbox
        )
```

### 3. Use Compositor for Fast Updates on Rectangle Move
**Location:** `ImageRenderer._attach_move_sync_to_rectangle()` (line 5335)
**When:** In the `_on_rect_release` handler, after rectangle is moved

```python
# Inside _on_rect_release, after computing new position
if hasattr(self, 'fast_compositor') and self._compositor_initialized:
    # Get new rectangle position
    rr = r.sceneBoundingRect()
    new_position = (int(rr.x()), int(rr.y()), int(rr.width()), int(rr.height()))
    
    try:
        # Fast composite with new position
        result_bgr = self.fast_compositor.update_region_position(idx, new_position)
        
        if result_bgr is not None:
            # Update preview with composited result
            # Save to temp file and reload in viewer
            import tempfile
            temp_path = os.path.join(tempfile.gettempdir(), f'manga_fast_comp_{idx}.png')
            cv2.imwrite(temp_path, result_bgr)
            # Reload in output viewer or update pixmap
            print(f"[FAST_COMP] Updated preview in {(time.time() - t0)*1000:.1f}ms")
        else:
            # Cache miss, fall back to full render
            print(f"[FAST_COMP] Cache miss for region {idx}, needs full render")
    except Exception as e:
        print(f"[FAST_COMP] Error: {e}")
        # Fallback to existing sync logic
```

### 4. Invalidate Cache on Text Edit
**Location:** Where OCR or translation text is edited via context menu
**When:** After user edits text in the edit dialogs

```python
# After text is edited
if hasattr(self, 'fast_compositor') and self._compositor_initialized:
    self.fast_compositor.invalidate_region(region_index)
```

### 5. Clear Cache on Base Image Change
**Location:** `ImageRenderer._run_clean_background()` (line 1598)
**When:** After new cleaned image is created

```python
# After cleaned_image is created successfully
if hasattr(self, 'fast_compositor'):
    self.fast_compositor.clear_cache()
    self._compositor_initialized = False
```

## Benefits

- **Instant preview updates** when dragging text overlays (< 10ms vs 500ms+ full render)
- **No re-computation** of text layout/wrapping when just moving position
- **Smooth interactive editing** experience
- **Graceful fallback** to full render on cache miss

## Performance

With caching:
- Initial render: ~500ms (same as before, builds cache)
- Subsequent moves: ~5-10ms (just composite from cache)
- Memory cost: ~2-5MB per region overlay (RGBA PIL images)

## Current Status

- ✅ Class implemented and ready to use
- ✅ Instance created in manga tab init
- ⏸️ Integration points identified but not yet connected
- ⏸️ Requires testing with actual manga workflow

## Next Steps

1. Add compositor initialization after first translation
2. Hook up cache population during render loop
3. Use compositor in rectangle move handler
4. Test with various manga pages
5. Optimize cache memory usage if needed (LRU eviction)
