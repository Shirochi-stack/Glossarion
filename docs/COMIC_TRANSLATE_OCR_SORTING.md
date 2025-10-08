# Comic-Translate OCR Sorting Implementation

## Overview

Implemented comic-translate's text region sorting algorithm to ensure proper manga reading order for all OCR providers, fixing translation mapping issues.

## Problem Statement

The original issue was that text regions weren't being sorted in the correct manga reading order (right-to-left, top-to-bottom), causing:
- Wrong translations mapped to wrong bubbles
- Overlapping text in rendered output
- Inconsistent reading flow

## Solution: Comic-Translate's Sorting Algorithm

### Key Components

#### 0. RT-DETR Region Sorting (BEFORE OCR)

**Most Important**: When bubble detection is enabled, RT-DETR text regions are sorted BEFORE OCR processing:

```python
# Get RT-DETR detections
rtdetr_detections = self._load_bubble_detector(ocr_settings, image_path)
all_regions = []
if 'text_bubbles' in rtdetr_detections:
    all_regions.extend(rtdetr_detections.get('text_bubbles', []))
if 'text_free' in rtdetr_detections:
    all_regions.extend(rtdetr_detections.get('text_free', []))

# SORT BEFORE OCR PROCESSING
all_regions = sorted(all_regions, key=lambda bbox: (
    bbox[1] + bbox[3] / 2,  # y_center (top to bottom)
    -(bbox[0] + bbox[2] / 2) if right_to_left else (bbox[0] + bbox[2] / 2)  # x_center
))

# Then process each region with OCR in sorted order
for i, (x, y, w, h) in enumerate(all_regions):
    # OCR each region...
```

This ensures the OCR results are in correct reading order from the start!

#### 1. Enhanced TextRegion Class

Added properties to `TextRegion` dataclass:

```python
@property
def center(self) -> Tuple[float, float]:
    """Get the center point of the text region"""
    x, y, w, h = self.bounding_box
    return (x + w / 2, y + h / 2)

@property
def xyxy(self) -> Tuple[int, int, int, int]:
    """Convert bounding box to (x1, y1, x2, y2) format"""
    x, y, w, h = self.bounding_box
    return (x, y, x + w, y + h)
```

These properties enable the sorting algorithm to work with center points and boundaries.

#### 2. Reading Order Sorting Function

Implemented `sort_regions_by_reading_order()` based on comic-translate's `sort_blk_list()`:

```python
def sort_regions_by_reading_order(regions: List[TextRegion], right_to_left: bool = True) -> List[TextRegion]:
    """
    Sort text regions by manga reading order (right-to-left, top-to-bottom).
    
    Algorithm:
    1. Sort regions by Y coordinate (top to bottom)
    2. For regions on the same horizontal band:
       - If right_to_left (manga/Japanese): sort by X descending (right to left)
       - Otherwise: sort by X ascending (left to right)
    """
```

### How It Works

The algorithm uses a sophisticated insertion sort:

1. **Primary Sort**: By Y coordinate (top to bottom)
   - Regions are first ordered vertically

2. **Horizontal Band Detection**: 
   - When a region's Y center falls within another region's vertical bounds
   - These regions are considered on the same "reading line"

3. **Secondary Sort**: By X coordinate
   - **For Manga (right_to_left=True)**: Higher X comes first (right → left)
   - **For Western (right_to_left=False)**: Lower X comes first (left → right)

### Implementation Details

#### Sorting Applied After Merging

The sorting is applied at the end of `detect_text_regions()`, after all processing:

```python
# Apply manga reading order sorting (comic-translate style)
if regions:
    # Determine reading direction based on source language
    source_lang = ocr_settings.get('language_hints', ['ja'])[0]
    right_to_left = source_lang in ['ja', 'ar', 'he']
    
    regions = sort_regions_by_reading_order(regions, right_to_left=right_to_left)
    self._log(f"📖 Sorted {len(regions)} regions by reading order")
```

This ensures:
- ✅ All OCR providers get consistent sorting
- ✅ Sorting happens after bubble detection and merging
- ✅ Language-aware direction (Japanese/Arabic/Hebrew vs Western)

## Comparison with Comic-Translate

### What We Adopted

✅ **Core Algorithm**: The insertion-based sorting logic  
✅ **Center-based Comparison**: Using region centers for positioning  
✅ **Vertical Band Detection**: Smart grouping of regions on same line  
✅ **Directional Support**: Right-to-left vs left-to-right reading

### Differences

🔧 **Data Structure**: We use `TextRegion` instead of comic-translate's `TextBlock`  
🔧 **Integration Point**: Applied after all provider-specific processing  
🔧 **Language Detection**: Simplified to check source language hints

## Benefits

### Correctness
- ✅ **Proper Reading Order**: Text regions sorted according to manga conventions
- ✅ **Accurate Mapping**: Translations correctly matched to original text
- ✅ **No Overlaps**: Prevents translation overlap from incorrect ordering

### Consistency
- ✅ **Universal Application**: Works with all OCR providers (RapidOCR, Azure, Google, etc.)
- ✅ **Language-Aware**: Automatically adjusts for right-to-left languages
- ✅ **Bubble Detection Compatible**: Works with or without RT-DETR detection

### Maintainability
- ✅ **Single Sort Point**: All sorting happens in one place
- ✅ **Well-Documented**: Clear algorithm explanation
- ✅ **Proven Logic**: Based on comic-translate's battle-tested code

## Usage Example

### For Japanese Manga (Right-to-Left)

```python
# Source language is Japanese
regions = detect_text_regions(image, language_hints=['ja'])

# Regions automatically sorted:
# Panel layout (numbers = reading order):
#     [3]  [2]  [1]   <- Top row (right to left)
#     [6]  [5]  [4]   <- Middle row (right to left)
#     [9]  [8]  [7]   <- Bottom row (right to left)
```

### For Western Comics (Left-to-Right)

```python
# Source language is English
regions = detect_text_regions(image, language_hints=['en'])

# Regions automatically sorted:
# Panel layout (numbers = reading order):
#     [1]  [2]  [3]   <- Top row (left to right)
#     [4]  [5]  [6]   <- Middle row (left to right)
#     [7]  [8]  [9]   <- Bottom row (left to right)
```

## Testing

### Verification Steps

1. **Visual Inspection**: Check rendered output for correct text placement
2. **Translation Order**: Verify translations match original text order
3. **No Overlaps**: Confirm no text overlapping in output
4. **Multi-Language**: Test with Japanese, English, Korean, Chinese

### Expected Results

- Text regions appear in natural reading order
- Translations correctly mapped to bubbles
- No visual artifacts from misaligned text
- Consistent behavior across all OCR providers

## Technical Notes

### Performance

- **Time Complexity**: O(n²) worst case, but typically much better
  - Most manga panels have 5-20 text regions
  - Insertion sort is efficient for small, partially-sorted data
  
- **Memory**: O(n) for the output list
  - No additional memory overhead

### Edge Cases Handled

✅ **Empty regions list**: Returns empty list immediately  
✅ **Single region**: Returns unchanged  
✅ **Overlapping regions**: Uses center points for stable sorting  
✅ **Vertical text**: Y-coordinate handles vertical arrangement  

## Future Enhancements

### Potential Improvements

1. **Adaptive Band Size**: Use median region height for band detection
2. **Column Detection**: Identify and sort vertical reading columns
3. **Page Layout Analysis**: Detect panel boundaries for better grouping
4. **Custom Reading Patterns**: Support non-standard layouts (e.g., 4-koma)

### Comic-Translate Features Not Yet Implemented

- **`group_items_into_lines`**: More sophisticated line grouping
- **Band ratio parameter**: Configurable vertical band tolerance
- **`sort_textblock_rectangles`**: Separate function for coordinate-text pairs

## References

### Comic-Translate Source Files

- `modules/utils/textblock.py`: Original `sort_blk_list()` function
- `pipeline/block_detection.py`: Usage in detection pipeline
- `modules/detection/base.py`: TextBlock creation and matching

### Related Documentation

- [COMIC_TRANSLATE_INTEGRATION.md](./COMIC_TRANSLATE_INTEGRATION.md): Text wrapping improvements
- See conversation history for full discussion of sorting issues

## Summary

Successfully implemented comic-translate's proven sorting algorithm to fix translation mapping issues. The solution has **TWO sorting points**:

### 1. RT-DETR Pre-Sort (When Bubble Detection Enabled)
- ✅ Sorts RT-DETR detected regions BEFORE OCR processing
- ✅ Ensures OCR processes regions in reading order from the start
- ✅ Applied to all OCR providers: RapidOCR, EasyOCR, PaddleOCR, DocTR, Custom API
- ✅ Uses simple center-based sorting (y_center, then x_center)

### 2. Final Sort (After All Processing)
- ✅ Uses sophisticated insertion-based algorithm from comic-translate
- ✅ Handles vertical band detection for regions on same reading line
- ✅ Works for non-RT-DETR providers (Azure, Google)
- ✅ Ensures consistent final output order

### Universal Features
- ✅ Handles both right-to-left (manga/Japanese) and left-to-right (Western) reading
- ✅ Language-aware based on source language settings
- ✅ Maintains compatibility with all existing features
- ✅ Two-layer approach ensures correctness at every stage

This ensures correct manga reading order and proper translation mapping! 🎉
