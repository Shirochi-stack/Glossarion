# Comic-Translate Auto Resize Implementation

## Summary
Reimplemented the auto resize logic from comic-translate's `pil_word_wrap` algorithm to provide better text fitting with top-down font sizing and column width optimization.

## What Changed

### 1. New Method: `_pil_word_wrap`
**Location:** Lines 7678-7814

This is a direct port of comic-translate's `pil_word_wrap` algorithm with the following features:

**Algorithm Flow:**
1. **Top-down approach**: Start with `init_font_size` (GUI max) and decrease by 0.75 until text fits
2. **Height check**: If text is too tall, reduce font size and restore original unwrapped text
3. **Width check**: If text is too wide, try column wrapping optimization:
   - Search for optimal column width (number of characters per line)
   - Use `hyphen_textwrap.wrap` for smart word wrapping with hyphenation
   - Find column width where wrapped text fits within `roi_width`
4. **Brute force at minimum**: If font reaches `min_font_size`, perform cost minimization:
   - Try all possible column widths (1 to len(text))
   - Calculate cost: `(width - roi_width)² + (height - roi_height)²`
   - Select column width with minimum cost

**Key Parameters:**
- `init_font_size`: Starting font size (from GUI max_font_size setting)
- `min_font_size`: Minimum allowed font size (from GUI min_font_size setting)
- `roi_width`, `roi_height`: Available space in bubble/region
- Uses `hyphen_textwrap` for smart wrapping with hyphenation support

### 2. Updated Method: `_fit_text_to_region`
**Location:** Lines 7825-7890

Simplified to use `_pil_word_wrap` instead of custom binary search:

**Before:**
- Complex binary search with multiple heuristics
- Bubble size-based font limits
- Custom wrapping logic

**After:**
- Calls `_pil_word_wrap` with GUI min/max font settings
- Converts returned wrapped text to lines array
- Maintains font multiplier mode support
- Simpler and more maintainable

## GUI Settings Integration

The implementation respects your existing GUI settings:

1. **`min_readable_size`** (GUI min font size):
   - Sets the lower bound for font sizing
   - Triggers brute-force cost optimization when reached

2. **`max_font_size_limit`** (GUI max font size):
   - Sets the starting point for top-down search
   - Maximum font size that will be tried

3. **`font_size_mode`** & **`font_size_multiplier`**:
   - Still supported after pil_word_wrap completes
   - Multiplier applied to final font size if mode is 'multiplier'

4. **`strict_text_wrapping`**:
   - Handled by existing `_wrap_text` method (still used for display)
   - `pil_word_wrap` uses hyphen_textwrap for smart wrapping

## Benefits

### From Comic-Translate Algorithm:
1. **Top-down approach**: More intuitive - starts large and shrinks only if needed
2. **Column width optimization**: Finds best line length for wrapping
3. **Smart hyphenation**: Uses hyphen_textwrap for better word breaks
4. **Brute-force fallback**: Guarantees best fit at minimum font size
5. **Battle-tested**: This algorithm is proven in comic-translate

### Code Quality:
1. **Simpler logic**: Removed complex heuristics and guesswork
2. **Better maintainability**: One algorithm to understand and debug
3. **Direct port**: Easy to compare with comic-translate if issues arise

## Dependencies

- **hyphen_textwrap.py**: Already present in your codebase at `src/hyphen_textwrap.py`
- Uses same PIL/Pillow libraries you already have

## Testing Recommendations

1. Test with various text lengths (short, medium, long)
2. Test with different bubble sizes (small, medium, large)
3. Test with GUI min/max font settings at different values
4. Verify font multiplier mode still works correctly
5. Check strict_text_wrapping toggle behavior

## Notes

- The original `_wrap_text` method is still used for the final display rendering
- `pil_word_wrap` determines font size and performs initial wrapping
- Font size stepping of 0.75 matches comic-translate (faster than integer steps, smoother results)
- Cost function minimization ensures best visual fit when text is difficult to fit
