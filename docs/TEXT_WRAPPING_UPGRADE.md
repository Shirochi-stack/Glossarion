# Text Wrapping and Hyphenation Upgrade

## Summary
Integrated comic-translate's enhanced TextWrapper class to provide intelligent text wrapping with proper hyphenation for manga translation. This replaces the previous greedy word-wrapping approach with a more sophisticated chunk-based algorithm.

## Changes Made

### 1. New Module: `hyphen_textwrap.py`
- **Location**: `Glossarion/src/hyphen_textwrap.py`
- **Source**: Adapted from comic-translate's modified Python textwrap module
- **Key Features**:
  - Chunk-based text splitting (breaks on hyphens, whitespace, punctuation)
  - Smart hyphenation when breaking long words across lines
  - Configurable hyphenation behavior (`hyphenate_broken_words` parameter)
  - Respects existing hyphens in compound words
  - Never breaks immediately before common punctuation (., , -)
  - Proper handling of word boundaries

### 2. Refactored `manga_translator.py::_wrap_text()`
- **Old Approach**: 
  - Greedy word-by-word wrapping
  - Only broke words when absolutely necessary
  - Manual hyphen insertion with strict character count rules
  - Required separate `_force_break_word()` method

- **New Approach**:
  - Uses TextWrapper with character-based column width
  - Iterates through column widths to find optimal wrapping
  - Measures actual pixel width using PIL font metrics
  - Automatically adds hyphens when breaking words
  - Cost-based optimization (prefers fewer lines with good width utilization)
  - Early exit when finding optimal solution

### 3. Removed Obsolete Code
- **Deleted**: `_force_break_word()` method (lines 8029-8097)
  - No longer needed as TextWrapper handles word breaking automatically
  - Previous implementation had overly strict character count requirements
  - Often produced awkward breaks like "a-pple" or single-letter hyphenation

## How It Works

### Text Wrapping Algorithm
1. **Quick Check**: If text fits on one line, return it immediately
2. **Estimate Columns**: Calculate starting column width based on font size and available pixel width
3. **Iterative Search**: Try different column widths around the estimate
4. **For Each Width**:
   - Wrap text using TextWrapper with that column count
   - Measure actual pixel width of each wrapped line
   - Check if all lines fit within max_width
   - Calculate cost = number_of_lines + width_underutilization_penalty
5. **Select Best**: Choose wrapping with lowest cost
6. **Early Exit**: Stop if found 3 or fewer lines with >85% width utilization

### Hyphenation Behavior
The TextWrapper intelligently adds hyphens when:
- A long word must be broken across lines
- The break point doesn't already end with punctuation (-, ., ,)
- The word chunk is substantial enough to warrant breaking

**Examples**:
- `"extraordinary"` → `"extraordi-"` + `"nary"`
- `"compound-word"` → `"compound-"` + `"word"` (keeps existing hyphen)
- `"Hello, world"` → `"Hello,"` + `"world"` (no hyphen after comma)

## Configuration

### Existing Config Parameters (Retained)
These parameters are still in the config for backward compatibility, though the new TextWrapper has its own built-in logic:

```python
# In config.json or GUI settings:
"manga_min_chars_before_hyphen": 2  # Minimum chars before hyphen
"manga_min_chars_after_hyphen": 2   # Minimum chars after hyphen
"manga_strict_text_wrapping": true  # Enable word breaking (DEFAULT: True as of v1.0)
```

**Note:** As of version 1.0, `manga_strict_text_wrapping` is **enabled by default** to provide the best text wrapping experience out of the box.

### TextWrapper Parameters
The `_wrap_text()` method now configures TextWrapper with:
- `width`: Character column count (varies during search)
- `break_long_words`: Controlled by `manga_strict_text_wrapping` setting
- `break_on_hyphens`: True (prefer breaking at existing hyphens)
- `hyphenate_broken_words`: Controlled by `manga_strict_text_wrapping`
- `drop_whitespace`: True (clean line starts/ends)
- `expand_tabs`: True (convert tabs to spaces)
- `replace_whitespace`: True (normalize whitespace)

## Benefits

### Improved Text Layout
- **Natural Hyphenation**: Hyphens appear where words are actually broken
- **Better Line Breaks**: Chunk-based splitting respects word structure
- **Optimal Width Usage**: Cost function balances line count and width utilization
- **Fewer Awkward Breaks**: Respects existing hyphens and punctuation

### Code Quality
- **Simpler**: Removed 69 lines of custom word-breaking logic
- **More Maintainable**: Uses well-tested TextWrapper implementation
- **Better Tested**: Based on Python's standard textwrap module
- **More Flexible**: Easy to adjust wrapping behavior via TextWrapper parameters

### Performance
- **Efficient Search**: Estimates starting column width to minimize iterations
- **Early Exit**: Stops when finding good-enough solution
- **Reasonable Range**: Only searches columns ± 10 to 30 from estimate
- **Cached Font Metrics**: PIL font measurements are fast

## Testing Recommendations

### Visual Inspection
1. Translate manga pages with varying bubble sizes
2. Check for proper hyphen placement in wrapped text
3. Verify no single-letter or awkward word breaks
4. Confirm text fits within bubbles when constrain_to_bubble is enabled

### Edge Cases to Test
- Very small bubbles (force short column widths)
- Very long words (test hyphenation)
- Text with existing hyphens (compound-words)
- Text with punctuation (ensure no hyphens before punctuation)
- Mixed languages (Japanese/English)
- Special characters and symbols

### Configuration Testing
- Test with `strict_text_wrapping` enabled/disabled
- Try different min/max font sizes
- Vary bubble constraint settings
- Test with different font families

## Known Limitations

1. **Character-Based Width**: TextWrapper works with character counts, not pixel widths
   - Mitigated by iterating multiple column widths and measuring actual pixel widths
   
2. **Fixed Search Range**: Currently searches ± 10-30 columns from estimate
   - Should handle most cases, but very unusual fonts might need adjustment
   
3. **No Language-Aware Breaking**: Doesn't understand language-specific hyphenation rules
   - Works well for English and similar languages
   - May need enhancement for other languages

4. **Cost Function Tuning**: Current cost function is `len(lines) + (1 - utilization) * 0.5`
   - May need adjustment based on user feedback
   - Can be easily modified in `_wrap_text()` method

## Future Enhancements

### Possible Improvements
1. **Language-Aware Hyphenation**: Integrate with pyphen or similar library
2. **Adaptive Search Range**: Adjust search range based on bubble size
3. **Cache Column Width**: Remember optimal column width for similar bubble sizes
4. **User Preference**: Add GUI option to prefer fewer/more lines
5. **Hyphenation Dictionary**: Allow custom hyphenation rules
6. **Width Prediction**: Machine learning model to predict optimal column width

### Integration with Other Features
- **Multi-line Alignment**: Ensure centered/justified alignment works with hyphens
- **Font Fallback**: Test hyphenation with different font families
- **RTL Languages**: Ensure proper handling of right-to-left text
- **Vertical Text**: Consider vertical manga layout requirements

## References

- **Source**: comic-translate's `hyphen_textwrap.py` module
- **Based on**: Python's standard library `textwrap.py`
- **Python Documentation**: https://docs.python.org/3/library/textwrap.html
- **comic-translate**: https://github.com/ogkalu2/comic-translate

## Version History

### v1.0 - Initial Integration (Current)
- Added `hyphen_textwrap.py` module
- Refactored `_wrap_text()` to use TextWrapper
- Removed `_force_break_word()` method
- **Changed default: `manga_strict_text_wrapping` now defaults to `True`**
- Maintained backward compatibility with config settings
- Successfully compiled and ready for testing

**Breaking Change:** If you had `strict_text_wrapping` disabled and want to keep it that way, you'll need to explicitly set it to `false` in your config.
