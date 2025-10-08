# Hyphenation Fix Summary

## Problem
The text wrapping was hyphenating too aggressively, creating breaks like:
- `a-pple` (1 char before hyphen)
- `appl-e` (1 char after hyphen)

This looks bad and breaks readability.

---

## Solution Implemented

### Fix Applied to `manga_translator.py`

**Lines Modified:** 7994-8101

**Key Changes:**
1. **Minimum Character Requirements:**
   - Added `MIN_CHARS_BEFORE_HYPHEN = 2` (configurable)
   - Added `MIN_CHARS_AFTER_HYPHEN = 2` (configurable)

2. **Early Exit for Short Words:**
   ```python
   if len(word) < MIN_CHARS_BEFORE_HYPHEN + MIN_CHARS_AFTER_HYPHEN:
       return [word]  # Don't break words shorter than 4 chars
   ```

3. **Binary Search Starts from Minimum:**
   ```python
   low = MIN_CHARS_BEFORE_HYPHEN  # Was: low = 1
   chars_that_fit = MIN_CHARS_BEFORE_HYPHEN  # Was: chars_that_fit = 1
   ```

4. **Safety Checks Before Breaking:**
   - Ensures at least 2 chars remain after break
   - Merges orphaned single chars with previous line
   - Respects minimums when searching for natural break points

5. **Configurable via Config:**
   ```python
   self.min_chars_before_hyphen = main_gui.config.get('manga_min_chars_before_hyphen', 2)
   self.min_chars_after_hyphen = main_gui.config.get('manga_min_chars_after_hyphen', 2)
   ```

---

## Comparison with Comic-Translate

### Comic-Translate's Approach (`hyphen_textwrap.py`)

**Key Features:**

1. **Modified Python `textwrap` Module:**
   - Based on standard library but enhanced
   - Lines 230-235 handle hyphenation:
   ```python
   if chunk[:end]:
       cur_line.append(chunk[:end])
       # Adds hyphen when splitting UNLESS certain chars exist
       if self.hyphenate_broken_words and chunk[:end][-1] not in ['-','.',',']:
           cur_line.append('-')
   ```

2. **Smart Hyphenation Check:**
   - Only adds hyphen if the break doesn't end with `-`, `.`, or `,`
   - This prevents double hyphens: `pre--fix`
   - Respects existing punctuation

3. **Break Options:**
   ```python
   break_on_hyphens=False,        # Don't break AT existing hyphens
   break_long_words=False,        # Don't force-break words if possible
   hyphenate_broken_words=True    # But DO add hyphens when forced to break
   ```

4. **Integration with Font Sizing:**
   - `pil_word_wrap()` function (lines 45-100)
   - Tries wrapping first, THEN reduces font size
   - Uses cost function to find best fit:
   ```python
   cost = (wrapped_width - roi_width)**2 + (wrapped_height - roi_height)**2
   ```

5. **Binary Search for Font Size:**
   - `pyside_word_wrap()` function (lines 153-242)
   - Uses binary search to find largest font that fits
   - Much faster than linear search

---

## Differences Between Implementations

| Feature | Glossarion (Your Code) | Comic-Translate |
|---------|------------------------|-----------------|
| **Approach** | Custom binary search for char count | Modified Python textwrap module |
| **Hyphen Addition** | Always adds if breaking | Checks for existing punctuation |
| **Min Chars** | Configurable (default 2) | Not explicitly enforced (relies on textwrap) |
| **Word Breaking** | Force breaks when needed | Tries to avoid breaking (`break_long_words=False`) |
| **Font Sizing** | Linear search (decrements by 2) | Binary search |
| **Wrapping Method** | Character-based wrapping | Word-based wrapping with hyphenation |

---

## Recommendations

### Immediate Improvements (Already Implemented)
✅ **Minimum character requirement** - Prevents 1-char breaks  
✅ **Configurable settings** - Users can adjust if needed  
✅ **Safety checks** - Multiple layers to prevent bad breaks

### Future Enhancements (Optional)

1. **Smart Punctuation Check** (from Comic-Translate):
   ```python
   # Before adding hyphen, check if word already ends with punctuation
   if break_at < len(remaining) and remaining[break_at-1] not in ['-', '.', ',', '!', '?']:
       lines.append(remaining[:break_at] + '-')
   else:
       lines.append(remaining[:break_at])
   ```

2. **Binary Search for Font Size** (from Comic-Translate):
   Replace your linear search (line 7895):
   ```python
   # Current: for font_size in range(start_size, min_size - 1, -2):
   
   # Better (binary search):
   lo, hi = min_size, max_size
   while lo <= hi:
       mid = (lo + hi) // 2
       font = self._get_font(mid)
       lines = self._wrap_text(text, font, usable_width, draw)
       total_height = len(lines) * (mid * 1.2)
       
       if total_height <= usable_height:
           # Fits! Try larger
           best_size = mid
           best_lines = lines
           lo = mid + 1
       else:
           # Too big, try smaller
           hi = mid - 1
   
   return best_size, best_lines
   ```

3. **Use Python's textwrap as Base:**
   - Could import comic-translate's `hyphen_textwrap.py`
   - Modify `_wrap_text()` to use it:
   ```python
   from modules.rendering.hyphen_textwrap import wrap as hyphen_wrap
   
   def _wrap_text(self, text: str, font: ImageFont, max_width: int, draw: ImageDraw):
       # Calculate character width
       avg_char_width = draw.textbbox((0, 0), "M", font=font)[2]
       estimated_columns = max_width // avg_char_width
       
       # Use hyphen_wrap with smart settings
       lines = hyphen_wrap(
           text,
           width=estimated_columns,
           break_on_hyphens=False,
           break_long_words=True,  # Only when absolutely necessary
           hyphenate_broken_words=True  # Add hyphens when breaking
       )
       
       return lines
   ```

---

## Configuration Options

### For Users (Add to Settings UI)

```python
# In manga settings dialog:
manga_min_chars_before_hyphen: int = 2  # Minimum chars before hyphen
manga_min_chars_after_hyphen: int = 2   # Minimum chars after hyphen
manga_strict_text_wrapping: bool = False  # Force-break long words?
```

### Recommended Values

| Text Type | Before | After | Notes |
|-----------|--------|-------|-------|
| English | 2 | 2 | Standard (current default) |
| English (Conservative) | 3 | 3 | Better readability, may overflow more |
| CJK Languages | 1 | 1 | Characters are fuller, less needed |
| URLs/Technical | 5 | 5 | Avoid breaking technical terms |

---

## Examples

### Before Fix:
```
This is a supercalifragilisticexpialidocious word.
```
Could become:
```
This is a s-
upercalifr-
a-
gilisticexp-
ialidocious
word.
```
❌ **Bad:** 1-char breaks, hard to read

### After Fix (Default: 2/2):
```
This is a super-
califragilistic-
expialidocious
word.
```
✅ **Good:** Minimum 2 chars, readable

### After Fix (Conservative: 3/3):
```
This is a
supercalifragilistic-
expialidocious word.
```
✅ **Better:** May overflow but more natural

---

## Testing

### Test Case 1: Short Words
**Input:** `"cat dog bird"`  
**Expected:** No hyphenation (all words < 4 chars)

### Test Case 2: Long Word
**Input:** `"supercalifragilisticexpialidocious"`  
**Expected:** Breaks with min 2 chars before/after hyphen

### Test Case 3: Edge Case
**Input:** `"ab"`  
**Expected:** No break (< min_before + min_after)

### Test Case 4: Existing Hyphen
**Input:** `"mother-in-law"`  
**Expected:** May break at existing hyphens, doesn't add extra

---

## Performance Impact

- **Negligible:** Additional checks are O(1) or O(log n)
- **Binary search in break point:** Same complexity as before
- **Configuration lookup:** Once per translator init

---

## Rollback

If users complain about overflow, they can:

1. **Disable strict wrapping:**
   ```python
   manga_strict_text_wrapping = False
   ```

2. **Lower minimum requirements:**
   ```python
   manga_min_chars_before_hyphen = 1
   manga_min_chars_after_hyphen = 1
   ```

3. **Increase font size range:**
   Allow smaller minimum font sizes for better fit

---

## Summary

✅ **Fixed:** No more 1-character hyphenation  
✅ **Configurable:** Users can adjust per their needs  
✅ **Learned from comic-translate:** Best practices applied  
🔄 **Future:** Can adopt binary search for font sizing  
🔄 **Future:** Can use textwrap module as base

Your implementation is now more robust and user-friendly!
