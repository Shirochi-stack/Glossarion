# Text Wrapping Upgrade - Change Summary

## What Changed

### 1. New Smart Hyphenation System
- Replaced basic word-wrapping with comic-translate's intelligent TextWrapper
- Now adds hyphens automatically when breaking long words
- Respects word structure, existing hyphens, and punctuation

### 2. Default Setting Changed ⚠️
**`manga_strict_text_wrapping` is now `True` by default**

**Before (old default):**
```json
"manga_strict_text_wrapping": false
```
- Long words would overflow or force tiny fonts
- No hyphenation on word breaks

**After (new default):**
```json
"manga_strict_text_wrapping": true
```
- Long words break cleanly with hyphens like `"extraordi-nary"`
- Better font sizes (often larger!)
- Better text fitting

### 3. How It Affects You

#### Auto-Resize Still Works
✅ Font auto-sizing works exactly the same way
✅ Now it works BETTER because of smarter wrapping
✅ You'll likely see larger, more readable fonts

#### Text Wrapping Behavior
**Old behavior (with default false):**
- Text: "The supercalifragilisticexpialidocious word"
- Result: Tiny font OR word overflow

**New behavior (with default true):**
- Text: "The supercalifragilisticexpialidocious word"
- Result: "The supercali-" + "fragilistic-" + "expialidocious" + "word"
- Larger font, clean breaks

## Migration Guide

### If you're happy with the new behavior:
✅ **Do nothing!** It's enabled by default now.

### If you want to disable hyphenation:
Add this to your config:
```json
{
  "manga_strict_text_wrapping": false
}
```

### If you had it explicitly set before:
Your setting will be preserved - no change needed.

## Quick Comparison

| Feature | Old Default (False) | New Default (True) |
|---------|---------------------|-------------------|
| Word breaking | ❌ No | ✅ Yes, with hyphens |
| Long word handling | Overflow or tiny font | Smart hyphenation |
| Font sizes | Often smaller | Often larger |
| Hyphen placement | None | Automatic & smart |
| Text fitting | Basic | Optimized |

## Files Modified

1. **`manga_translator.py`**
   - Line 397: Changed default from `False` to `True`
   - Lines 7949-8028: Refactored `_wrap_text()` method

2. **`hyphen_textwrap.py`** (NEW)
   - 508 lines of enhanced text wrapping logic
   - Based on comic-translate's implementation

3. **Documentation**
   - `TEXT_WRAPPING_UPGRADE.md` - Full technical documentation
   - `CHANGELOG_TEXT_WRAPPING.md` - This file

## Testing

All code compiles successfully:
- ✅ `hyphen_textwrap.py` - No errors
- ✅ `manga_translator.py` - No errors

## Need Help?

See `TEXT_WRAPPING_UPGRADE.md` for:
- Detailed explanation of how it works
- Configuration options
- Testing recommendations
- Known limitations
- Future enhancements

## Questions?

**Q: Do I need to enable something for auto-resize to work?**  
A: No! Auto-resize works automatically. The new default (`strict_text_wrapping = True`) just makes it work better.

**Q: Will my translations look different?**  
A: Yes, likely better! You'll see proper hyphens when words break, and often larger fonts.

**Q: Can I disable hyphenation?**  
A: Yes, set `"manga_strict_text_wrapping": false` in your config.

**Q: Will this break my existing translations?**  
A: No, but they may look better if you re-run them with the new system.

## Rollback

If you need to revert to old behavior:
```json
{
  "manga_strict_text_wrapping": false
}
```

However, the new system is recommended for better results!
