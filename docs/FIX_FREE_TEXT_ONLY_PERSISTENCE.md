# Fix: Free Text Only Background Opacity Persistence

## Problem
The "Free text only background opacity" toggle was not persisting its state. When enabled, it would revert to off after reopening the application or changing settings.

## Root Cause
The issue was caused by incomplete conversion from Tkinter to PySide6:

1. **Missing Value Update**: The checkbox `stateChanged` signal was connected to `_apply_rendering_settings()`, but that method never updated the `free_text_only_bg_opacity_value` variable when the checkbox changed.

2. **Tkinter Reference**: The code referenced a non-existent `free_text_only_bg_opacity_var` Tkinter variable at line 7311, which doesn't exist in PySide6.

3. **No Proper Handler**: There was no dedicated handler to update the value and trigger save when the checkbox state changed.

## Solution

### 1. Added Dedicated Handler Method
Created `_on_ft_only_bg_opacity_changed()` at line 4103 to:
- Read the checkbox state using PySide6's `isChecked()`
- Update `free_text_only_bg_opacity_value`
- Trigger `_save_rendering_settings()` to persist

### 2. Updated Checkbox Connection (Line 3011)
```python
# Before:
self.ft_only_checkbox.stateChanged.connect(self._apply_rendering_settings)

# After:
self.ft_only_checkbox.stateChanged.connect(lambda state: self._on_ft_only_bg_opacity_changed())
```

### 3. Fixed `_apply_rendering_settings()` (Line 7257-7263)
Changed from reading non-existent Tkinter variable to reading checkbox directly:
```python
# Before (Tkinter leftover):
if hasattr(self, 'free_text_only_bg_opacity_var'):
    self.main_gui.config['manga_free_text_only_bg_opacity'] = bool(self.free_text_only_bg_opacity_var.get())

# After (PySide6):
if hasattr(self, 'ft_only_checkbox'):
    ft_only_enabled = self.ft_only_checkbox.isChecked()
    self.translator.free_text_only_bg_opacity = bool(ft_only_enabled)
    self.free_text_only_bg_opacity_value = ft_only_enabled
```

### 4. Removed Dead Code (Line 7313-7314)
Removed the leftover Tkinter variable reference that was trying to save the setting.

## Changes Summary

### Files Modified
- `manga_integration.py`

### Lines Changed
1. **Line 3011**: Updated checkbox connection to new handler
2. **Line 4103-4108**: Added `_on_ft_only_bg_opacity_changed()` method
3. **Line 7257-7263**: Fixed `_apply_rendering_settings()` to use PySide6 checkbox
4. **Line 7313-7314**: Removed dead Tkinter code

## Testing
After this fix:
1. ✅ Toggle "Free text only background opacity" ON → it stays enabled
2. ✅ Close and reopen the manga integration dialog → setting is preserved
3. ✅ Setting is saved to config file correctly
4. ✅ Setting is applied to translator during translation

## Related
This fix completes the Tkinter to PySide6 conversion for this particular setting, ensuring proper state management using PySide6 patterns instead of Tkinter variable patterns.

---

**Date**: 2025-10-04  
**Issue**: Free text only BG opacity toggle doesn't persist  
**Status**: ✅ Fixed
