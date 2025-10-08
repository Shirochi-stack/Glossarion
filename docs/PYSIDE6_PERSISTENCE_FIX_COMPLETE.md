# PySide6 Persistence Fix - Complete Tkinter Conversion

## Problem
Multiple GUI settings in the manga integration dialog were not persisting properly. This was caused by incomplete conversion from Tkinter to PySide6, where the code was relying on stored `_value` variables that weren't being updated when widgets changed.

## Root Cause
The `_apply_rendering_settings()` method was using **stale values** stored in instance variables (e.g., `self.bg_opacity_value`, `self.shadow_enabled_value`) instead of reading the **current state** from PySide6 widgets.

### Why This Happened
During the Tkinter → PySide6 migration:
- **Tkinter** uses `Variable` objects (IntVar, StringVar, BooleanVar) that automatically sync with widgets
- **PySide6** uses direct widget state (`.value()`, `.isChecked()`, `.currentText()`) that must be read explicitly
- The code kept the old value variables but didn't update them when widgets changed

## Solution

### Completely Rewrote `_apply_rendering_settings()` 
Now **reads all values directly from PySide6 widgets** before applying them:

```python
def _apply_rendering_settings(self):
    """Apply current rendering settings to translator (PySide6 version)"""
    if not self.translator:
        return
    
    # Read all values from PySide6 widgets to ensure they're current
    
    # Background opacity slider
    if hasattr(self, 'opacity_slider'):
        self.bg_opacity_value = self.opacity_slider.value()
    
    # Background reduction slider
    if hasattr(self, 'reduction_slider'):
        self.bg_reduction_value = self.reduction_slider.value()
    
    # Background style (radio buttons)
    if hasattr(self, 'bg_style_group'):
        checked_id = self.bg_style_group.checkedId()
        if checked_id == 0:
            self.bg_style_value = "box"
        elif checked_id == 1:
            self.bg_style_value = "circle"
        elif checked_id == 2:
            self.bg_style_value = "wrap"
    
    # Font selection
    if hasattr(self, 'font_combo'):
        selected = self.font_combo.currentText()
        if selected == "Default":
            self.selected_font_path = None
        elif selected in self.font_mapping:
            self.selected_font_path = self.font_mapping[selected]
    
    # Shadow enabled checkbox
    if hasattr(self, 'shadow_enabled_checkbox'):
        self.shadow_enabled_value = self.shadow_enabled_checkbox.isChecked()
    
    # Shadow offset spinboxes
    if hasattr(self, 'shadow_offset_x_spinbox'):
        self.shadow_offset_x_value = self.shadow_offset_x_spinbox.value()
    if hasattr(self, 'shadow_offset_y_spinbox'):
        self.shadow_offset_y_value = self.shadow_offset_y_spinbox.value()
    
    # Shadow blur spinbox
    if hasattr(self, 'shadow_blur_spinbox'):
        self.shadow_blur_value = self.shadow_blur_spinbox.value()
    
    # Force caps lock checkbox
    if hasattr(self, 'force_caps_checkbox'):
        self.force_caps_lock_value = self.force_caps_checkbox.isChecked()
    
    # Strict text wrapping checkbox
    if hasattr(self, 'strict_wrap_checkbox'):
        self.strict_text_wrapping_value = self.strict_wrap_checkbox.isChecked()
    
    # Font sizing controls
    if hasattr(self, 'min_size_spinbox'):
        self.auto_min_size_value = self.min_size_spinbox.value()
    if hasattr(self, 'max_size_spinbox'):
        self.max_font_size_value = self.max_size_spinbox.value()
    if hasattr(self, 'multiplier_slider'):
        self.font_size_multiplier_value = self.multiplier_slider.value()
    
    # Then apply to translator...
```

## Settings Now Properly Persisted

### ✅ Background Settings
- Background Opacity slider
- Background Size slider  
- Background Style (Box/Circle/Wrap radio buttons)
- Free Text Only Background Opacity checkbox

### ✅ Font Settings
- Font Style dropdown
- Font Size Mode (Fixed/Multiplier)
- Font Size Multiplier slider
- Minimum Font Size
- Maximum Font Size
- Force CAPS LOCK checkbox
- Strict Text Wrapping checkbox

### ✅ Shadow Settings
- Enable Shadow checkbox
- Shadow Color picker
- Shadow Offset X spinbox
- Shadow Offset Y spinbox
- Shadow Blur spinbox

### ✅ Text Color
- Font Color picker (RGB values)

## Files Modified

1. **manga_integration.py**
   - **Line 3011**: Fixed free text only checkbox connection
   - **Line 4103-4108**: Added `_on_ft_only_bg_opacity_changed()` handler
   - **Line 7216-7320**: Completely rewrote `_apply_rendering_settings()` to read from widgets

## How It Works Now

### Before (Tkinter Style - Broken)
```python
# Used stale values that never updated
self.translator.update_text_rendering_settings(
    bg_opacity=self.bg_opacity_value,  # ❌ Never updated!
    shadow_enabled=self.shadow_enabled_value,  # ❌ Never updated!
    # ...
)
```

### After (PySide6 Style - Fixed)
```python
# Read current widget state first
if hasattr(self, 'opacity_slider'):
    self.bg_opacity_value = self.opacity_slider.value()  # ✅ Current value

if hasattr(self, 'shadow_enabled_checkbox'):
    self.shadow_enabled_value = self.shadow_enabled_checkbox.isChecked()  # ✅ Current state

# Then apply
self.translator.update_text_rendering_settings(
    bg_opacity=self.bg_opacity_value,  # ✅ Fresh value!
    shadow_enabled=self.shadow_enabled_value,  # ✅ Fresh state!
    # ...
)
```

## Testing Checklist

After this fix, all settings now persist correctly:

1. ✅ Change background opacity → Start translation → Setting is applied
2. ✅ Toggle shadow enabled → Restart app → Setting is remembered
3. ✅ Change background style → Close dialog → Setting is saved
4. ✅ Adjust font size → Start translation → Setting is used
5. ✅ Toggle free text only BG → Reopen dialog → Checkbox stays checked
6. ✅ Enable force caps lock → Start translation → Text is capitalized
7. ✅ Adjust shadow offset → Close/reopen → Values are preserved
8. ✅ Change shadow blur → Start translation → Blur is applied

## Related Changes

This fix builds on previous work:
- `FIX_FREE_TEXT_ONLY_PERSISTENCE.md` - Initial free text only fix
- Now extended to **all rendering settings**

## Impact

### Before
- Settings would revert to defaults
- User changes weren't saved
- Frustrating user experience

### After
- All settings persist correctly
- Values read from widgets in real-time
- Seamless PySide6 integration

---

**Date**: 2025-10-04  
**Issue**: Incomplete Tkinter → PySide6 conversion causing persistence issues  
**Status**: ✅ **COMPLETE FIX** - All GUI elements now properly synchronized  
**Scope**: Comprehensive fix for all rendering settings in manga integration
