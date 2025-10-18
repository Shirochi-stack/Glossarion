# Manga Image Preview & Manual Editing Guide

## Overview
The Manga Image Preview section in Glossarion provides a powerful interface for viewing, manually editing, and translating manga images. It features dual viewers (source and translated output), manual editing tools, and an automated translation workflow.

## Interface Layout

### Main Components

1. **Title Bar with Toggle**
   - 📷 Image Preview & Editing title
   - "Enable Manual Editing" checkbox with spinning Halgakos icon
   - Manual editing is **disabled by default** (preview mode)

2. **Dual Viewer Tabs**
   - **📄 Source Tab**: Original manga images with editing tools
   - **✨ Translated Output Tab**: Shows translated results
   - Thumbnails panel (right side) - appears when multiple images loaded

3. **Tool Sections**
   - **Manual Tools**: Drawing and editing tools (hidden when manual editing disabled)
   - **Translation Workflow**: Automated translation pipeline buttons

4. **File Info & Download**
   - Current file display
   - Download Images button (consolidates translated results)

---

## Getting Started

### 1. Loading Images
- Images are loaded automatically when selected through the file browser
- Multiple images create a thumbnail panel for easy navigation
- Click thumbnails to switch between images

### 2. Enable Manual Editing
**Important**: Manual editing is disabled by default for safety.

To enable manual editing:
1. Check the "Enable Manual Editing" checkbox in the title bar
2. The checkbox will turn blue when enabled
3. Manual editing tools will become visible
4. Your preference is automatically saved

---

## Manual Editing Tools

### Navigation & View Controls (Always Visible)
- **✅ Pan Tool**: Default tool for navigating and selecting objects
- **⊟ Fit to View**: Resize image to fit window
- **🔍+ Zoom In**: Increase zoom level
- **🔍- Zoom Out**: Decrease zoom level
- **Mouse Wheel + Shift**: Zoom in/out at cursor position

### Drawing Tools (Manual Editing Enabled Only)

#### **◭ Box Drawing Tool**
- Click and drag to create text detection boxes
- Boxes appear as semi-transparent pink rectangles with red borders
- Selected boxes turn yellow with thicker borders
- Use for manually marking text regions that auto-detection missed

### Management Tools
- **🗑 Delete Selected**: Remove currently selected box (yellow highlight)
- **❌ Clear Boxes**: Remove all detection boxes
- **Box Counter**: Shows number of active detection boxes

---

## Translation Workflow

### Automated Translation Pipeline
The workflow buttons work in sequence:

1. **Detect Text**
   - Automatically finds text regions using AI detection
   - Creates green detection boxes around text
   - RT-DETR model identifies speech bubbles vs. free text

2. **Clean**
   - Removes detected text from the image
   - Uses inpainting to fill text areas with multiple iterations for better quality
   - Manual brush strokes guide the cleaning process
   - Boxes marked for inpainting exclusion are automatically skipped
   - Inpainting iterations can be configured for optimal results

3. **Recognize Text**
   - Performs OCR on detected text regions
   - Extracts original text for translation
   - Results stored for the translation step

4. **Translate**
   - Translates recognized text to target language
   - Renders translated text onto cleaned image
   - Result appears in "Translated Output" tab

---

## Working with Detection Boxes

### Automatic Detection
- Green boxes appear after "Detect Text"
- Different types: text bubbles (always processed) vs. free text (toggle-able)
- Box colors indicate different classifications

### Right-Click Context Menu
**Important**: After running "Recognize Text" or "Translate", you can right-click on any detection box to access various editing options:

1. **Edit OCR Results**
   - Right-click any detection box after text recognition
   - Select "📝 Edit OCR: "[preview text]"" from the context menu
   - Opens an editable dialog showing the recognized text
   - Make corrections to fix OCR errors
   - **Save button is required** to update the overlay with changes

2. **Edit Translation Results**
   - Right-click any detection box after translation
   - Select "🌍 Edit Translation: "[preview text]"" from the context menu
   - Opens a dialog with both original and translated text
   - Edit either the original text or the translation
   - **Save button is required** to update the overlay with changes
   - Text overlays update immediately after saving

3. **Clean This Text**
   - Right-click any detection box
   - Select "🧹 Clean this text" to apply targeted cleaning
   - Removes text content from the selected region only
   - Useful for spot-cleaning specific problematic areas

4. **OCR This Text**
   - Right-click any detection box
   - Select "👁 OCR this text" to perform text recognition on this box only
   - Processes just the selected region instead of all detected text
   - Helpful for re-processing individual boxes with poor recognition

5. **Translate This Text**
   - Right-click any detection box (after OCR)
   - Select "🌍 Translate this text" to translate just this box
   - Processes individual text regions independently
   - Allows selective translation of specific content

6. **Inpainting Exclusion**
   - Right-click any detection box
   - Select "🚫 Exclude from inpainting" to protect this region during cleaning
   - Excluded boxes will not be processed during the Clean step
   - Useful for preserving important visual elements or text you want to keep
   - Toggle option - can be enabled/disabled per box

### Manual Editing
When manual editing is enabled:

1. **Creating New Boxes**
   - Select Box Drawing tool (◭)
   - Click and drag to create rectangle
   - Minimum size required (10x10 pixels)

2. **Selecting Boxes**
   - Use Pan tool (✅) 
   - Click on any box to select it
   - Selected box turns yellow with thicker border

3. **Moving Boxes**
   - Selected boxes can be dragged to new positions
   - Hover cursor changes to indicate moveable state

4. **Deleting Boxes**
   - Select box first (click on it)
   - Click Delete Selected (🗑) button
   - Or use Clear Boxes (❌) to remove all

### Box Persistence & Auto-Save
- Manual boxes are automatically saved per image
- Position and editing state auto-saved continuously
- Restored when switching between images
- Survives application restarts
- **Auto-save replaces manual "save position" functionality**

### Incremental Editing Support
- All editing operations support incremental updates
- Changes are applied progressively without losing previous work
- OCR and translation results can be refined iteratively
- Context menu operations work on individual boxes for precise control

---


## Output and Download

### Viewing Results
- **Source Tab**: Original image with your edits and detection boxes
- **Translated Output Tab**: Final result with translated text rendered
- Switch tabs to compare before/after

### Downloading Translated Images
1. Complete translation workflow for desired images
2. Green "📥 Download Images" button becomes enabled
3. Click to consolidate all translated images into single folder
4. Creates "translated" folder in source directory
5. Handles filename conflicts automatically

---

## Keyboard Shortcuts and Mouse Controls

### Mouse Navigation
- **Left Click**: Select boxes/tools (Pan mode)
- **Right Click**: Open context menu on detection boxes with editing options:
  - Edit OCR/Translation results
  - Clean, OCR, or Translate individual boxes
  - Access incremental editing features
- **Click + Drag**: Create boxes (Box Draw mode) or draw strokes (Brush/Eraser modes)
- **Shift + Mouse Wheel**: Zoom in/out at cursor position
- **Mouse Wheel**: Scroll content when zoomed in

### Tool Selection
- Tools are selected by clicking their buttons
- Only one tool active at a time
- Pan tool is the default and safest option

### View Controls
- **Fit to View**: Quickly return to full image view
- **Zoom controls**: Fine-tune magnification level
- **Tab switching**: Compare source vs. translated output instantly

---

This guide covers the complete functionality of the Manga Image Preview section. Start with the automated workflow, then use manual editing tools to refine results as needed. The interface is designed to be safe by default while providing powerful editing capabilities when enabled.