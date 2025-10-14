# Azure Document Intelligence + RT-DETR Integration

## Overview

This document explains how Azure Document Intelligence (successor to Azure AI Vision) integrates with RT-DETR bubble detection in the manga translation pipeline, and why the "full-image approach" provides superior OCR quality.

---

## Architecture

### Cloud API vs Local Models

**Azure Document Intelligence** is fundamentally different from local OCR models:

- ✅ **Cloud API Service** - No model loading required
- ✅ **Credential-based initialization** - Only API endpoint + key needed
- ✅ **Optimized for full documents** - Best performance on complete pages
- ✅ **Built-in layout analysis** - Understands document structure

This is correctly implemented in:
- `ocr_manager.py` (lines 1864-2035): `AzureDocumentIntelligenceProvider`
- `manga_translator.py` (lines 3857-3869): Initialization with credentials only

**No model loading happens** - the `load_model()` method only creates the API client.

---

## RT-DETR Integration Strategies

### ❌ **Old Approach: Per-Region OCR** (Suboptimal)

```
RT-DETR → Crop each bubble → Send each crop to Azure API separately
```

**Problems:**
1. **Context Loss**: Each crop loses surrounding document layout
2. **Tight Cropping**: No padding around text regions
3. **API Overhead**: N API calls for N bubbles (slow + expensive)
4. **Feature Underutilization**: Azure's layout analysis works best on full pages
5. **Character Truncation Risk**: Tight crops may cut off partial characters

**Example:**
```python
# Old approach (per-region)
for (x, y, w, h) in all_regions:
    cropped = image[y:y+h, x:x+w]  # No padding!
    result = azure_api.ocr(cropped)  # Separate API call
```

---

### ✅ **New Approach: Full-Image OCR + Matching** (Optimal)

```
RT-DETR → Azure OCR on full image → Match OCR results to RT-DETR blocks
```

**Benefits:**
1. **✅ Full Context**: Azure sees the complete page layout
2. **✅ Single API Call**: One call instead of N (faster + cheaper)
3. **✅ Better Layout Analysis**: Azure's reading order detection works properly
4. **✅ No Cropping Artifacts**: All text is in original position
5. **✅ Higher Accuracy**: Azure Document Intelligence performs best on complete documents

**Implementation** (inspired by comic-translate):
```python
# Step 1: Run Azure on FULL image
full_image_ocr = azure_api.ocr(complete_image)

# Step 2: Match OCR results to RT-DETR bubble regions
matched_blocks = match_ocr_to_rtdetr_blocks(
    full_image_ocr,      # All detected text lines
    rtdetr_regions,      # Bubble bounding boxes
    source_lang          # For reading order
)
```

---

## Implementation Details

### Full-Image Workflow

Located in `manga_translator.py` lines 4270-4343:

```python
elif self.ocr_provider == 'azure-document-intelligence':
    if ocr_settings.get('bubble_detection_enabled', False):
        # Get RT-DETR bubble regions
        rtdetr_detections = self._load_bubble_detector(ocr_settings, image_path)
        all_regions = rtdetr_detections.get('text_bubbles', []) + \
                      rtdetr_detections.get('text_free', [])
        
        # Step 1: OCR FULL IMAGE (not cropped regions!)
        full_image_ocr = self.ocr_manager.detect_text(
            image,  # Complete image
            'azure-document-intelligence',
            confidence=confidence_threshold
        )
        
        # Step 2: Match OCR lines to RT-DETR bubble boxes
        matched_blocks = match_ocr_to_rtdetr_blocks(
            full_image_ocr,
            all_regions,
            source_lang
        )
        
        # Step 3: Create results with RT-DETR bubble bounds
        for block in matched_blocks:
            result = OCRResult(
                text=block['text'],
                bbox=block['bbox'],  # RT-DETR bubble box
                bubble_bounds=block['bbox']  # For rendering
            )
            ocr_results.append(result)
```

### Matching Algorithm

The `match_ocr_to_rtdetr_blocks()` function:

1. **Spatial Matching**: Associates OCR text lines with RT-DETR bubbles based on bounding box overlap
2. **Text Aggregation**: Combines multiple OCR lines within same bubble
3. **Reading Order**: Respects manga reading direction (right-to-left for Japanese)
4. **Empty Bubble Handling**: Skips RT-DETR regions with no matched text

---

## Why This Works Better

### Context Preservation

**Without cropping:**
- Azure sees: "これは素晴らしい日です" (complete sentence in full page context)

**With tight cropping:**
- Azure sees: "素晴ら" (partial text, missing context, possible character truncation)

### Layout Analysis Features

Azure Document Intelligence provides:
- **Reading order detection** - Works on full document, not isolated crops
- **Line grouping** - Detects multi-line text blocks naturally
- **Language detection** - More accurate with full page context
- **Confidence scores** - Better calibrated for complete documents

### Performance

- **Old**: 10 bubbles = 10 API calls = ~5-10 seconds + 10× cost
- **New**: 10 bubbles = 1 API call = ~1-2 seconds + 1× cost

---

## Configuration

### Enable Bubble Detection + Azure

In GUI or config:
```json
{
  "ocr_provider": "azure-document-intelligence",
  "bubble_detection_enabled": true,
  "azure_document_intelligence_endpoint": "https://your-resource.cognitiveservices.azure.com/",
  "azure_document_intelligence_key": "your-api-key"
}
```

### Disable Bubble Detection

For full-page OCR without RT-DETR:
```json
{
  "bubble_detection_enabled": false
}
```

This sends the entire image to Azure Document Intelligence without region filtering.

---

## Comparison with Other Providers

### RapidOCR
- **Also uses full-image approach** with RT-DETR matching
- Local model, so per-region cropping wouldn't hurt as much
- But still benefits from full-page context

### manga-ocr / DocTR / EasyOCR
- **Still use per-region approach**
- These are local models optimized for cropped regions
- Padding could help, but not as critical as with Azure

### Azure Document Intelligence (Cloud) - `azure-document-intelligence`
- **NOW uses full-image approach** for best results
- Cloud API designed for document analysis
- Per-region cropping wastes its capabilities
- Lines 4270-4343 in manga_translator.py

### Azure Vision (Cloud) - `azure`
- **NOW uses full-image approach** for best results
- Older Azure Computer Vision API
- Also benefits from full-page context
- Lines 3136-3247 in manga_translator.py

---

## Troubleshooting

### Issue: Poor OCR quality with Azure + RT-DETR

**Check:**
1. ✅ Is `bubble_detection_enabled: true`?
2. ✅ Is full-image approach being used? (Look for "Step 1: Running Azure Document Intelligence on full image" in logs)
3. ✅ Are RT-DETR regions being detected? (Check for "Sorted X RT-DETR regions")
4. ✅ Are OCR lines being matched to blocks? (Check for "Matched text to X RT-DETR blocks")

**If seeing "Processing each region with Azure Document Intelligence":**
- ⚠️ Old per-region approach is active
- Update code to use full-image approach (see manga_translator.py lines 4270-4343)

### Issue: Empty results

**Possible causes:**
1. RT-DETR not detecting any text regions
2. Azure OCR not finding any text
3. Matching algorithm not associating OCR with RT-DETR boxes

**Debug:**
- Set `bubble_detection_enabled: false` to test Azure without RT-DETR
- Check Azure credentials are valid
- Verify image contains readable text

---

## Credits

This approach is inspired by [comic-translate](https://github.com/ogkalu2/comic-translate), which pioneered the "full-image OCR + RT-DETR matching" strategy for manga translation.

## References

- Azure Document Intelligence: https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/
- RT-DETR: https://github.com/lyuwenyu/RT-DETR
- comic-translate: https://github.com/ogkalu2/comic-translate
