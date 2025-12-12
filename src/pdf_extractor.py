"""
PDF extraction utility for Glossarion
Extracts text from PDF files and converts them to a format suitable for translation
"""

import os
import tempfile
import sys
import time
import concurrent.futures
import re
import base64
import subprocess
import shutil
import glob
from typing import Dict, List, Tuple, Optional
from pathlib import Path

def _extract_chunk(args):
    """
    Worker function to extract text from a range of pages.
    args: (pdf_path, start_page, end_page)
    """
    pdf_path, start_page, end_page = args
    text_parts = []
    
    try:
        import fitz
        doc = fitz.open(pdf_path)
        
        for i in range(start_page, end_page):
            try:
                page = doc[i]
                text = page.get_text()
                if text.strip():
                    text_parts.append((i, text))
            except Exception as e:
                text_parts.append((i, f"[Extraction Error: {e}]"))
                
        doc.close()
    except Exception as e:
        # Return empty list or error indicator
        pass
        
    return text_parts

def _externalize_data_uri_images(html: str, images_dir: str, page_index: int) -> str:
    """Replace data URI images with files under images_dir and return modified HTML."""
    os.makedirs(images_dir, exist_ok=True)
    # Match src="data:image/<type>;base64,<data>"
    # Support both double and single quotes around src attribute
    pattern = re.compile(r'(src\s*=\s*(?:\"|\')data:(image\/[^;]+);base64,([^\"\']+)(?:\"|\'))', re.IGNORECASE)

    def repl(m):
        mime = m.group(2)  # 'image/png', 'image/jpeg', etc.
        b64 = m.group(3)
        ext = 'png'
        if 'jpeg' in mime or 'jpg' in mime:
            ext = 'jpg'
        elif 'gif' in mime:
            ext = 'gif'
        elif 'webp' in mime:
            ext = 'webp'
        # Deterministic filename
        fname = f"p{page_index:04d}_{abs(hash(b64)) & 0xffffffff:x}.{ext}"
        out_path = os.path.join(images_dir, fname)
        try:
            with open(out_path, 'wb') as f:
                f.write(base64.b64decode(b64))
            return f'src="images/{fname}"'
        except Exception:
            return m.group(1)  # fallback to original data URI

    return pattern.sub(repl, html)

def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF file.
    Returns the extracted text as a string.
    
    Tries multiple PDF libraries in order of preference:
    1. PyMuPDF (fitz) - fastest and most accurate
    2. pypdf/PyPDF2 - pure Python fallback
    3. pdfplumber - another fallback option
    """
    
    print(f"📄 Analyzing PDF: {os.path.basename(pdf_path)}")
    
    # Try PyMuPDF first (best quality)
    try:
        import fitz  # PyMuPDF
        
        # Check page count first
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        doc.close()
        
        print(f"📄 Found {total_pages} pages. Starting extraction...")
        
        # Use parallel processing for larger files (> 50 pages)
        # This significantly speeds up extraction for large PDFs
        if total_pages > 50:
            # Determine optimal worker count
            max_workers = min(os.cpu_count() or 4, 8)
            
            # Divide pages into chunks
            chunk_size = (total_pages + max_workers - 1) // max_workers
            ranges = []
            for i in range(0, total_pages, chunk_size):
                end = min(i + chunk_size, total_pages)
                ranges.append((pdf_path, i, end))
            
            print(f"🚀 Using {len(ranges)} parallel workers for extraction...")
            
            all_parts = []
            completed_pages = 0
            
            with concurrent.futures.ProcessPoolExecutor(max_workers=len(ranges)) as executor:
                # Submit all tasks
                futures = [executor.submit(_extract_chunk, r) for r in ranges]
                
                # Process results as they complete
                for future in concurrent.futures.as_completed(futures):
                    try:
                        chunk_results = future.result()
                        all_parts.extend(chunk_results)
                        
                        # Update progress
                        # Note: chunk_results contains tuples of (page_index, text)
                        # We count how many pages were in this chunk range
                        # (approximate progress update based on chunks completed)
                        completed_pages += len(chunk_results)
                        
                        # Simple progress bar
                        percent = min(100, (completed_pages / total_pages) * 100)
                        bar_len = 30
                        filled = int(bar_len * percent / 100)
                        bar = '█' * filled + '░' * (bar_len - filled)
                        print(f"    Extraction: [{bar}] {percent:.1f}%", end='\r')
                        
                    except Exception as e:
                        print(f"    Worker failed: {e}")
            
            print(f"    Extraction complete! Processing results...          ")
            
            # Sort by page index to ensure correct order
            all_parts.sort(key=lambda x: x[0])
            return "\n\n".join([p[1] for p in all_parts])
            
        else:
            # Sequential extraction for small files
            doc = fitz.open(pdf_path)
            text_parts = []
            
            for page_num in range(total_pages):
                # Update progress every 10 pages
                if page_num % 10 == 0:
                    print(f"    Extracting page {page_num + 1}/{total_pages}...", end='\r')
                
                page = doc[page_num]
                text = page.get_text()
                if text.strip():
                    text_parts.append(text)
            
            doc.close()
            print(f"    Extraction complete! {total_pages} pages processed.      ")
            return "\n\n".join(text_parts)
    
    except ImportError:
        pass  # Try next method
    except Exception as e:
        print(f"⚠️ PyMuPDF extraction failed: {e}. Trying fallbacks...")
    
    # Try pypdf/PyPDF2
    
    # Try pypdf/PyPDF2
    try:
        try:
            from pypdf import PdfReader  # pypdf (newer)
        except ImportError:
            from PyPDF2 import PdfReader  # PyPDF2 (older)
        
        reader = PdfReader(pdf_path)
        text_parts = []
        
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if text.strip():
                text_parts.append(text)
        
        return "\n\n".join(text_parts)
    
    except ImportError:
        pass  # Try next method
    
    # Try pdfplumber
    try:
        import pdfplumber
        text_parts = []
        
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text and text.strip():
                    text_parts.append(text)
        
        return "\n\n".join(text_parts)
    
    except ImportError:
        pass  # No PDF library available
    
    # No PDF library available
    raise ImportError(
        "No PDF library available. Please install one of:\n"
        "  pip install PyMuPDF  (recommended)\n"
        "  pip install pypdf\n"
        "  pip install pdfplumber"
    )


def convert_pdf_to_temp_txt(pdf_path):
    """
    Convert a PDF file to a temporary .txt file.
    Returns the path to the temporary .txt file.
    """
    try:
        # Extract text from PDF
        text = extract_text_from_pdf(pdf_path)
        
        # Create a temporary .txt file
        # Use the PDF's basename + .txt extension
        pdf_basename = os.path.splitext(os.path.basename(pdf_path))[0]
        temp_dir = tempfile.gettempdir()
        txt_path = os.path.join(temp_dir, f"glossarion_pdf_{pdf_basename}.txt")
        
        # Write text to temp file
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(text)
        
        return txt_path
    
    except Exception as e:
        raise Exception(f"Failed to convert PDF to text: {str(e)}")


def is_pdf_library_available():
    """Check if any PDF extraction library is available"""
    try:
        import fitz
        return True
    except ImportError:
        pass
    
    try:
        from pypdf import PdfReader
        return True
    except ImportError:
        pass
    
    try:
        from PyPDF2 import PdfReader
        return True
    except ImportError:
        pass
    
    try:
        import pdfplumber
        return True
    except ImportError:
        pass
    
    return False


def extract_images_from_pdf(pdf_path: str, output_dir: str) -> Dict[int, List[Dict]]:
    """
    Extract all images from a PDF file.
    Returns a dictionary mapping page numbers to lists of image info dicts.
    
    Each image info dict contains:
    - 'index': Image index on the page
    - 'filename': Generated filename for the image
    - 'path': Full path to saved image file
    - 'bbox': Bounding box (x0, y0, x1, y1) on the page
    - 'width': Image width
    - 'height': Image height
    """
    try:
        import fitz
        
        # Create images directory
        images_dir = os.path.join(output_dir, 'images')
        os.makedirs(images_dir, exist_ok=True)
        
        doc = fitz.open(pdf_path)
        images_by_page = {}
        
        print(f"📷 Extracting images from PDF...")
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            image_list = page.get_images(full=True)
            
            if not image_list:
                continue
            
            page_images = []
            
            for img_index, img in enumerate(image_list):
                try:
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]
                    
                    # Generate filename
                    filename = f"page_{page_num + 1}_img_{img_index + 1}.{image_ext}"
                    filepath = os.path.join(images_dir, filename)
                    
                    # Save image
                    with open(filepath, "wb") as img_file:
                        img_file.write(image_bytes)
                    
                    # Get image position on page
                    image_rects = page.get_image_rects(xref)
                    bbox = image_rects[0] if image_rects else (0, 0, 0, 0)
                    
                    page_images.append({
                        'index': img_index,
                        'filename': filename,
                        'path': filepath,
                        'bbox': bbox,
                        'width': base_image.get('width', 0),
                        'height': base_image.get('height', 0)
                    })
                    
                except Exception as e:
                    print(f"  ⚠️ Failed to extract image {img_index} from page {page_num + 1}: {e}")
            
            if page_images:
                images_by_page[page_num] = page_images
        
        doc.close()
        
        total_images = sum(len(imgs) for imgs in images_by_page.values())
        print(f"✅ Extracted {total_images} images from {len(images_by_page)} pages")
        
        return images_by_page
        
    except ImportError:
        print("⚠️ PyMuPDF not available - cannot extract images")
        return {}
    except Exception as e:
        print(f"❌ Failed to extract images: {e}")
        return {}


def _map_font_to_web_safe(font_name: str) -> str:
    """
    Map PDF font names to web-safe CSS font stacks.
    """
    font_lower = font_name.lower()
    
    # Serif fonts
    if any(name in font_lower for name in ['times', 'garamond', 'georgia', 'palatino', 'baskerville']):
        return "'Times New Roman', Times, Georgia, serif"
    
    # Sans-serif fonts
    if any(name in font_lower for name in ['arial', 'helvetica', 'verdana', 'calibri', 'sans']):
        return "Arial, Helvetica, 'Segoe UI', sans-serif"
    
    # Monospace fonts
    if any(name in font_lower for name in ['courier', 'mono', 'consolas']):
        return "'Courier New', Courier, monospace"
    
    # Default to serif
    return "'Times New Roman', Times, serif"


def _detect_text_alignment(blocks: List[Dict], page_width: float) -> str:
    """
    Detect the most common text alignment in blocks.
    Returns 'left', 'center', 'right', or 'justify'.
    """
    alignments = []
    
    for block in blocks:
        if block.get("type") == 0:  # Text block
            bbox = block.get("bbox", [])
            if len(bbox) >= 4:
                x0, y0, x1, y1 = bbox
                block_width = x1 - x0
                
                # Calculate position relative to page
                left_margin = x0
                right_margin = page_width - x1
                
                # Check if centered (equal margins within tolerance)
                if abs(left_margin - right_margin) < 30:
                    alignments.append('center')
                # Check if right-aligned
                elif right_margin < 50 and left_margin > 100:
                    alignments.append('right')
                # Check if justified (full width)
                elif block_width > page_width * 0.7:
                    alignments.append('justify')
                # Default to left
                else:
                    alignments.append('left')
    
    # Return most common alignment
    if not alignments:
        return 'justify'
    
    from collections import Counter
    return Counter(alignments).most_common(1)[0][0]


def generate_css_from_pdf(pdf_path: str) -> str:
    """
    Generate CSS styling based on PDF font and layout information.
    Scans the PDF to extract actual font families, sizes, alignment, and styling.
    """
    try:
        import fitz
        from collections import Counter
        
        doc = fitz.open(pdf_path)
        
        # Collect detailed font information from first few pages
        fonts = []
        font_sizes = []
        heading_fonts = {}  # size -> font info
        alignments = []
        colors = []
        line_heights = []
        
        # Sample first 10 pages or less for better accuracy
        sample_pages = min(10, len(doc))
        
        print(f"🎨 Analyzing PDF styling from {sample_pages} pages...")
        
        for page_num in range(sample_pages):
            page = doc[page_num]
            page_width = page.rect.width
            blocks = page.get_text("dict")["blocks"]
            
            # Detect alignment from blocks
            page_alignment = _detect_text_alignment(blocks, page_width)
            alignments.append(page_alignment)
            
            for block in blocks:
                if block.get("type") == 0:  # Text block
                    bbox = block.get("bbox", [])
                    
                    for line in block.get("lines", []):
                        line_bbox = line.get("bbox", [])
                        
                        # Calculate line height if possible
                        if len(line_bbox) >= 4:
                            line_height = line_bbox[3] - line_bbox[1]
                            if line_height > 0:
                                line_heights.append(line_height)
                        
                        for span in line.get("spans", []):
                            font_name = span.get("font", "")
                            font_size = span.get("size", 0)
                            color = span.get("color", 0)
                            flags = span.get("flags", 0)
                            
                            if font_name:
                                fonts.append(font_name)
                            
                            if font_size > 0:
                                font_sizes.append(font_size)
                                
                                # Track potential heading fonts (larger sizes with bold)
                                is_bold = (flags & 16) != 0
                                if font_size > 12 and is_bold:
                                    if font_size not in heading_fonts or is_bold:
                                        heading_fonts[font_size] = {
                                            'font': font_name,
                                            'bold': is_bold,
                                            'size': font_size
                                        }
                            
                            if color:
                                colors.append(color)
        
        doc.close()
        
        # Analyze collected data
        # Most common font family
        most_common_font = Counter(fonts).most_common(1)[0][0] if fonts else "Times-Roman"
        font_family = _map_font_to_web_safe(most_common_font)
        
        # Determine base font size (median for body text)
        base_font_size = "11pt"
        if font_sizes:
            font_sizes.sort()
            # Filter out heading sizes (top 10%)
            body_font_sizes = font_sizes[:int(len(font_sizes) * 0.9)]
            if body_font_sizes:
                median_size = body_font_sizes[len(body_font_sizes) // 2]
                base_font_size = f"{median_size:.1f}pt"
        
        # Determine most common alignment, default to justify
        text_align = 'justify'  # Default to justify
        if alignments:
            detected_align = Counter(alignments).most_common(1)[0][0]
            # Only use detected alignment if it's clearly dominant
            # Otherwise stick with justify for body text
            if detected_align:
                text_align = detected_align
        
        # Calculate line height ratio
        line_height_ratio = 1.6
        if line_heights and font_sizes:
            avg_line_height = sum(line_heights) / len(line_heights)
            avg_font_size = sum(font_sizes) / len(font_sizes)
            if avg_font_size > 0:
                line_height_ratio = avg_line_height / avg_font_size
                line_height_ratio = max(1.2, min(2.0, line_height_ratio))  # Clamp
        
        # Determine text color (most common non-white color)
        text_color = "#000000"
        if colors:
            # Convert most common color to hex
            most_common_color = Counter(colors).most_common(1)[0][0]
            if most_common_color != 0xffffff:  # Ignore white
                text_color = f"#{most_common_color:06x}"
        
        # Generate heading styles based on detected heading fonts
        heading_styles = ""
        sorted_headings = sorted(heading_fonts.items(), key=lambda x: x[0], reverse=True)
        
        for idx, (size, info) in enumerate(sorted_headings[:3], 1):
            h_tag = f"h{idx}"
            h_font = _map_font_to_web_safe(info['font'])
            h_size = f"{size:.1f}pt"
            h_weight = "bold" if info['bold'] else "normal"
            
            heading_styles += f"""
{h_tag} {{
    font-family: {h_font};
    font-size: {h_size};
    font-weight: {h_weight};
    margin: 1em 0 0.5em 0;
    text-align: left;
}}
"""
        
        # If no headings detected, use relative sizing
        if not heading_styles:
            heading_styles = f"""
h1 {
    font-family: {font_family};
    font-size: 2em;
    font-weight: bold;
    margin: 1em 0 0.5em 0;
    text-align: center;
}

h2 {
    font-family: {font_family};
    font-size: 1.5em;
    font-weight: bold;
    margin: 0.8em 0 0.4em 0;
    text-align: center;
}

h3 {
    font-family: {font_family};
    font-size: 1.2em;
    font-weight: bold;
    margin: 0.6em 0 0.3em 0;
    text-align: center;
}
"""
        
        # Generate CSS with detected properties
        css = f"""/* CSS generated from PDF */
/* Detected font: {most_common_font} */
/* Base size: {base_font_size} */
/* Alignment: {text_align} */

body {{
    font-family: {font_family};
    font-size: {base_font_size};
    line-height: {line_height_ratio:.2f};
    color: #000000;  /* Force black text */
    background-color: #ffffff;
    margin: 2em;
    text-align: justify;
}}

p {{
    margin: 0.5em 0;
    text-align: justify;
    text-justify: inter-word;
    color: #000000;  /* Force black text for paragraphs */
}}
{heading_styles}
img {{
    max-width: 100%;
    height: auto;
    display: block;
    margin: 1em auto;  /* Center images */
}}

.pdf-block {{
    margin: 0.5em 0;
}}

/* Table of Contents styling */
.toc {{
    margin: 2em 0;
    padding: 1em;
    background-color: #f9f9f9;
    border: 1px solid #ddd;
}}

.toc-title {{
    font-size: 1.5em;
    font-weight: bold;
    margin-bottom: 0.5em;
}}

.toc-entry {{
    margin: 0.5em 0;
    padding: 0.2em 0;
    display: flex;
    justify-content: space-between;
}}

.toc-entry a {{
    text-decoration: underline;
    color: #0000EE;  /* Blue links in TOC (standard hyperlink blue) */
    flex-grow: 1;
}}

.toc-entry a:hover {{
    color: #551A8B;  /* Purple on hover (visited link color) */
}}

.toc-entry a:visited {{
    color: #551A8B;  /* Purple for visited links */
}}

.toc-page {{
    margin-left: 1em;
    color: #000000;  /* Black page numbers */
    font-weight: normal;
}}

.toc-level-1 {{ padding-left: 0; font-weight: bold; }}
.toc-level-2 {{ padding-left: 1em; }}
.toc-level-3 {{ padding-left: 2em; }}

/* Preserve alignment classes - use !important to override defaults */
.align-left, p.align-left {{
    text-align: left !important;
}}

.align-center, p.align-center, img.align-center {{
    text-align: center !important;
    display: block;
    margin-left: auto !important;
    margin-right: auto !important;
}}

.align-right, p.align-right {{
    text-align: right !important;
}}

.align-justify, p.align-justify {{
    text-align: justify !important;
    text-justify: inter-word;
}}

/* Page breaks for proper document flow */
.page-break {{
    page-break-before: always;
    break-before: page;
    clear: both;
    height: 0;
    margin: 0;
    padding: 0;
}}

/* Print-specific page breaks */
@media print {{
    .page-break {{
        page-break-before: always;
        break-before: page;
    }}
}}

/* Improve spacing for centered content */
.align-center, p.align-center {{
    text-align: center !important;
    margin: 1.5em 0;
}}

/* Centered headings */
h1.align-center, h2.align-center, h3.align-center {{
    text-align: center !important;
}}

/* Add space around headings */
h1, h2, h3 {{
    margin-top: 2em;
    margin-bottom: 1em;
}}

/* First heading on page needs less top margin */
h1:first-child, h2:first-child {{
    margin-top: 0.5em;
}}

/* Ensure centered elements don't get overridden */
.align-center * {{
    text-align: center;
}}
"""
        
        print(f"✅ Generated CSS with font: {most_common_font}, size: {base_font_size}, align: {text_align}")
        return css
        
    except Exception as e:
        print(f"⚠️ Failed to generate CSS from PDF: {e}")
        import traceback
        traceback.print_exc()
        # Return default CSS
        return """/* Default CSS */

body {
    font-family: 'Times New Roman', Times, serif;
    font-size: 11pt;
    line-height: 1.6;
    text-align: justify;
    margin: 2em;
}

p {
    margin: 0.5em 0;
    text-align: justify;
}

img {
    max-width: 100%;
    height: auto;
    display: block;
    margin: 1em 0;
}"""


def _detect_toc_patterns(text: str) -> bool:
    """
    Detect if text looks like a table of contents entry.
    Common patterns: page numbers, dots/leaders, chapter numbers.
    """
    text = text.strip()
    
    # Skip if too long (TOC entries are usually short)
    if len(text) > 200:
        return False
    
    # Skip if too short
    if len(text) < 3:
        return False
    
    # Check for TOC patterns
    toc_patterns = [
        r'\.\.+\s*\d+',                    # Dots followed by page number (e.g., "Chapter 1.....10")
        r'^.+\s+\d{1,3}$',                 # Text followed by space and 1-3 digit page number (e.g., "Introduction 4")
        r'^Chapter\s+\d+',                 # Starts with "Chapter N"
        r'^\d+\.\d+\s+',                   # Starts with section number like "1.1 "
        r'^[IVX]+\.\s+',                   # Roman numerals (I, II, III, etc.)
        r'^The\s+[A-Z][a-z]+.*\d{1,3}$',  # "The Machine 8" style
        r'^In\s+the\s+[A-Z][a-z]+.*\d{1,3}$',  # "In the Golden Age 19" style
    ]
    
    for pattern in toc_patterns:
        if re.search(pattern, text):
            return True
    return False


def _extract_toc_from_outline(doc) -> str:
    """
    Extract table of contents from PDF outline/bookmarks.
    Returns HTML representation of the TOC.
    """
    try:
        toc = doc.get_toc(simple=False)
        if not toc:
            return ""
        
        html = ['<div class="toc">\n<div class="toc-title">Table of Contents</div>']
        
        for level, title, page_num, *rest in toc:
            # Create anchor ID from title
            anchor_id = re.sub(r'[^a-zA-Z0-9]+', '-', title.lower()).strip('-')
            html.append(f'<div class="toc-entry toc-level-{level}">')
            html.append(f'<a href="#{anchor_id}">{title}</a> <span class="page-num">{page_num}</span>')
            html.append('</div>')
        
        html.append('</div>\n')
        return '\n'.join(html)
    except Exception as e:
        print(f"⚠️ Could not extract TOC from outline: {e}")
        return ""


def _detect_block_alignment(block: Dict, page_width: float) -> str:
    """
    Detect text alignment based on block position.
    Returns alignment class: 'align-center', 'align-right', 'align-justify', or 'align-left'.
    """
    bbox = block.get("bbox", [])
    if len(bbox) < 4:
        return "align-justify"  # Default if no bbox
    
    left = bbox[0]
    right = bbox[2]
    block_width = right - left
    
    # Calculate margins
    left_margin = left
    right_margin = page_width - right
    margin_diff = abs(left_margin - right_margin)
    
    # Calculate how centered the block is (percentage)
    # For a perfectly centered block, this approaches 0
    center_offset_ratio = margin_diff / page_width if page_width > 0 else 1.0
    
    # Detect alignment based on margins and position
    # More lenient centering detection: if margins are within 8% of page width, consider centered
    if center_offset_ratio < 0.08 or margin_diff < 50:
        return "align-center"
    elif right_margin < 50 and left_margin > 100:
        return "align-right"
    elif block_width > page_width * 0.7:
        return "align-justify"
    elif left_margin < 100:
        return "align-left"
    else:
        return "align-justify"


def _is_page_break_candidate(page_num: int, blocks: List[Dict], page_height: float) -> bool:
    """
    Detect if a page should be treated as a page break (title page, chapter start, etc.).
    """
    # First few pages often have special formatting
    if page_num < 3:
        return True
    
    # Check if page has very few text blocks (likely a title/separator page)
    text_blocks = [b for b in blocks if b.get("type") == 0]
    if len(text_blocks) <= 3:
        return True
    
    # Check if first block is centered and large (likely chapter title)
    if text_blocks:
        first_block = text_blocks[0]
        bbox = first_block.get("bbox", [])
        if len(bbox) >= 4:
            y_pos = bbox[1]  # Top position
            # If first text is in top 20% of page, might be chapter start
            if y_pos < page_height * 0.2:
                return True
    
    return False


def _extract_with_pdf2htmlex(pdf_path: str, output_dir: str, page_by_page: bool = False) -> Tuple[str, Dict[int, List[Dict]]]:
    """Use pdf2htmlEX to convert PDF to HTML with near-100% layout fidelity.
    Returns (html or list[(page_num, html)], images_by_page). Images are external files.
    """
    # Determine executable name
    exe = shutil.which('pdf2htmlEX') or shutil.which('pdf2htmlex')
    if not exe:
        raise FileNotFoundError('pdf2htmlEX executable not found')

    # Use a temporary build dir
    tmp_dir = os.path.join(output_dir, '_pdf2htmlex')
    os.makedirs(tmp_dir, exist_ok=True)

    # Build command with split pages and external assets
    cmd = [
        exe,
        '--split-pages', '1',           # one HTML per page
        '--embed-css', '0',             # write external CSS
        '--embed-image', '0',           # write external image files
        '--embed-font', '0',            # external fonts
        '--embed-javascript', '0',
        '--process-outline', '1',       # preserve links/outline
        '--dest-dir', tmp_dir,
        pdf_path
    ]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        msg = (e.stderr or b'').decode('utf-8', 'ignore')
        raise RuntimeError(f'pdf2htmlEX failed: {msg[:500]}')

    # Locate CSS and copy to output as styles.css
    css_files = glob.glob(os.path.join(tmp_dir, '*.css'))
    css_path = None
    if css_files:
        css_path = os.path.join(output_dir, 'styles.css')
        try:
            shutil.copy2(css_files[0], css_path)
        except Exception:
            css_path = None

    # Collect per-page HTML
    page_files = sorted(glob.glob(os.path.join(tmp_dir, 'page-*.html')))
    if not page_files:
        # Some versions output <basename>.page-1.html
        page_files = sorted(glob.glob(os.path.join(tmp_dir, '*page-*.html')))

    images_by_page: Dict[int, List[Dict]] = {}
    pages: List[Tuple[int, str]] = []

    # Prepare images dir
    images_dir = os.path.join(output_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)

    # Move/copy images from tmp to images_dir and rewrite src paths
    # pdf2htmlEX typically writes images next to HTML with names like fNN.png
    def _rewrite_and_copy_assets(html_text: str) -> str:
        # Find src="..." attributes that refer to a local file (not data:, not http)
        def repl(m):
            src = m.group(2)
            if src.startswith('data:') or '://' in src:
                return m.group(0)
            # Resolve source path in tmp_dir
            src_path = os.path.normpath(os.path.join(tmp_dir, src))
            if not os.path.exists(src_path):
                # try relative basename
                src_path = os.path.join(tmp_dir, os.path.basename(src))
                if not os.path.exists(src_path):
                    return m.group(0)
            # Copy to images_dir
            dst_name = os.path.basename(src_path)
            dst_path = os.path.join(images_dir, dst_name)
            try:
                if not os.path.exists(dst_path):
                    shutil.copy2(src_path, dst_path)
                return f'<img{m.group(1)}src="images/{dst_name}"'
            except Exception:
                return m.group(0)
        return re.sub(r'<img([^>]+)src="([^"]+)"', repl, html_text, flags=re.IGNORECASE)

    for pf in page_files:
        # Extract page number from filename
        m = re.search(r'page-(\d+)\.html$', pf)
        if not m:
            m = re.search(r'page-(\d+)\.html', pf)
        page_num = int(m.group(1)) if m else len(pages) + 1
        with open(pf, 'r', encoding='utf-8', errors='ignore') as f:
            html_text = f.read()
        # Point CSS to ../styles.css for word_count folder context
        if css_path:
            html_text = re.sub(r'<link[^>]+href="[^"]+\.css"[^>]*>', '<link rel="stylesheet" href="../styles.css">', html_text, flags=re.IGNORECASE)
        # Rewrite and copy images
        html_text = _rewrite_and_copy_assets(html_text)
        pages.append((page_num, html_text))

    pages.sort(key=lambda x: x[0])
    if page_by_page:
        return pages, images_by_page
    else:
        combined = '\n\n'.join(p[1] for p in pages)
        return combined, images_by_page


def _hex_color_from_int(c: int) -> str:
    try:
        r = (c >> 16) & 255
        g = (c >> 8) & 255
        b = c & 255
        return f"#{r:02X}{g:02X}{b:02X}"
    except Exception:
        return "#000000"


def _extract_absolute_html(pdf_path: str, output_dir: str, page_by_page: bool = False) -> Tuple[str, Dict[int, List[Dict]]]:
    """Build per-page HTML with absolutely-positioned text spans and images.
    Pure Python using PyMuPDF; preserves layout while keeping text selectable.
    """
    try:
        import fitz
    except Exception as e:
        raise

    images_by_page: Dict[int, List[Dict]] = {}
    # Reuse existing image extractor if present
    try:
        images_by_page = extract_images_from_pdf(pdf_path, output_dir)
    except Exception:
        images_by_page = {}

    doc = fitz.open(pdf_path)
    pages: List[Tuple[int, str]] = []

    for i in range(len(doc)):
        page = doc[i]
        pw, ph = page.rect.width, page.rect.height
        html_parts: List[str] = []
        html_parts.append('<!-- render:absolute -->')
        html_parts.append(f'<div class="pdf-page" id="page-{i+1}" style="position:relative;width:{pw}px;height:{ph}px;margin:0 auto;background:#ffffff;">')

        # Text as absolutely positioned spans
        pdata = page.get_text("dict")
        for block in pdata.get("blocks", []):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span.get("text", "")
                    if not text.strip():
                        continue
                    x0, y0, x1, y1 = span.get("bbox", [0, 0, 0, 0])
                    fs = span.get("size", 12)
                    font = span.get("font", "")
                    color = _hex_color_from_int(span.get("color", 0))
                    # Map font to a sane web-safe family if needed
                    fam = _map_font_to_web_safe(font) if ' _map_font_to_web_safe' in globals() else 'serif'
                    # Escape HTML special chars
                    esc = (text
                           .replace("&", "&amp;")
                           .replace("<", "&lt;")
                           .replace(">", "&gt;"))
                    style = f"position:absolute;left:{x0}px;top:{y0}px;font-size:{fs}px;color:{color};font-family:{fam};white-space:pre;"
                    html_parts.append(f'<span style="{style}">{esc}</span>')

        # Place images by bbox if we have them
        for img in images_by_page.get(i, []) + images_by_page.get(i+1, []):
            # Some callers key by 0-index, some by 1-index; include both
            bbox = img.get('bbox') or []
            if len(bbox) >= 4:
                x0, y0, x1, y1 = bbox[:4]
                w = max(1, int(x1 - x0))
                h = max(1, int(y1 - y0))
                fname = img.get('filename')
                if fname:
                    style = f'position:absolute;left:{x0}px;top:{y0}px;width:{w}px;height:{h}px;object-fit:contain;'
                    html_parts.append(f'<img src="images/{fname}" alt="img" style="{style}" />')

        # Make PDF links clickable with absolute overlays
        try:
            links = page.get_links()
        except Exception:
            links = []
        for ln in links or []:
            r = ln.get('from')
            if not r:
                continue
            x0, y0, x1, y1 = r[0], r[1], r[2], r[3]
            href = None
            if ln.get('page') is not None and ln.get('page') >= 0:
                href = f'#page-{int(ln["page"]) + 1}'
            elif ln.get('uri'):
                href = ln['uri']
            if not href:
                continue
            style = f'position:absolute;left:{x0}px;top:{y0}px;width:{max(0,x1-x0)}px;height:{max(0,y1-y0)}px;display:block;opacity:0;z-index:5;'
            html_parts.append(f'<a href="{href}" style="{style}"></a>')

        html_parts.append('</div>')
        pages.append((i+1, '\n'.join(html_parts)))

    doc.close()

    if page_by_page:
        return pages, images_by_page
    else:
        return '\n\n'.join(p[1] for p in pages), images_by_page


def extract_pdf_with_formatting(pdf_path: str, output_dir: str, extract_images: bool = True, page_by_page: bool = False) -> Tuple[str, Dict[int, List[Dict]]]:
    """
    Extract PDF content with HTML formatting and images.
    Preserves font types, alignment, and table of contents.
    
    Args:
        pdf_path: Path to PDF file
        output_dir: Output directory
        extract_images: Whether to extract images
        page_by_page: If True, returns list of (page_num, html_content) tuples instead of combined HTML
    
    Returns:
    - HTML content with proper structure and image placeholders (or list if page_by_page=True)
    - Dictionary of extracted images by page number
    """
    try:
        import fitz
        
        print(f"📄 Extracting PDF with formatting: {os.path.basename(pdf_path)}")
        
        # Extract images first if enabled (semantic path). In XHTML path we also export embedded images to files.
        images_by_page = {}
        if extract_images:
            images_by_page = extract_images_from_pdf(pdf_path, output_dir)
        images_dir = os.path.join(output_dir, 'images')
        
        doc = fitz.open(pdf_path)
        
        # If user requests external renderer pdf2htmlEX
        render_mode = os.getenv("PDF_RENDER_MODE", "absolute").lower()  # pdf2htmlex | absolute | xhtml | html | semantic
        if render_mode in ("pdf2htmlex", "pdf2htmlEX"):
            try:
                return _extract_with_pdf2htmlex(pdf_path, output_dir, page_by_page)
            except Exception as e:
                print(f"⚠️ pdf2htmlEX path failed, falling back to internal modes: {e}")
        
        # Pure-Python absolute layout (selectable text, fixed positions)
        if render_mode == "absolute":
            return _extract_absolute_html(pdf_path, output_dir, page_by_page)
        
        # If user requests exact page rendering, use MuPDF's built-in XHTML/HTML renderer for near 1:1 output
        if render_mode in ("xhtml", "html"):
            page_list = []
            total_pages = len(doc)
            print(f"📄 Rendering {total_pages} pages via MuPDF {render_mode.upper()} (near 1:1 layout)...")
            import re as _re
            for i in range(total_pages):
                page = doc[i]
                try:
                    page_html = page.get_text("xhtml" if render_mode == "xhtml" else "html")
                except Exception:
                    page_html = page.get_text("html")
                # Replace any data URI images with files under images/
                try:
                    page_html = _externalize_data_uri_images(page_html, images_dir, i + 1)
                except Exception:
                    pass
                # Inject page anchor and clickable overlays for native PDF links (preserve TOC entries)
                try:
                    anchor = f'<a id="page-{i + 1}" style="position:absolute;left:0;top:0;width:0;height:0;display:block;overflow:hidden"></a>'
                    links = page.get_links()
                    overlays = []
                    for ln in links or []:
                        r = ln.get('from')
                        if not r:
                            continue
                        try:
                            x0, y0, x1, y1 = float(r.x0), float(r.y0), float(r.x1), float(r.y1)
                        except Exception:
                            # r may be a tuple
                            x0, y0, x1, y1 = r[0], r[1], r[2], r[3]
                        href = None
                        if ln.get('page') is not None and ln.get('page') >= 0:
                            href = f'#page-{int(ln["page"]) + 1}'
                        elif ln.get('uri'):
                            href = ln['uri']
                        if not href:
                            continue
                        overlays.append(
                            f'<a class="pdf-link-overlay" href="{href}" style="position:absolute; left:{x0}px; top:{y0}px; width:{x1 - x0}px; height:{y1 - y0}px; display:block; opacity:0; z-index:9999; pointer-events:auto;"></a>'
                        )
                    overlays_html = ''.join(overlays)
                    m = _re.search(r'(<body[^>]*>)', page_html, flags=_re.IGNORECASE)
                    if m:
                        page_html = page_html.replace(m.group(1), m.group(1) + anchor + overlays_html, 1)
                    else:
                        page_html = anchor + overlays_html + page_html
                except Exception:
                    pass
                
                # Post-process: add centering styles to h1/h2 tags on title pages (first few pages)
                if i < 3:  # First 3 pages often have title/copyright/TOC
                    try:
                        from bs4 import BeautifulSoup
                        soup = BeautifulSoup(page_html, 'html.parser')
                        # Center all h1 and h2 tags on these pages
                        for tag in soup.find_all(['h1', 'h2']):
                            # Add both class and inline style for maximum compatibility
                            existing_style = tag.get('style', '')
                            if existing_style and not existing_style.endswith(';'):
                                existing_style += ';'
                            tag['style'] = existing_style + 'text-align:center!important;'
                            tag['class'] = tag.get('class', []) + ['align-center']
                        page_html = str(soup)
                    except Exception as e:
                        # If beautifulsoup fails, just use the page as-is
                        pass
                
                page_list.append((i + 1, page_html))
            doc.close()
            if page_by_page:
                return page_list, images_by_page
            else:
                combined = "\n\n".join(html for _, html in page_list)
                return combined, images_by_page
        
        html_parts = []
        
        # TOC handling mode: preserve (default) means do NOT synthesize a new TOC
        toc_mode = os.getenv("PDF_TOC_MODE", "preserve").lower()  # preserve | synthesize
        
        # Get TOC from PDF outline only if we plan to synthesize
        toc_from_outline = None
        toc_page_number = 0  # Page where TOC should be inserted (0-indexed)
        if toc_mode == "synthesize":
            try:
                toc_data = doc.get_toc(simple=False)
                if toc_data:
                    toc_page_number = 0
                    toc_from_outline = _extract_toc_from_outline(doc)
                    if toc_from_outline:
                        print(f"📑 Found TOC in PDF outline, will insert at beginning")
            except:
                pass
        
        has_outline_toc = bool(toc_from_outline)
        
        total_pages = len(doc)
        print(f"📄 Processing {total_pages} pages with formatting...")
        
        # Track TOC state
        in_toc_section = False
        toc_page_detected = False
        
        for page_num in range(total_pages):
            if page_num % 10 == 0 and page_num > 0:
                print(f"    Processing page {page_num}/{total_pages}...", end='\r')
            
            page = doc[page_num]
            page_width = page.rect.width
            page_height = page.rect.height
            
            # Get structured text with block information
            blocks = page.get_text("dict")["blocks"]
            
            # Check if this page should have a page break before it
            should_page_break = _is_page_break_candidate(page_num, blocks, page_height)
            
            page_html = []
            current_para = []
            current_para_styles = []
            
            # Map class to inline style to avoid reliance on external CSS
            _align_css = {
                "align-center": "text-align:center!important;",
                "align-right": "text-align:right!important;",
                "align-left": "text-align:left!important;",
                "align-justify": "text-align:justify!important;"
            }
            for block_idx, block in enumerate(blocks):
                if block.get("type") == 0:  # Text block
                    block_text = []
                    block_styles = []
                    
                    # Detect block alignment
                    alignment_class = _detect_block_alignment(block, page_width)
                    if not alignment_class:
                        alignment_class = "align-justify"  # Always have a default
                    
                    # Track fonts and styles in this block
                    block_fonts = []
                    max_font_size = 0
                    has_bold = False
                    
                    for line in block.get("lines", []):
                        line_text = []
                        for span in line.get("spans", []):
                            text = span.get("text", "").strip()
                            if text:
                                font_name = span.get("font", "")
                                font_size = span.get("size", 0)
                                flags = span.get("flags", 0)
                                color = span.get("color", 0)
                                
                                # Track styling
                                is_bold = (flags & 16) != 0
                                is_italic = (flags & 2) != 0
                                
                                if font_size > max_font_size:
                                    max_font_size = font_size
                                if is_bold:
                                    has_bold = True
                                if font_name:
                                    block_fonts.append(font_name)
                                
                                # Apply inline styles if different from default
                                styled_text = text
                                if is_bold and font_size <= 12:
                                    styled_text = f"<strong>{text}</strong>"
                                elif is_italic:
                                    styled_text = f"<em>{text}</em>"
                                
                                line_text.append(styled_text)
                        
                        if line_text:
                            block_text.append(" ".join(line_text))
                    
                    # Determine if this is a heading or TOC entry
                    block_content = " ".join(block_text)
                    
                    # Check for TOC patterns
                    is_toc_entry = _detect_toc_patterns(block_content)
                    
                    if toc_mode == "synthesize" and is_toc_entry and page_num < 10:
                        # Start or continue synthesized TOC section
                        if not in_toc_section:
                            in_toc_section = True
                            toc_page_detected = True
                            page_html.append('<div class="toc">')
                            page_html.append('<div class="toc-title">Table of Contents</div>')
                        
                        from bs4 import BeautifulSoup
                        clean_text = BeautifulSoup(block_content, 'html.parser').get_text()
                        match = re.match(r'^(.+?)\s+(\d{1,3})$', clean_text)
                        if match:
                            title = match.group(1).strip()
                            page_num_text = match.group(2)
                            anchor = re.sub(r'[^a-zA-Z0-9]+', '-', title.lower()).strip('-')
                            page_html.append(f'<div class="toc-entry"><a href="#{anchor}">{title}</a> <span class="toc-page">{page_num_text}</span></div>')
                        else:
                            page_html.append(f'<div class="toc-entry">{clean_text}</div>')
                        continue
                    elif toc_mode == "synthesize" and in_toc_section:
                        page_html.append('</div><!-- end toc -->')
                        in_toc_section = False
                    
                    # Determine if this is a heading
                    if max_font_size > 14 and has_bold:
                        # Flush current paragraph
                        if current_para:
                            para_class = current_para_styles[0] if current_para_styles else "align-justify"
                            para_style = _align_css.get(para_class, "")
                            para_tag = f'<p class="{para_class}" style="{para_style}">'
                            page_html.append(f'{para_tag}{"".join(current_para)}</p>')
                            current_para = []
                            current_para_styles = []
                        
                        # Create anchor for potential TOC linking
                        anchor_id = re.sub(r'[^a-zA-Z0-9]+', '-', block_content[:50].lower()).strip('-')
                        h_class = f' class="{alignment_class}"' if alignment_class else ''
                        h_style = f' style="{_align_css.get(alignment_class, "")}"' if alignment_class else ''
                        page_html.append(f'<h1 id="{anchor_id}"{h_class}{h_style}>{block_content}</h1>')
                    elif max_font_size > 12 and has_bold:
                        # Flush current paragraph
                        if current_para:
                            para_class = current_para_styles[0] if current_para_styles else "align-justify"
                            para_style = _align_css.get(para_class, "")
                            para_tag = f'<p class="{para_class}" style="{para_style}">'
                            page_html.append(f'{para_tag}{"".join(current_para)}</p>')
                            current_para = []
                            current_para_styles = []
                        
                        anchor_id = re.sub(r'[^a-zA-Z0-9]+', '-', block_content[:50].lower()).strip('-')
                        h_class = f' class="{alignment_class}"' if alignment_class else ''
                        h_style = f' style="{_align_css.get(alignment_class, "")}"' if alignment_class else ''
                        page_html.append(f'<h2 id="{anchor_id}"{h_class}{h_style}>{block_content}</h2>')
                    else:
                        # Regular paragraph content
                        # Detect paragraph boundaries based on sentences and formatting
                        should_start_new_para = False
                        
                        if current_para:
                            prev_alignment = current_para_styles[-1] if current_para_styles else "align-justify"
                            
                            # Check if previous block ended with sentence-ending punctuation
                            prev_text = current_para[-1] if current_para else ""
                            ends_with_sentence = bool(re.search(r'[.!?]\s*$', prev_text.strip()))
                            
                            # Check if current block looks like start of new paragraph
                            # Indicators: starts with capital, previous ended with period, indentation change
                            is_short_line = len(block_content.strip()) < 40
                            starts_with_capital = bool(re.match(r'^[A-Z]', block_content.strip()))
                            
                            # Start new paragraph if:
                            # 1. Alignment changed (different text type)
                            # 2. Previous block ended with sentence AND current is short (likely paragraph break)
                            # 3. Current block is very short and starts with capital (likely new paragraph)
                            if alignment_class != prev_alignment:
                                should_start_new_para = True
                            elif ends_with_sentence and is_short_line and starts_with_capital:
                                should_start_new_para = True
                            elif is_short_line and len(current_para) > 5 and starts_with_capital:
                                # Multiple blocks accumulated and this looks like paragraph break
                                should_start_new_para = True
                            
                            if should_start_new_para:
                                # Flush existing paragraph
                                para_class = "align-justify"  # Force justify for body text
                                para_style = _align_css.get(para_class, "")
                                para_tag = f'<p class="{para_class}" style="{para_style}">'
                                page_html.append(f'{para_tag}{" ".join(current_para)}</p>')
                                current_para = []
                                current_para_styles = []
                        
                        # Add to current paragraph - combine blocks with space
                        current_para.append(block_content.strip())
                        current_para_styles.append("align-justify")  # Force justify
                
                elif block.get("type") == 1:  # Image block
                    # Flush current paragraph before image
                    if current_para:
                        para_class = "align-justify"  # Force justify for body text
                        para_style = _align_css.get(para_class, "")
                        para_tag = f'<p class="{para_class}" style="{para_style}">'
                        page_html.append(f'{para_tag}{" ".join(current_para)}</p>')
                        current_para = []
                        current_para_styles = []
                    
                    # Insert image tag if we have image info
                    if page_num in images_by_page:
                        # Try to match image by bbox or just use next available
                        for img_info in images_by_page[page_num]:
                            img_filename = img_info['filename']
                            img_width = img_info.get('width', 0)
                            img_height = img_info.get('height', 0)
                            bbox = img_info.get('bbox', [])
                            
                            # Detect if image is centered
                            img_alignment = ""
                            if len(bbox) >= 4:
                                img_block = {"bbox": bbox}
                                img_alignment = _detect_block_alignment(img_block, page_width)
                            
                            # Create img tag with relative path and centering if detected
                            img_class = f' class="{img_alignment}"' if img_alignment else ''
                            img_style = ''
                            if img_alignment == 'align-center':
                                img_style = ' style="display:block;margin:1em auto;"'
                            elif img_alignment == 'align-right':
                                img_style = ' style="display:block;margin-left:auto;"'
                            img_tag = f'<img src="images/{img_filename}"{img_class}{img_style}'
                            if img_width and img_height:
                                img_tag += f' width="{img_width}" height="{img_height}"'
                            img_tag += ' alt="PDF Image" />'
                            
                            page_html.append(img_tag)
                            break  # Use first image for this block
            
            # DON'T flush paragraph at end of page - let it continue to next page
            # This allows paragraphs to flow naturally across page boundaries
            # Only flush if we're on the last page or if there's a clear paragraph break
            
            # Close TOC if page ends while still in TOC section
            if in_toc_section:
                page_html.append('</div><!-- end toc at page end -->')
                # Don't reset in_toc_section here in case TOC continues on next page
            
            # Add page content with page break if needed
            if page_html:
                # Insert TOC from outline BEFORE page content only if synthesizing
                if toc_mode == "synthesize" and toc_from_outline and page_num == toc_page_number:
                    html_parts.append(toc_from_outline)
                    toc_from_outline = None  # Only insert once
                
                page_content = '\n'.join(page_html)
                
                # Add page break div for title pages and chapter starts
                if should_page_break and page_num > 0:
                    html_parts.append('<div class="page-break"></div>')
                
                html_parts.append(page_content)
        
        doc.close()
        
        # Log TOC synthesis (only if enabled)
        if toc_mode == "synthesize" and toc_page_detected:
            print(f"📑 Detected and preserved table of contents in document")
        
        # Return pages separately or combined based on mode
        if page_by_page:
            # Return list of (page_num, html_content) tuples
            # Note: html_parts may have fewer items than total_pages if some pages were empty
            # But we want consecutive numbering starting from 1
            page_list = []
            for i, html_content in enumerate(html_parts):
                page_list.append((i + 1, html_content))  # Use 1-indexed page numbers
            print(f"✅ Extracted {len(page_list)} pages with HTML formatting (page-by-page mode)")
            return page_list, images_by_page
        else:
            # Combine all pages into single HTML
            full_html = '\n\n'.join(html_parts)
            print(f"✅ Extracted {total_pages} pages with HTML formatting      ")
            return full_html, images_by_page
        
    except ImportError:
        print("⚠️ PyMuPDF not available - falling back to plain text")
        text = extract_text_from_pdf(pdf_path)
        # Wrap in basic HTML
        html = '\n'.join(f'<p>{para}</p>' for para in text.split('\n\n') if para.strip())
        return html, {}
    except Exception as e:
        print(f"❌ Failed to extract PDF with formatting: {e}")
        import traceback
        traceback.print_exc()
        raise


def _embed_images_as_data_uris(html: str, base_path: Optional[str], images_dir: Optional[str]) -> str:
    """Embed <img src> files as data URIs (used only at final PDF step if enabled)."""
    def _read_file_bytes(p: str) -> Optional[bytes]:
        try:
            with open(p, 'rb') as f:
                return f.read()
        except Exception:
            return None
    def _to_data_uri(path: str) -> Optional[str]:
        ext = os.path.splitext(path)[1].lower()
        mime = 'image/png'
        if ext in ('.jpg', '.jpeg'):
            mime = 'image/jpeg'
        elif ext == '.gif':
            mime = 'image/gif'
        elif ext == '.webp':
            mime = 'image/webp'
        data = _read_file_bytes(path)
        if data is None:
            return None
        return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"

    # Replace src="images/..." or relative paths
    img_re = re.compile(r'<img([^>]+)src="([^"]+)"', re.IGNORECASE)
    def repl(m):
        attrs = m.group(1)
        src = m.group(2)
        
        # Skip if already a data URI
        if src.startswith('data:'):
            return m.group(0)
        
        # Resolve file path
        candidate_paths = []
        
        # If src is like "images/foo.png" and we have an images_dir
        if images_dir and not os.path.isabs(src):
            if src.startswith('images/'):
                # Extract filename after "images/"
                img_filename = src.split('images/', 1)[1]
                candidate_paths.append(os.path.join(images_dir, img_filename))
            else:
                # Try treating src as relative to images_dir
                candidate_paths.append(os.path.join(images_dir, src))
        
        # If src is relative and we have a base_path, try that
        if base_path and not os.path.isabs(src):
            candidate_paths.append(os.path.normpath(os.path.join(base_path, src)))
        
        # If src is already absolute
        if os.path.isabs(src):
            candidate_paths.append(src)
        
        # Try to find and embed the image
        data_uri = None
        for p in candidate_paths:
            if os.path.exists(p):
                data_uri = _to_data_uri(p)
                if data_uri:
                    break
        
        if data_uri:
            return f'<img{attrs}src="{data_uri}"'
        return m.group(0)  # Keep original if not found
    
    return img_re.sub(repl, html)


def create_pdf_from_html(html_content: str, output_path: str, css_path: Optional[str] = None, images_dir: Optional[str] = None) -> bool:
    """
    Create a PDF from HTML content with proper rendering of formatting and images.
    
    Args:
        html_content: HTML content to convert
        output_path: Path where PDF should be saved
        css_path: Optional path to CSS file to apply
        images_dir: Optional directory containing images referenced in HTML
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Try using weasyprint first (best HTML rendering)
        try:
            # On Windows, try to auto-detect MSYS2 DLL directory for WeasyPrint
            if sys.platform == 'win32' and 'WEASYPRINT_DLL_DIRECTORIES' not in os.environ:
                msys2_paths = [
                    r'C:\msys64\mingw64\bin',
                    r'C:\msys64\ucrt64\bin',
                    r'C:\msys32\mingw64\bin',
                    os.path.expandvars(r'%USERPROFILE%\msys64\mingw64\bin'),
                ]
                for msys_path in msys2_paths:
                    if os.path.exists(msys_path):
                        gobject_dll = os.path.join(msys_path, 'libgobject-2.0-0.dll')
                        if os.path.exists(gobject_dll):
                            os.environ['WEASYPRINT_DLL_DIRECTORIES'] = msys_path
                            print(f"   • Auto-detected MSYS2 DLLs: {msys_path}")
                            break
            
            from weasyprint import HTML, CSS
            from weasyprint.text.fonts import FontConfiguration
            
            print("📄 Using WeasyPrint for PDF generation...")
            
            # Create font configuration for proper font handling
            font_config = FontConfiguration()
            
            # Prepare HTML document
            # If HTML doesn't have full document structure, wrap it
            if not html_content.strip().lower().startswith('<!doctype') and '<html' not in html_content.lower():
                full_html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <style>
        body {{ font-family: serif; margin: 2em; line-height: 1.6; }}
        img {{ max-width: 100%; height: auto; }}
    </style>
</head>
<body>
{html_content}
</body>
</html>"""
            else:
                full_html = html_content
            
            # Set base URL for resolving relative paths (images, CSS)
            # WeasyPrint needs this to resolve relative paths like "images/foo.png"
            base_url = None
            if output_path:
                # Use the output directory as base URL - this allows "images/foo.png" to resolve correctly
                output_dir = os.path.abspath(os.path.dirname(output_path))
                base_url = f"file:///{output_dir.replace(os.sep, '/')}/"
                print(f"   • Using base URL for image resolution: {base_url}")
            
            # Optionally embed images as data URIs at the final step
            # This is generally not needed as WeasyPrint can handle file paths with proper base_url
            if os.getenv('PDF_EMBED_IMAGES', '0') == '1':
                print("   • Embedding images as data URIs...")
                # base_path for resolving relative image paths
                base_path = os.path.dirname(output_path) if output_path else None
                full_html = _embed_images_as_data_uris(full_html, base_path, images_dir)
            else:
                # Verify images are accessible
                if images_dir and os.path.exists(images_dir):
                    import re
                    img_refs = re.findall(r'src="images/([^"]+)"', full_html, re.IGNORECASE)
                    missing_images = []
                    for img_file in img_refs:
                        img_path = os.path.join(images_dir, img_file)
                        if not os.path.exists(img_path):
                            missing_images.append(img_file)
                    if missing_images:
                        print(f"   ⚠️ Warning: {len(missing_images)} referenced images not found in images directory")
                        if len(missing_images) <= 5:
                            for img in missing_images:
                                print(f"      - {img}")

            # Create HTML object
            html_doc = HTML(string=full_html, base_url=base_url)
            
            # Prepare CSS
            stylesheets = []
            if css_path and os.path.exists(css_path):
                with open(css_path, 'r', encoding='utf-8') as f:
                    css_content = f.read()
                stylesheets.append(CSS(string=css_content, font_config=font_config))
            
            # Generate PDF
            html_doc.write_pdf(output_path, stylesheets=stylesheets, font_config=font_config)
            
            print(f"✅ Successfully created PDF with WeasyPrint: {os.path.basename(output_path)}")
            return True
            
        except ImportError as e:
            print("⚠️ WeasyPrint not available, trying alternative methods...")
            print(f"   • ImportError: {e}")
            if sys.platform == 'win32':
                print("   • For Windows: Install MSYS2 and run: pacman -S mingw-w64-x86_64-pango")
                print("   • Then set: WEASYPRINT_DLL_DIRECTORIES=C:\\msys64\\mingw64\\bin")
            pass
        except Exception as e:
            error_msg = str(e)
            print(f"⚠️ WeasyPrint failed: {error_msg[:200]}")
            
            # Provide specific guidance for DLL errors on Windows
            if sys.platform == 'win32' and ('cannot load library' in error_msg or 'DLL' in error_msg):
                print("\n⚠️ WeasyPrint DLL Error - Follow these steps:")
                print("   1. Install MSYS2 from https://www.msys2.org/")
                print("   2. Open MSYS2 terminal and run: pacman -S mingw-w64-x86_64-pango")
                print("   3. Set environment variable: WEASYPRINT_DLL_DIRECTORIES=C:\\msys64\\mingw64\\bin")
                print("   4. Or manually set it before running:")
                print("      $env:WEASYPRINT_DLL_DIRECTORIES='C:\\msys64\\mingw64\\bin'")
                print("\n   • Falling back to alternative PDF generation methods...\n")
            pass
        
        # Try xhtml2pdf (pure Python, bundleable) as fallback
        try:
            from xhtml2pdf import pisa
            
            print("📄 Using xhtml2pdf for PDF generation (pure Python, bundleable)...")
            
            # Prepare HTML with embedded CSS
            full_html = html_content
            if css_path and os.path.exists(css_path):
                try:
                    with open(css_path, 'r', encoding='utf-8') as f:
                        css_content = f.read()
                    
                    # Remove CSS features that xhtml2pdf doesn't support well
                    # Remove CSS comments that might cause parsing issues
                    import re as css_re
                    css_content = css_re.sub(r'/\*.*?\*/', '', css_content, flags=css_re.DOTALL)
                    
                    # Embed CSS into HTML
                    if '<head>' in full_html:
                        full_html = full_html.replace('<head>', f'<head><style>{css_content}</style>', 1)
                    else:
                        full_html = f'<html><head><style>{css_content}</style></head><body>{full_html}</body></html>'
                    print("   \u2022 Embedded CSS styling")
                except Exception as css_err:
                    print(f"   \u26a0\ufe0f Could not embed CSS: {css_err}")
            
            # Embed images as data URIs for xhtml2pdf (it doesn't handle file paths well)
            base_path = os.path.dirname(output_path) if output_path else None
            full_html = _embed_images_as_data_uris(full_html, base_path, images_dir)
            print("   • Embedded images as data URIs for xhtml2pdf")
            
            # Convert HTML to PDF
            with open(output_path, 'wb') as pdf_file:
                pisa_status = pisa.CreatePDF(
                    full_html.encode('utf-8'),
                    dest=pdf_file,
                    encoding='utf-8'
                )
            
            if not pisa_status.err:
                print(f"✅ Successfully created PDF with xhtml2pdf: {os.path.basename(output_path)}")
                if images_dir and os.path.exists(images_dir):
                    print(f"   • Images embedded in PDF")
                return True
            else:
                print(f"⚠️ xhtml2pdf reported errors during conversion")
                return False
            
        except ImportError:
            print("⚠️ xhtml2pdf not available, trying next method...")
            print("   • Install with: pip install xhtml2pdf")
            pass
        except Exception as e:
            print(f"⚠️ xhtml2pdf failed: {e}")
            pass
        
        # Try pdfkit (wkhtmltopdf wrapper) as fallback (not bundleable)
        try:
            import pdfkit
            
            print("📄 Using pdfkit for PDF generation (requires wkhtmltopdf)...")
            
            # Configure options
            options = {
                'encoding': 'UTF-8',
                'enable-local-file-access': '',
                'quiet': ''
            }
            
            # Add CSS if available
            if css_path and os.path.exists(css_path):
                options['user-style-sheet'] = css_path
            
            # Convert from string
            pdfkit.from_string(html_content, output_path, options=options)
            
            print(f"✅ Successfully created PDF with pdfkit: {os.path.basename(output_path)}")
            return True
            
        except (ImportError, Exception) as e:
            print(f"⚠️ pdfkit not available or failed: {e}")
            pass
        
        # Last resort: Use PyMuPDF to convert HTML (limited support)
        try:
            import fitz
            from bs4 import BeautifulSoup
            
            print("\n⚠️ WARNING: Using PyMuPDF fallback - this will create a TEXT-ONLY PDF!")
            print("   • All HTML formatting, CSS styles, and images will be LOST")
            print("   • The HTML file has been saved with full formatting and images")
            print("   • Recommended: Use the HTML file or install WeasyPrint for proper PDF")
            print("   • Creating basic text PDF...\n")
            
            # Parse HTML to extract text while preserving some structure
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove script and style tags
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Extract text with basic formatting
            text_content = soup.get_text(separator='\n', strip=True)
            
            # Use the text-based PDF creator
            result = create_pdf_from_text(text_content, output_path)
            
            if result:
                # Save a note file next to the PDF
                note_path = output_path.replace('.pdf', '_README.txt')
                try:
                    with open(note_path, 'w', encoding='utf-8') as f:
                        f.write("NOTE: This PDF was created with limited fallback support.\n")
                        f.write("It contains only plain text without formatting or images.\n\n")
                        f.write("For the full version with formatting and images, use the HTML file:\n")
                        f.write(f"  {output_path.replace('.pdf', '.html')}\n\n")
                        f.write("Or install WeasyPrint for proper PDF generation:\n")
                        f.write("  1. Install MSYS2 from https://www.msys2.org/\n")
                        f.write("  2. Run: pacman -S mingw-w64-x86_64-pango\n")
                        f.write("  3. Set: $env:WEASYPRINT_DLL_DIRECTORIES='C:\\msys64\\mingw64\\bin'\n")
                except:
                    pass
            
            return result
            
        except Exception as e:
            print(f"❌ PyMuPDF fallback failed: {e}")
            return False
        
    except Exception as e:
        print(f"❌ Error creating PDF from HTML: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_pdf_from_text(text, output_path):
    """
    Create a simple PDF from plain text using PyMuPDF (fitz).
    Returns True if successful, False otherwise.
    
    Note: This function only handles plain text. For HTML content, use create_pdf_from_html() instead.
    
    Also strips artificial "# Page X" markers that may have been inserted
    when concatenating multiple page texts.
    """
    try:
        import fitz
        import textwrap

        # Remove "# Page X" markers from the source text (one per line)
        cleaned_lines = []
        for line in text.splitlines():
            if re.match(r"^\s*#\s*Page\s+\d+\s*$", line):
                # Skip this marker line entirely
                continue
            cleaned_lines.append(line)
        text = "\n".join(cleaned_lines)

        doc = fitz.open()
        font_size = 11
        line_height = 14
        margin = 50
        
        # Default A4 size
        page_width, page_height = fitz.paper_size("a4")
        
        # Estimate chars per line for wrapping
        # A4 width 595 - 100 margin = 495 pts
        # font size 11 -> avg char width ~5-6 pts
        # 495 / 6 ~= 82 chars. Let's use 85 to be safe/dense.
        wrap_width = 85
        
        lines = []
        for paragraph in text.split('\n'):
            # Handle empty lines (paragraph breaks)
            if not paragraph.strip():
                lines.append("")
                continue
                
            # Wrap paragraph
            lines.extend(textwrap.wrap(paragraph, width=wrap_width))
        
        # Try to find a suitable TrueType font with Unicode support
        # This is needed for proper rendering of Turkish and other non-ASCII characters
        font_file = None
        
        # Common font paths on Windows, Linux, and Mac
        font_candidates = [
            # Windows
            "C:/Windows/Fonts/Arial.ttf",
            "C:/Windows/Fonts/times.ttf",
            "C:/Windows/Fonts/calibri.ttf",
            # Linux
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            # Mac
            "/System/Library/Fonts/Helvetica.ttc",
            "/Library/Fonts/Arial.ttf",
        ]
        
        for candidate in font_candidates:
            if os.path.exists(candidate):
                font_file = candidate
                break
        
        if not font_file:
            print("⚠️ No TrueType font found. PDF may not display non-ASCII characters correctly.")
        
        # Create a Font object with the TrueType file for proper Unicode support
        # Use fontname parameter instead to reference a TrueType font
        if font_file:
            font = fitz.Font(fontfile=font_file)
        else:
            font = None
        
        y = margin
        page = doc.new_page()
        
        # Use TextWriter for better Unicode support
        tw = fitz.TextWriter(page.rect)
        
        # Insert text line by line using TextWriter
        for line in lines:
            if y > page_height - margin:
                # Write accumulated text to current page
                tw.write_text(page)
                
                # Create new page and TextWriter
                page = doc.new_page()
                tw = fitz.TextWriter(page.rect)
                y = margin
            
            # Add text to TextWriter with proper Unicode support
            try:
                if font:
                    # Use custom TrueType font for proper Unicode rendering
                    tw.append((margin, y), line, font=font, fontsize=font_size)
                else:
                    # Fallback: try with default font
                    tw.append((margin, y), line, fontsize=font_size)
            except Exception:
                # Skip problematic lines if any
                pass
                
            y += line_height
        
        # Write remaining text to final page
        if page:
            tw.write_text(page)
            
        doc.save(output_path)
        doc.close()
        return True
        
    except Exception as e:
        print(f"Error creating PDF: {e}")
        return False
