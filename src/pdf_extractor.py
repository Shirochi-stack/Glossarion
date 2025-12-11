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
from typing import Dict, List, Tuple, Optional

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
        
        # Determine most common alignment
        text_align = Counter(alignments).most_common(1)[0][0] if alignments else 'justify'
        
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
h1 {{
    font-family: {font_family};
    font-size: 2em;
    font-weight: bold;
    margin: 1em 0 0.5em 0;
    text-align: left;
}}

h2 {{
    font-family: {font_family};
    font-size: 1.5em;
    font-weight: bold;
    margin: 0.8em 0 0.4em 0;
    text-align: left;
}}

h3 {{
    font-family: {font_family};
    font-size: 1.2em;
    font-weight: bold;
    margin: 0.6em 0 0.3em 0;
    text-align: left;
}}
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
    color: {text_color};
    background-color: #ffffff;
    margin: 2em;
    text-align: {text_align};
}}

p {{
    margin: 0.5em 0;
    text-align: {text_align};
    text-justify: inter-word;
}}
{heading_styles}
img {{
    max-width: 100%;
    height: auto;
    display: block;
    margin: 1em 0;
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
    margin: 0.3em 0;
    padding-left: 1em;
}}

.toc-entry a {{
    text-decoration: none;
    color: {text_color};
}}

.toc-entry a:hover {{
    text-decoration: underline;
    color: #0066cc;
}}

.toc-level-1 {{ padding-left: 0; font-weight: bold; }}
.toc-level-2 {{ padding-left: 1em; }}
.toc-level-3 {{ padding-left: 2em; }}

/* Preserve alignment classes */
.align-left {{ text-align: left; }}
.align-center {{ text-align: center; }}
.align-right {{ text-align: right; }}
.align-justify {{ text-align: justify; }}
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
    # Check for TOC patterns
    toc_patterns = [
        r'\.\.+\s*\d+',  # Dots followed by page number
        r'\s+\d+\s*$',     # Ends with page number
        r'^Chapter\s+\d+',  # Starts with "Chapter N"
        r'^\d+\.\d+',      # Starts with section number like "1.1"
        r'\d+\s*$',        # Just a page number at end
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
    Detect alignment of a specific text block.
    """
    bbox = block.get("bbox", [])
    if len(bbox) < 4:
        return ""
    
    x0, y0, x1, y1 = bbox
    block_width = x1 - x0
    left_margin = x0
    right_margin = page_width - x1
    
    # Check alignment
    if abs(left_margin - right_margin) < 30:
        return "align-center"
    elif right_margin < 50 and left_margin > 100:
        return "align-right"
    elif block_width > page_width * 0.7:
        return "align-justify"
    elif left_margin < 100:
        return "align-left"
    
    return ""


def extract_pdf_with_formatting(pdf_path: str, output_dir: str, extract_images: bool = True) -> Tuple[str, Dict[int, List[Dict]]]:
    """
    Extract PDF content with HTML formatting and images.
    Preserves font types, alignment, and table of contents.
    
    Returns:
    - HTML content with proper structure and image placeholders
    - Dictionary of extracted images by page number
    """
    try:
        import fitz
        
        print(f"📄 Extracting PDF with formatting: {os.path.basename(pdf_path)}")
        
        # Extract images first if enabled
        images_by_page = {}
        if extract_images:
            images_by_page = extract_images_from_pdf(pdf_path, output_dir)
        
        doc = fitz.open(pdf_path)
        html_parts = []
        
        # Try to extract TOC from PDF outline
        toc_html = _extract_toc_from_outline(doc)
        if toc_html:
            html_parts.append(toc_html)
            print(f"📑 Extracted table of contents from PDF outline")
        
        total_pages = len(doc)
        print(f"📄 Processing {total_pages} pages with formatting...")
        
        # Track potential TOC pages
        toc_entries = []
        in_toc_section = False
        
        for page_num in range(total_pages):
            if page_num % 10 == 0 and page_num > 0:
                print(f"    Processing page {page_num}/{total_pages}...", end='\r')
            
            page = doc[page_num]
            page_width = page.rect.width
            
            # Get structured text with block information
            blocks = page.get_text("dict")["blocks"]
            
            page_html = []
            current_para = []
            current_para_styles = []
            
            for block_idx, block in enumerate(blocks):
                if block.get("type") == 0:  # Text block
                    block_text = []
                    block_styles = []
                    
                    # Detect block alignment
                    alignment_class = _detect_block_alignment(block, page_width)
                    
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
                    
                    if is_toc_entry and page_num < 10:  # TOCs usually in first 10 pages
                        # Start or continue TOC section
                        if not in_toc_section:
                            in_toc_section = True
                            if not toc_html:  # Only add if we didn't extract from outline
                                toc_entries.append('<div class="toc">')
                                toc_entries.append('<div class="toc-title">Table of Contents</div>')
                        
                        toc_entries.append(f'<div class="toc-entry">{block_content}</div>')
                        continue  # Skip adding to main content
                    elif in_toc_section and page_num < 10:
                        # Check if we're still in TOC
                        # If block has normal content, end TOC section
                        if len(block_content) > 50 and not is_toc_entry:
                            in_toc_section = False
                            if toc_entries:
                                toc_entries.append('</div>')
                    
                    # Determine if this is a heading
                    if max_font_size > 14 and has_bold:
                        # Flush current paragraph
                        if current_para:
                            para_class = current_para_styles[0] if current_para_styles else ""
                            para_tag = f'<p class="{para_class}">' if para_class else '<p>'
                            page_html.append(f'{para_tag}{"".join(current_para)}</p>')
                            current_para = []
                            current_para_styles = []
                        
                        # Create anchor for potential TOC linking
                        anchor_id = re.sub(r'[^a-zA-Z0-9]+', '-', block_content[:50].lower()).strip('-')
                        h_class = f' class="{alignment_class}"' if alignment_class else ''
                        page_html.append(f'<h1 id="{anchor_id}"{h_class}>{block_content}</h1>')
                    elif max_font_size > 12 and has_bold:
                        # Flush current paragraph
                        if current_para:
                            para_class = current_para_styles[0] if current_para_styles else ""
                            para_tag = f'<p class="{para_class}">' if para_class else '<p>'
                            page_html.append(f'{para_tag}{"".join(current_para)}</p>')
                            current_para = []
                            current_para_styles = []
                        
                        anchor_id = re.sub(r'[^a-zA-Z0-9]+', '-', block_content[:50].lower()).strip('-')
                        h_class = f' class="{alignment_class}"' if alignment_class else ''
                        page_html.append(f'<h2 id="{anchor_id}"{h_class}>{block_content}</h2>')
                    else:
                        # Regular paragraph content
                        current_para.append(block_content + " ")
                        if alignment_class:
                            current_para_styles.append(alignment_class)
                
                elif block.get("type") == 1:  # Image block
                    # Flush current paragraph
                    if current_para:
                        para_class = current_para_styles[0] if current_para_styles else ""
                        para_tag = f'<p class="{para_class}">' if para_class else '<p>'
                        page_html.append(f'{para_tag}{"".join(current_para)}</p>')
                        current_para = []
                        current_para_styles = []
                    
                    # Insert image tag if we have image info
                    if page_num in images_by_page:
                        # Try to match image by bbox or just use next available
                        for img_info in images_by_page[page_num]:
                            img_filename = img_info['filename']
                            img_width = img_info.get('width', 0)
                            img_height = img_info.get('height', 0)
                            
                            # Create img tag with relative path
                            img_tag = f'<img src="images/{img_filename}"'
                            if img_width and img_height:
                                img_tag += f' width="{img_width}" height="{img_height}"'
                            img_tag += ' alt="PDF Image" />'
                            
                            page_html.append(img_tag)
                            break  # Use first image for this block
            
            # Flush any remaining paragraph
            if current_para:
                para_class = current_para_styles[0] if current_para_styles else ""
                para_tag = f'<p class="{para_class}">' if para_class else '<p>'
                page_html.append(f'{para_tag}{"".join(current_para)}</p>')
            
            # Add page content
            if page_html:
                html_parts.append('\n'.join(page_html))
        
        doc.close()
        
        # Add collected TOC entries if any
        if toc_entries and not toc_html:
            if in_toc_section:
                toc_entries.append('</div>')
            html_parts.insert(0, '\n'.join(toc_entries))
            print(f"📑 Detected and preserved table of contents from content")
        
        # Combine all pages
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
            base_url = None
            if images_dir and os.path.exists(images_dir):
                base_url = f"file:///{os.path.abspath(os.path.dirname(images_dir)).replace(os.sep, '/')}/"
            elif output_path:
                base_url = f"file:///{os.path.abspath(os.path.dirname(output_path)).replace(os.sep, '/')}/"
            
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
            
        except ImportError:
            print("⚠️ WeasyPrint not available, trying alternative methods...")
            pass
        
        # Try pdfkit (wkhtmltopdf wrapper) as fallback
        try:
            import pdfkit
            
            print("📄 Using pdfkit for PDF generation...")
            
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
            
            print("📄 Using PyMuPDF fallback for PDF generation (limited HTML support)...")
            
            # Parse HTML to extract text while preserving some structure
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove script and style tags
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Extract text with basic formatting
            text_content = soup.get_text(separator='\n', strip=True)
            
            # Use the text-based PDF creator
            return create_pdf_from_text(text_content, output_path)
            
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
