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


def generate_css_from_pdf(pdf_path: str) -> str:
    """
    Generate CSS styling based on PDF font and layout information.
    Returns CSS string with default justified alignment.
    """
    try:
        import fitz
        
        doc = fitz.open(pdf_path)
        
        # Collect font information from first few pages
        fonts = set()
        font_sizes = []
        
        # Sample first 5 pages or less
        sample_pages = min(5, len(doc))
        
        for page_num in range(sample_pages):
            page = doc[page_num]
            blocks = page.get_text("dict")["blocks"]
            
            for block in blocks:
                if block.get("type") == 0:  # Text block
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            font_name = span.get("font", "")
                            font_size = span.get("size", 0)
                            if font_name:
                                fonts.add(font_name)
                            if font_size > 0:
                                font_sizes.append(font_size)
        
        doc.close()
        
        # Determine base font size (most common)
        base_font_size = "11pt"
        if font_sizes:
            # Get median font size
            font_sizes.sort()
            median_size = font_sizes[len(font_sizes) // 2]
            base_font_size = f"{median_size:.1f}pt"
        
        # Generate CSS
        css = f"""/* CSS generated from PDF */

body {{
    font-family: 'Times New Roman', Times, serif;
    font-size: {base_font_size};
    line-height: 1.6;
    color: #000000;
    background-color: #ffffff;
    margin: 2em;
    text-align: justify;
}}

p {{
    margin: 0.5em 0;
    text-align: justify;
    text-justify: inter-word;
}}

h1 {{
    font-size: 2em;
    font-weight: bold;
    margin: 1em 0 0.5em 0;
    text-align: left;
}}

h2 {{
    font-size: 1.5em;
    font-weight: bold;
    margin: 0.8em 0 0.4em 0;
    text-align: left;
}}

h3 {{
    font-size: 1.2em;
    font-weight: bold;
    margin: 0.6em 0 0.3em 0;
    text-align: left;
}}

img {{
    max-width: 100%;
    height: auto;
    display: block;
    margin: 1em 0;
}}

.pdf-block {{
    margin: 0.5em 0;
}}
"""
        
        return css
        
    except Exception as e:
        print(f"⚠️ Failed to generate CSS from PDF: {e}")
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
}
"""


def extract_pdf_with_formatting(pdf_path: str, output_dir: str, extract_images: bool = True) -> Tuple[str, Dict[int, List[Dict]]]:
    """
    Extract PDF content with HTML formatting and images.
    
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
        
        total_pages = len(doc)
        print(f"📄 Processing {total_pages} pages with formatting...")
        
        for page_num in range(total_pages):
            if page_num % 10 == 0 and page_num > 0:
                print(f"    Processing page {page_num}/{total_pages}...", end='\r')
            
            page = doc[page_num]
            
            # Get structured text with block information
            blocks = page.get_text("dict")["blocks"]
            
            page_html = []
            current_para = []
            
            for block_idx, block in enumerate(blocks):
                if block.get("type") == 0:  # Text block
                    block_text = []
                    
                    for line in block.get("lines", []):
                        line_text = []
                        for span in line.get("spans", []):
                            text = span.get("text", "").strip()
                            if text:
                                # Check if this is a heading (larger font size)
                                font_size = span.get("size", 0)
                                flags = span.get("flags", 0)
                                
                                # Bold flag is bit 4 (16)
                                is_bold = (flags & 16) != 0
                                
                                # Simple heuristic: large + bold = heading
                                if font_size > 14 and is_bold:
                                    # Flush current paragraph
                                    if current_para:
                                        page_html.append(f'<p>{"".join(current_para)}</p>')
                                        current_para = []
                                    page_html.append(f'<h1>{text}</h1>')
                                elif font_size > 12 and is_bold:
                                    # Flush current paragraph
                                    if current_para:
                                        page_html.append(f'<p>{"".join(current_para)}</p>')
                                        current_para = []
                                    page_html.append(f'<h2>{text}</h2>')
                                else:
                                    line_text.append(text)
                        
                        if line_text:
                            block_text.append(" ".join(line_text))
                    
                    # Add block text to current paragraph
                    if block_text:
                        block_content = " ".join(block_text)
                        current_para.append(block_content + " ")
                
                elif block.get("type") == 1:  # Image block
                    # Flush current paragraph
                    if current_para:
                        page_html.append(f'<p>{"".join(current_para)}</p>')
                        current_para = []
                    
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
                page_html.append(f'<p>{"".join(current_para)}</p>')
            
            # Add page content
            if page_html:
                html_parts.append('\n'.join(page_html))
        
        doc.close()
        
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
        raise


def create_pdf_from_text(text, output_path):
    """
    Create a simple PDF from text using PyMuPDF (fitz).
    Returns True if successful, False otherwise.
    
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
