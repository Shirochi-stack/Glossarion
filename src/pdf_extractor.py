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
