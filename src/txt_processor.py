# txt_processor.py
import os
import re
import json
from typing import List, Tuple, Dict
from bs4 import BeautifulSoup
from chapter_splitter import ChapterSplitter
from decimal import Decimal
import hashlib
from pdf_extractor import extract_text_from_pdf, extract_pdf_with_formatting, generate_css_from_pdf
import shutil

class TextFileProcessor:
    """Process plain text files for translation"""
    
    def __init__(self, file_path: str, output_dir: str):
        self.file_path = file_path
        self.output_dir = output_dir
        self.file_base = os.path.splitext(os.path.basename(file_path))[0]
        
        # Initialize chapter splitter
        model_name = os.getenv("MODEL", "gpt-3.5-turbo")
        self.chapter_splitter = ChapterSplitter(model_name=model_name)
        
        # Check output format settings for PDFs
        self.pdf_output_format = os.getenv("PDF_OUTPUT_FORMAT", "html").lower()  # html, markdown, or txt
        self.pdf_extract_images = os.getenv("PDF_EXTRACT_IMAGES", "1") == "1"
        self.pdf_generate_css = os.getenv("PDF_GENERATE_CSS", "1") == "1"
        self.html2text_enabled = os.getenv("USE_HTML2TEXT", "0") == "1"
        
    def extract_chapters(self) -> List[Dict]:
        """Extract chapters from text file or PDF"""
        content = ""
        images_info = {}
        is_html_content = False
        
        if self.file_path.lower().endswith('.pdf'):
            try:
                # Determine output format
                if self.pdf_output_format in ['html', 'markdown']:
                    print(f"📄 Extracting PDF with formatting (format: {self.pdf_output_format})")
                    content, images_info = extract_pdf_with_formatting(
                        self.file_path, 
                        self.output_dir, 
                        extract_images=self.pdf_extract_images
                    )
                    is_html_content = True
                    
                    # Generate CSS if enabled
                    if self.pdf_generate_css and self.pdf_output_format == 'html':
                        css_content = generate_css_from_pdf(self.file_path)
                        css_path = os.path.join(self.output_dir, 'styles.css')
                        with open(css_path, 'w', encoding='utf-8') as f:
                            f.write(css_content)
                        print(f"✅ Generated styles.css")
                    
                    # Convert to markdown if html2text is enabled
                    if self.html2text_enabled or self.pdf_output_format == 'markdown':
                        content = self._html_to_markdown(content)
                        print(f"✅ Converted HTML to Markdown")
                else:
                    print(f"📄 Extracting text from PDF: {os.path.basename(self.file_path)}")
                    content = extract_text_from_pdf(self.file_path)
            except Exception as e:
                print(f"❌ Failed to extract text from PDF: {e}")
                content = "" # Handle empty content gracefully
        else:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        
        # Treat entire file as single document - no chapter detection
        raw_chapters = [{
            'num': 1,
            'title': self.file_base,
            'content': content,
            'is_html': is_html_content,
            'images_info': images_info
        }]
        
        # Process for splitting if needed
        final_chapters = self._process_chapters_for_splitting(raw_chapters)
        
        print(f"📚 Extracted {len(final_chapters)} total chunks")
        return final_chapters
    
    def _html_to_markdown(self, html_content: str) -> str:
        """Convert HTML to Markdown while preserving img tags"""
        try:
            # Try to use html2text if available
            import html2text
            
            h = html2text.HTML2Text()
            h.body_width = 0
            h.unicode_snob = True
            h.images_as_html = True  # Keep img tags as HTML
            h.images_to_alt = False
            h.protect_links = True
            
            markdown = h.handle(html_content)
            return markdown
        except ImportError:
            print("⚠️ html2text not available, keeping HTML format")
            return html_content
        except Exception as e:
            print(f"⚠️ Failed to convert HTML to markdown: {e}")
            return html_content
    
    def _process_chapters_for_splitting(self, raw_chapters: List[Dict]) -> List[Dict]:
        """Process chapters and split them if they exceed token limits"""
        final_chapters = []
        
        # Create word_count folder for storing original chunks
        word_count_dir = os.path.join(self.output_dir, 'word_count')
        os.makedirs(word_count_dir, exist_ok=True)
        
        # Calculate based on OUTPUT token limits
        max_output_tokens = int(os.getenv("MAX_OUTPUT_TOKENS", "8192"))
        compression_factor = float(os.getenv("COMPRESSION_FACTOR", "0.8"))
        safety_margin_output = 500
        
        # Calculate chunk size based on output limit
        available_tokens = int((max_output_tokens - safety_margin_output) / compression_factor)
        available_tokens = max(available_tokens, 1000)
        
        print(f"📊 Text file chunk size: {available_tokens:,} tokens (based on {max_output_tokens:,} output limit, compression: {compression_factor})")
        
        for chapter_data in raw_chapters:
            # Keep content in its format (may be HTML/markdown/plain text)
            chapter_content = chapter_data['content']
            is_html = chapter_data.get('is_html', False)
            images_info = chapter_data.get('images_info', {})
            chapter_tokens = self.chapter_splitter.count_tokens(chapter_content)
            
            if chapter_tokens > available_tokens:
                # Chapter needs splitting
                print(f"Chapter {chapter_data['num']} ({chapter_data['title']}) has {chapter_tokens} tokens, splitting...")
                
                # Pass filename for content type detection
                chunks = self.chapter_splitter.split_chapter(chapter_content, available_tokens, filename=self.file_path)
                
                # Add each chunk as a separate chapter
                for chunk_content, chunk_idx, total_chunks in chunks:
                    chunk_title = chapter_data['title']
                    if total_chunks > 1:
                        chunk_title = f"{chapter_data['title']} (Part {chunk_idx}/{total_chunks})"
                    
                    # Create float chapter numbers for chunks: 1.0, 1.1, 1.2, etc.
                    chunk_num = round(chapter_data['num'] + (chunk_idx - 1) * 0.1, 1)
                    
                    # Determine file extension based on format
                    if is_html and self.file_path.lower().endswith('.pdf'):
                        if self.html2text_enabled or self.pdf_output_format == 'markdown':
                            file_ext = '.md'
                        else:
                            file_ext = '.html'
                    else:
                        file_ext = '.txt'
                    
                    # Generate filename for this chunk
                    chunk_filename = f"section_{int(chapter_data['num'])}_{chunk_idx - 1}{file_ext}"
                    
                    # Save original chunk content to word_count folder (only if it doesn't exist)
                    original_chunk_path = os.path.join(word_count_dir, chunk_filename)
                    if not os.path.exists(original_chunk_path):
                        with open(original_chunk_path, 'w', encoding='utf-8') as f:
                            f.write(chunk_content)
                    
                    final_chapters.append({
                        'num': chunk_num,
                        'title': chunk_title,
                        'body': chunk_content,
                        'filename': chunk_filename,
                        # Don't set original_basename for chunks - let filename generation use decimal logic
                        'content_hash': self._generate_hash(chunk_content),
                        'file_size': len(chunk_content),
                        'has_images': False,
                        'is_chunk': True,
                        'chunk_info': {
                            'chunk_idx': chunk_idx,
                            'total_chunks': total_chunks,
                            'original_chapter': chapter_data['num']
                        }
                    })
            else:
                # Chapter is small enough, add as-is
                # Determine file extension based on format
                if is_html and self.file_path.lower().endswith('.pdf'):
                    if self.html2text_enabled or self.pdf_output_format == 'markdown':
                        file_ext = '.md'
                    else:
                        file_ext = '.html'
                else:
                    file_ext = '.txt'
                
                chapter_filename = f"section_{chapter_data['num']}{file_ext}"
                
                # Save original content to word_count folder (only if it doesn't exist)
                original_chunk_path = os.path.join(word_count_dir, chapter_filename)
                if not os.path.exists(original_chunk_path):
                    with open(original_chunk_path, 'w', encoding='utf-8') as f:
                        f.write(chapter_content)
                
                final_chapters.append({
                    'num': chapter_data['num'],  # Keep as integer for non-split chapters
                    'title': chapter_data['title'],
                    'body': chapter_content,
                    'filename': chapter_filename,
                    'original_basename': os.path.basename(self.file_path),  # Add original filename for .csv/.json/.txt detection
                    'content_hash': self._generate_hash(chapter_content),
                    'file_size': len(chapter_content),
                    'has_images': False,
                    'is_chunk': False
                })
        
        # Ensure we have at least one chapter
        if not final_chapters:
            # Fallback: create a single chapter with all content
            all_content = '\n\n'.join(ch['content'] for ch in raw_chapters if ch.get('content'))
            if not all_content and raw_chapters:
                all_content = raw_chapters[0].get('content', '')
                
            final_chapters.append({
                'num': 1,
                'title': 'Section 1',  # Changed from self.file_base
                'body': all_content or 'Empty file',
                'filename': 'section_1.txt',  # Changed to avoid using file_base
                'original_basename': os.path.basename(self.file_path),  # Add original filename for .csv/.json/.txt detection
                'content_hash': self._generate_hash(all_content or ''),
                'file_size': len(all_content or ''),
                'has_images': False,
                'is_chunk': False
            })
        
        # Copy images folder to output if it exists
        images_src = os.path.join(self.output_dir, 'images')
        if os.path.exists(images_src):
            # Copy to word_count folder as well for consistency
            word_count_images = os.path.join(word_count_dir, 'images')
            if not os.path.exists(word_count_images):
                shutil.copytree(images_src, word_count_images)
                print(f"📁 Copied images folder to word_count directory")
        
        return final_chapters
    
    
    def _generate_hash(self, content: str) -> str:
        """Generate hash for content"""
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def save_original_structure(self):
        """Save original text file structure info"""
        metadata = {
            'source_file': os.path.basename(self.file_path),
            'type': 'text',
            'encoding': 'utf-8'
        }
        
        metadata_path = os.path.join(self.output_dir, 'metadata.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    def create_output_structure(self, translated_chapters: List[Tuple[str, str]]) -> str:
        """Create output text/HTML/markdown file (or PDF) from translated chapters"""
        # Sort chapters by filename to ensure correct order
        sorted_chapters = sorted(translated_chapters, key=lambda x: x[0])
        
        # Determine output format based on source chapters
        is_html_format = False
        is_markdown_format = False
        if sorted_chapters:
            first_filename = sorted_chapters[0][0]
            if first_filename.endswith('.html'):
                is_html_format = True
            elif first_filename.endswith('.md'):
                is_markdown_format = True
        
        # Combine all content
        all_content = []
        for filename, content in sorted_chapters:
            # Add chapter separator if needed
            if len(all_content) > 0:
                if is_html_format:
                    all_content.append('<hr />\n\n')
                elif is_markdown_format:
                    all_content.append('\n\n---\n\n')
                else:
                    all_content.append('\n\n' + '='*50 + '\n\n')
            
            all_content.append(content)
        
        full_content = "".join(all_content)
        
        # Create output filename based on format
        if self.file_path.lower().endswith('.pdf'):
            # For PDF sources, check format
            if is_html_format:
                # Create HTML output
                output_filename = f"{self.file_base}_translated.html"
                output_path = os.path.join(self.output_dir, output_filename)
                
                # Wrap in full HTML document with CSS link
                html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self.file_base} - Translated</title>
    <link rel="stylesheet" href="styles.css">
</head>
<body>
{full_content}
</body>
</html>"""
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(html_doc)
                
                # Copy images folder to output directory if it exists
                images_src = os.path.join(self.output_dir, 'images')
                if os.path.exists(images_src):
                    print(f"🖼️ Images folder already in output directory")
                
                print(f"✅ Created translated HTML file: {output_filename}")
                return output_path
                
            elif is_markdown_format:
                # Create markdown output
                output_filename = f"{self.file_base}_translated.md"
                output_path = os.path.join(self.output_dir, output_filename)
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(full_content)
                
                # Copy images folder if it exists
                images_src = os.path.join(self.output_dir, 'images')
                if os.path.exists(images_src):
                    print(f"🖼️ Images folder already in output directory")
                
                print(f"✅ Created translated Markdown file: {output_filename}")
                return output_path
            
            else:
                # Try to create PDF from plain text
                output_filename = f"{self.file_base}_translated.pdf"
                output_path = os.path.join(self.output_dir, output_filename)
                try:
                    from pdf_extractor import create_pdf_from_text
                    if create_pdf_from_text(full_content, output_path):
                        print(f"✅ Created translated PDF file: {output_filename}")
                        return output_path
                    else:
                        print(f"⚠️ Failed to create PDF, falling back to text file")
                except Exception as e:
                    print(f"⚠️ Error creating PDF: {e}")
        
        # Fallback or default to text
        output_filename = f"{self.file_base}_translated.txt"
        output_path = os.path.join(self.output_dir, output_filename)
        
        # Write the translated text
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(full_content)
        
        print(f"✅ Created translated text file: {output_filename}")
        return output_path
