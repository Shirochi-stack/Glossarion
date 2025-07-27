# txt_processor.py
import os
import re
import json
from typing import List, Tuple, Dict
from bs4 import BeautifulSoup
from chapter_splitter import ChapterSplitter
from decimal import Decimal
import hashlib

class TextFileProcessor:
    """Process plain text files for translation"""
    
    def __init__(self, file_path: str, output_dir: str):
        self.file_path = file_path
        self.output_dir = output_dir
        self.file_base = os.path.splitext(os.path.basename(file_path))[0]
        
        # Initialize chapter splitter
        model_name = os.getenv("MODEL", "gpt-3.5-turbo")
        self.chapter_splitter = ChapterSplitter(model_name=model_name)
        
    def extract_chapters(self) -> List[Dict]:
        """Extract chapters from text file"""
        with open(self.file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # First, detect chapters in the content
        raw_chapters = self._detect_chapters(content)
        
        # Then, process each chapter for splitting if needed
        final_chapters = self._process_chapters_for_splitting(raw_chapters)
        
        print(f"📚 Extracted {len(final_chapters)} total chunks from {len(raw_chapters)} detected chapters")
        return final_chapters
    
    def _detect_chapters(self, content: str) -> List[Dict]:
        """Detect chapter boundaries in the text"""
        chapters = []
        
        # Chapter detection patterns
        chapter_patterns = [
            # English patterns
            (r'^Chapter\s+(\d+).*$', 'chapter'),
            (r'^CHAPTER\s+(\d+).*$', 'chapter'),
            (r'^Ch\.\s*(\d+).*$', 'chapter'),
            # Numbered sections
            (r'^(\d+)\.\s+(.*)$', 'numbered'),
            (r'^Part\s+(\d+).*$', 'part'),
            # Scene breaks (these don't have numbers)
            (r'^\*\s*\*\s*\*.*$', 'break'),
            (r'^---+.*$', 'break'),
            (r'^===+.*$', 'break'),
        ]
        
        # Find all chapter markers and their positions
        chapter_breaks = []
        lines = content.split('\n')
        
        for line_num, line in enumerate(lines):
            for pattern, pattern_type in chapter_patterns:
                match = re.match(pattern, line.strip())
                if match:
                    chapter_breaks.append({
                        'line_num': line_num,
                        'line': line,
                        'type': pattern_type,
                        'match': match
                    })
                    break
        
        if not chapter_breaks:
            # No chapter markers found, treat as single chapter
            print(f"No chapter markers found in {self.file_base}, treating as single document")
            # FIX: Use "Section 1" instead of filename to avoid number extraction issues
            chapters = [{
                'num': 1,
                'title': 'Section 1',  # Changed from self.file_base
                'content': content
            }]
        else:
            # Split content by chapter markers
            print(f"Found {len(chapter_breaks)} chapter markers in {self.file_base}")
            
            for i, chapter_break in enumerate(chapter_breaks):
                # Determine chapter number and title
                chapter_num, chapter_title = self._extract_chapter_info(chapter_break, i)
                
                # Get content for this chapter
                start_line = chapter_break['line_num'] + 1  # Start after the chapter marker
                
                # Find where this chapter ends
                if i < len(chapter_breaks) - 1:
                    end_line = chapter_breaks[i + 1]['line_num']
                else:
                    end_line = len(lines)
                
                # Extract chapter content
                chapter_lines = lines[start_line:end_line]
                chapter_content = '\n'.join(chapter_lines).strip()
                
                if chapter_content:  # Only add if there's actual content
                    chapters.append({
                        'num': chapter_num,
                        'title': chapter_title,
                        'content': chapter_content
                    })
        
        return chapters
    
    def _extract_chapter_info(self, chapter_break: Dict, index: int) -> Tuple[int, str]:
        """Extract chapter number and title from a chapter break"""
        if chapter_break['type'] == 'break':
            # Scene breaks don't have numbers
            chapter_num = index + 1
            chapter_title = f"Section {chapter_num}"
        else:
            # Try to extract number from match
            match_groups = chapter_break['match'].groups()
            if match_groups and match_groups[0]:  # Check if group exists AND is not empty
                try:
                    # Strip whitespace and check if it's a valid number
                    num_str = match_groups[0].strip()
                    if num_str:  # Only try to convert if not empty
                        chapter_num = int(num_str)
                        chapter_title = chapter_break['line'].strip()
                    else:
                        # Empty match group, use index
                        chapter_num = index + 1
                        chapter_title = chapter_break['line'].strip()
                except (ValueError, IndexError):
                    # Failed to convert to int, use index
                    chapter_num = index + 1
                    chapter_title = chapter_break['line'].strip()
            else:
                # No match groups or empty match
                chapter_num = index + 1
                chapter_title = chapter_break['line'].strip()
        
        return chapter_num, chapter_title
    
    def _process_chapters_for_splitting(self, raw_chapters: List[Dict]) -> List[Dict]:
        """Process chapters and split them if they exceed token limits"""
        final_chapters = []
        
        # Calculate based on OUTPUT token limits
        max_output_tokens = int(os.getenv("MAX_OUTPUT_TOKENS", "8192"))
        compression_factor = float(os.getenv("COMPRESSION_FACTOR", "0.8"))
        safety_margin_output = 500
        
        # Calculate chunk size based on output limit
        available_tokens = int((max_output_tokens - safety_margin_output) / compression_factor)
        available_tokens = max(available_tokens, 1000)
        
        print(f"📊 Text file chunk size: {available_tokens:,} tokens (based on {max_output_tokens:,} output limit, compression: {compression_factor})")
        
        for chapter_data in raw_chapters:
            # Convert chapter content to HTML format
            chapter_html = self._text_to_html(chapter_data['content'])
            chapter_tokens = self.chapter_splitter.count_tokens(chapter_html)
            
            if chapter_tokens > available_tokens:
                # Chapter needs splitting
                print(f"Chapter {chapter_data['num']} ({chapter_data['title']}) has {chapter_tokens} tokens, splitting...")
                
                chunks = self.chapter_splitter.split_chapter(chapter_html, available_tokens)
                
                # Add each chunk as a separate chapter
                for chunk_html, chunk_idx, total_chunks in chunks:
                    chunk_title = chapter_data['title']
                    if total_chunks > 1:
                        chunk_title = f"{chapter_data['title']} (Part {chunk_idx}/{total_chunks})"
                    
                    # Create float chapter numbers for chunks: 1.0, 1.1, 1.2, etc.
                    chunk_num = round(chapter_data['num'] + (chunk_idx - 1) * 0.1, 1)
                    
                    final_chapters.append({
                        'num': chunk_num,
                        'title': chunk_title,
                        'body': chunk_html,
                        'filename': f"section_{int(chapter_data['num'])}_part{chunk_idx}.txt",  # Changed to avoid using file_base
                        'content_hash': self._generate_hash(chunk_html),
                        'file_size': len(chunk_html),
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
                final_chapters.append({
                    'num': chapter_data['num'],  # Keep as integer for non-split chapters
                    'title': chapter_data['title'],
                    'body': chapter_html,
                    'filename': f"section_{chapter_data['num']}.txt",  # Changed to avoid using file_base
                    'content_hash': self._generate_hash(chapter_html),
                    'file_size': len(chapter_html),
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
                'body': self._text_to_html(all_content or 'Empty file'),
                'filename': 'section_1.txt',  # Changed to avoid using file_base
                'content_hash': self._generate_hash(all_content or ''),
                'file_size': len(all_content or ''),
                'has_images': False,
                'is_chunk': False
            })
        
        return final_chapters
    
    def _text_to_html(self, text: str) -> str:
        """Convert plain text to HTML format"""
        # Escape HTML characters
        text = text.replace('&', '&amp;')
        text = text.replace('<', '&lt;')
        text = text.replace('>', '&gt;')
        
        # Split into paragraphs
        paragraphs = text.split('\n\n')
        
        # Wrap each paragraph in <p> tags
        html_parts = []
        for para in paragraphs:
            para = para.strip()
            if para:
                # Check if it's a chapter heading
                if re.match(r'^(Chapter|CHAPTER|Ch\.|Part)\s+\d+', para):
                    html_parts.append(f'<h1>{para}</h1>')
                else:
                    # Replace single newlines with <br> within paragraphs
                    para = para.replace('\n', '<br>\n')
                    html_parts.append(f'<p>{para}</p>')
        
        # Create a simple HTML structure
        html = f"""<html>
<head>
    <title>{self.file_base}</title>
    <meta charset="utf-8"/>
</head>
<body>
    {''.join(html_parts)}
</body>
</html>"""
        
        return html
    
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
        """Create output text file from translated chapters"""
        # Sort chapters by filename to ensure correct order
        sorted_chapters = sorted(translated_chapters, key=lambda x: x[0])
        
        # Combine all content
        all_content = []
        for filename, content in sorted_chapters:
            # Extract text from HTML
            soup = BeautifulSoup(content, 'html.parser')
            text_content = soup.get_text()
            
            # Add chapter separator if needed
            if len(all_content) > 0:
                all_content.append('\n\n' + '='*50 + '\n\n')
            
            all_content.append(text_content)
        
        # Create output filename
        output_filename = f"{self.file_base}_translated.txt"
        output_path = os.path.join(self.output_dir, output_filename)
        
        # Write the translated text
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(''.join(all_content))
        
        print(f"✅ Created translated text file: {output_filename}")
        return output_path
