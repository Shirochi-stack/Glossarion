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
        
        # Treat entire file as single document - no chapter detection
        raw_chapters = [{
            'num': 1,
            'title': self.file_base,
            'content': content
        }]
        
        # Process for splitting if needed
        final_chapters = self._process_chapters_for_splitting(raw_chapters)
        
        print(f"📚 Extracted {len(final_chapters)} total chunks")
        return final_chapters
    
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
            # Keep content as plain text
            chapter_content = chapter_data['content']
            chapter_tokens = self.chapter_splitter.count_tokens(chapter_content)
            
            if chapter_tokens > available_tokens:
                # Chapter needs splitting
                print(f"Chapter {chapter_data['num']} ({chapter_data['title']}) has {chapter_tokens} tokens, splitting...")
                
                chunks = self.chapter_splitter.split_chapter(chapter_content, available_tokens)
                
                # Add each chunk as a separate chapter
                for chunk_content, chunk_idx, total_chunks in chunks:
                    chunk_title = chapter_data['title']
                    if total_chunks > 1:
                        chunk_title = f"{chapter_data['title']} (Part {chunk_idx}/{total_chunks})"
                    
                    # Create float chapter numbers for chunks: 1.0, 1.1, 1.2, etc.
                    chunk_num = round(chapter_data['num'] + (chunk_idx - 1) * 0.1, 1)
                    
                    final_chapters.append({
                        'num': chunk_num,
                        'title': chunk_title,
                        'body': chunk_content,
                        'filename': f"section_{int(chapter_data['num'])}_part{chunk_idx}.txt",  # Changed to avoid using file_base
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
                final_chapters.append({
                    'num': chapter_data['num'],  # Keep as integer for non-split chapters
                    'title': chapter_data['title'],
                    'body': chapter_content,
                    'filename': f"section_{chapter_data['num']}.txt",  # Changed to avoid using file_base
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
                'content_hash': self._generate_hash(all_content or ''),
                'file_size': len(all_content or ''),
                'has_images': False,
                'is_chunk': False
            })
        
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
        """Create output text file from translated chapters"""
        # Sort chapters by filename to ensure correct order
        sorted_chapters = sorted(translated_chapters, key=lambda x: x[0])
        
        # Combine all content
        all_content = []
        for filename, content in sorted_chapters:
            # Content is already plain text, no need to parse HTML
            text_content = content
            
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
