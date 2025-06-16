# extract_glossary_from_txt.py
import os
import json
from typing import List
from txt_processor import TextFileProcessor
from chapter_splitter import ChapterSplitter
from bs4 import BeautifulSoup

def extract_chapters_from_txt(txt_path: str) -> List[str]:
    """Extract chapters from text file for glossary extraction"""
    processor = TextFileProcessor(txt_path, os.path.dirname(txt_path))
    chapters = processor.extract_chapters()
    
    # Initialize chapter splitter
    model_name = os.getenv("MODEL", "gpt-3.5-turbo")
    chapter_splitter = ChapterSplitter(model_name=model_name)
    
    # Get max tokens from environment
    max_input_tokens = int(os.getenv("MAX_INPUT_TOKENS", "1000000"))
    
    # Calculate available tokens (leaving room for system prompt and context)
    system_prompt_size = 2000  # Estimate for glossary system prompt
    context_size = 5000  # Estimate for context history
    safety_margin = 1000
    available_tokens = max_input_tokens - system_prompt_size - context_size - safety_margin
    
    text_chapters = []
    
    for idx, chapter in enumerate(chapters):
        # Check if chapter needs splitting
        chapter_tokens = chapter_splitter.count_tokens(chapter['body'])
        
        if chapter_tokens > available_tokens:
            print(f"Chapter {idx+1} has {chapter_tokens} tokens, splitting into smaller chunks...")
            
            # Use ChapterSplitter to split the HTML content
            chunks = chapter_splitter.split_chapter(chapter['body'], available_tokens)
            
            # Extract text from each chunk
            for chunk_html, chunk_idx, total_chunks in chunks:
                soup = BeautifulSoup(chunk_html, 'html.parser')
                text = soup.get_text(strip=True)
                if text:
                    text_chapters.append(text)
                    print(f"  Added chunk {chunk_idx}/{total_chunks} ({chapter_splitter.count_tokens(text)} tokens)")
        else:
            # Chapter is small enough, extract text as-is
            soup = BeautifulSoup(chapter['body'], 'html.parser')
            text = soup.get_text(strip=True)
            if text:
                text_chapters.append(text)
    
    print(f"Total text chunks for glossary extraction: {len(text_chapters)}")
    return text_chapters