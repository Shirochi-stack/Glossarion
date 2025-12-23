# extract_glossary_from_txt.py
import os
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
    
    # Translation-aligned safe chunk budget
    compression_factor = float(os.getenv("GLOSSARY_COMPRESSION_FACTOR", os.getenv("COMPRESSION_FACTOR", "1.0")))
    raw_output_env = os.getenv("GLOSSARY_MAX_OUTPUT_TOKENS", os.getenv("MAX_OUTPUT_TOKENS", "65536"))
    try:
        effective_output = int(str(raw_output_env).strip())
    except Exception:
        effective_output = 65536
    if effective_output <= 0:
        effective_output = 65536
    safety_margin_output = 500
    available_tokens = int((effective_output - safety_margin_output) / max(compression_factor, 0.01))
    available_tokens = max(available_tokens, 1000)
    chapter_split_enabled = os.getenv("GLOSSARY_ENABLE_CHAPTER_SPLIT", "1") == "1"
    print(f"📊 Chapter chunk budget: {available_tokens:,} tokens (output limit {effective_output:,}, compression {compression_factor})")
    
    text_chapters = []
    
    for idx, chapter in enumerate(chapters):
        # Check if chapter needs splitting
        chapter_tokens = chapter_splitter.count_tokens(chapter['body'])
        
        if chapter_split_enabled and chapter_tokens > available_tokens:
            print(f"Chapter {idx+1} has {chapter_tokens} tokens, splitting into smaller chunks (budget {available_tokens})...")
            
            # Use ChapterSplitter to split the HTML content
            # Pass filename for content type detection
            chunks = chapter_splitter.split_chapter(chapter['body'], available_tokens, filename=txt_path)
            
            # Extract text from each chunk
            for chunk_html, chunk_idx, total_chunks in chunks:
                soup = BeautifulSoup(chunk_html, 'html.parser')
                text = soup.get_text(strip=True)
                if text:
                    text_chapters.append(text)
                    print(f"  Added chunk {chunk_idx}/{total_chunks} ({chapter_splitter.count_tokens(text)} tokens)")
        else:
            # Chapter is small enough or splitting disabled, extract text as-is
            soup = BeautifulSoup(chapter['body'], 'html.parser')
            text = soup.get_text(strip=True)
            if text:
                text_chapters.append(text)
    
    print(f"Total text chunks for glossary extraction: {len(text_chapters)}")
    return text_chapters
