# extract_glossary_from_txt.py
import os
from typing import List
from txt_processor import TextFileProcessor
from chapter_splitter import ChapterSplitter
from bs4 import BeautifulSoup

def extract_chapters_from_txt(txt_path: str) -> List[str]:
    """Extract chapters from text file for glossary extraction"""
    # Use glossary output directory so split_glossary.cache doesn't collide
    # with the translation split.cache
    output_path = os.getenv("OUTPUT_PATH", "")
    if output_path:
        glossary_output_dir = os.path.dirname(output_path)
    else:
        glossary_output_dir = os.path.dirname(txt_path)
    os.makedirs(glossary_output_dir, exist_ok=True)
    
    # Include input filename in cache suffix so each file gets its own cache
    # e.g. split_glossary_MyNovel.cache
    file_base = os.path.splitext(os.path.basename(txt_path))[0]
    processor = TextFileProcessor(txt_path, glossary_output_dir, cache_suffix=f'_glossary_{file_base}')
    
    # For PDF files, use subprocess extraction to prevent GUI lag
    if txt_path.lower().endswith('.pdf') and os.getenv("USE_ASYNC_CHAPTER_EXTRACTION", "0") == "1":
        try:
            import json as _json
            from pdf_extraction_manager import PdfExtractionManager
            
            _pdf_ext_config = {
                "pdf_path": txt_path,
                "output_dir": glossary_output_dir,
                "render_mode": os.getenv("PDF_RENDER_MODE", "xhtml").lower(),
                "extract_images": os.getenv("PDF_EXTRACT_IMAGES", "1") == "1",
                "generate_css": os.getenv("PDF_GENERATE_CSS", "1") == "1",
                "html2text": os.getenv("USE_HTML2TEXT", "0") == "1",
                "css_override_path": os.getenv("EPUB_CSS_OVERRIDE_PATH", "").strip(),
                "attach_css_enabled": os.getenv("ATTACH_CSS_TO_CHAPTERS", "0") == "1"
            }
            _config_path = os.path.join(glossary_output_dir, '_pdf_glossary_extraction_config.json')
            with open(_config_path, 'w', encoding='utf-8') as f:
                _json.dump(_pdf_ext_config, f, ensure_ascii=False)
            
            _mgr = PdfExtractionManager(log_callback=print)
            _result = _mgr.extract_pdf_sync(_config_path, timeout=600)
            
            if _result and _result.get("success"):
                _result_path = _result.get("result_path")
                if _result_path and os.path.exists(_result_path):
                    with open(_result_path, 'r', encoding='utf-8') as f:
                        _ext_data = _json.load(f)
                    
                    _content = _ext_data.get("content", [])
                    render_mode = os.getenv("PDF_RENDER_MODE", "xhtml").lower()
                    
                    if _ext_data.get("is_page_list") and isinstance(_content, list):
                        raw_chapters = []
                        for item in _content:
                            page_num = item[0] if isinstance(item, list) else item
                            page_html = item[1] if isinstance(item, list) else ""
                            raw_chapters.append({
                                'num': page_num,
                                'title': f"{file_base} - Page {page_num}",
                                'content': page_html,
                                'is_html': True,
                                'images_info': {},
                                'has_images': True if render_mode == 'image' else False,
                                'image_count': 1 if render_mode == 'image' else 0
                            })
                        chapters = processor._process_chapters_for_splitting(raw_chapters)
                    else:
                        chapters = [{'num': 1, 'title': file_base, 'body': _content if isinstance(_content, str) else '', 'filename': 'section_1.html', 'source_file': txt_path, 'content_hash': '', 'file_size': 0, 'has_images': False, 'image_count': 0, 'is_chunk': False}]
                    
                    # Clean up temp files
                    for p in [_config_path, _result_path]:
                        try:
                            os.remove(p)
                        except Exception:
                            pass
                else:
                    print("⚠️ Subprocess extraction succeeded but result file not found, falling back to in-process...")
                    chapters = processor.extract_chapters()
            else:
                _err = _result.get("error", "Unknown") if _result else "No result"
                print(f"⚠️ Subprocess PDF extraction failed ({_err}), falling back to in-process...")
                chapters = processor.extract_chapters()
        except Exception as e:
            print(f"⚠️ Subprocess PDF extraction error ({e}), falling back to in-process...")
            chapters = processor.extract_chapters()
    else:
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

    
    text_chapters = []
    
    for idx, chapter in enumerate(chapters):
        # Check if chapter needs splitting
        chapter_tokens = chapter_splitter.count_tokens(chapter['body'])
        
        if chapter_split_enabled and chapter_tokens > available_tokens:

            
            # Use ChapterSplitter to split the HTML content
            # Pass filename for content type detection
            chunks = chapter_splitter.split_chapter(chapter['body'], available_tokens, filename=txt_path)
            
            # Extract text from each chunk
            for chunk_html, chunk_idx, total_chunks in chunks:
                soup = BeautifulSoup(chunk_html, 'html.parser')
                text = soup.get_text(strip=True)
                if text:
                    text_chapters.append(text)

        else:
            # Chapter is small enough or splitting disabled, extract text as-is
            soup = BeautifulSoup(chapter['body'], 'html.parser')
            text = soup.get_text(strip=True)
            if text:
                text_chapters.append(text)
    
    print(f"Total text chunks for glossary extraction: {len(text_chapters)}")
    return text_chapters
